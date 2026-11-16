"""Crossfit player-node live-team corrector with anti-flicker apply.

Replaces the sm_v2_4 control-design corrector head; keeps boosted D + boundary
ensemble (v2.3 stack) as the baseline being corrected.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from boosted_dead_model import boosted_logits
from eval_external import write_csv
from state_machine import _merge_short_spells
from pipeline_090 import v23_team_logits
from node_ownership_features import (
    FEATURE_VERSION, design_node_corrector_features)


HARD_WEIGHT = 3.0
DEFAULT_MARGIN = 1.5
DEFAULT_WINDOW = 5
DEFAULT_MIN_ISLAND = 12
MERGE_MINIMUM = 18


def training_matrix(games, stride=5, hard_weight=HARD_WEIGHT):
    xs, ys, ws = [], [], []
    for game in games:
        data = load_match(game)
        f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
        baseline = predict_version(f, graded, (2, 3))
        v24 = predict_version(f, graded, (2, 4))
        logits = v23_team_logits(f, graded, nodes, baseline)
        x = design_node_corrector_features(f, baseline, logits, nodes, graded)
        take = graded & f['ball_present'] & np.isin(gt, ['H', 'A']) & np.isin(baseline, ['H', 'A'])
        idx = np.flatnonzero(take)[::stride]
        hard = (v24[idx] != gt[idx]) | (baseline[idx] != gt[idx])
        xs.append(x[idx])
        ys.append((gt[idx] == 'H').astype(float))
        ws.append(np.where(hard, hard_weight, 1.).astype(np.float64))
        print('node features', game, len(idx), 'hard', int(hard.sum()), flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def train(games, seed=73):
    x, y, w = training_matrix(games)
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=250, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=1., max_bins=127,
        early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=w)
    return model


def export_model(estimator, path, training_matches):
    trees = []
    for iteration in estimator._predictors:
        nodes = iteration[0].nodes
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf',
                      missing_left='missing_go_to_left')
        trees.append({key: nodes[source].tolist() for key, source in fields.items()})
    Path(path).write_text(json.dumps(dict(
        feature_version=FEATURE_VERSION,
        baseline=float(estimator._baseline_prediction[0, 0]),
        n_features=int(estimator.n_features_in_),
        training_matches=list(training_matches), trees=trees), separators=(',', ':')) + '\n')


def load_packed(path):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported node corrector schema')
    arrays = [{k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
               for k, v in tree.items()} for tree in data['trees']]
    return dict(baseline=float(data['baseline']), trees=arrays)


def corrector_logits(features, graded, nodes, baseline, model_path):
    team_logits = v23_team_logits(features, graded, nodes, baseline)
    x = design_node_corrector_features(
        features, baseline, team_logits, nodes, graded)
    return boosted_logits(x, load_packed(model_path))


def _suppress_short_islands(flip_mask, labels, graded, min_island):
    """Drop flip islands shorter than min_island unless touching D/X barrier."""
    if min_island <= 1 or not np.any(flip_mask):
        return flip_mask
    out = flip_mask.copy()
    n = len(flip_mask)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        length = j - i
        if length < min_island:
            left_barrier = (i == 0) or (not graded[i - 1]) or labels[i - 1] in ('D', 'X')
            right_barrier = (j == n) or (not graded[j]) or labels[j] in ('D', 'X')
            if not (left_barrier or right_barrier):
                out[i:j] = False
        i = j
    return out


def apply_node_corrector(features, graded, nodes, model_path,
                         margin=DEFAULT_MARGIN, window=DEFAULT_WINDOW,
                         min_island=DEFAULT_MIN_ISLAND, minimum=MERGE_MINIMUM,
                         baseline=None):
    if baseline is None:
        baseline = predict_version(features, graded, (2, 3))
    labels = np.asarray(baseline).copy()
    corr = corrector_logits(features, graded, nodes, labels, model_path)
    present = np.asarray(features['ball_present'], bool)
    live = graded & present & np.isin(labels, ['H', 'A'])
    corr_home = corr >= 0
    base_home = labels == 'H'
    propose = live & (corr_home != base_home) & (np.abs(corr) >= float(margin))
    # Window agreement on proposed team (corr sign) among live candidates.
    propose_lab = np.where(corr_home, 'H', 'A')
    # Only enforce agreement among frames that are also proposed (or live);
    # require neighboring frames in ±W to share the same proposed label AND
    # also want the flip (or already match that label).
    want = propose.copy()
    if window > 0:
        agree = np.ones(len(labels), bool)
        for d in range(1, int(window) + 1):
            same_l = np.ones(len(labels), bool)
            same_r = np.ones(len(labels), bool)
            same_l[d:] = propose_lab[d:] == propose_lab[:-d]
            same_r[:-d] = propose_lab[:-d] == propose_lab[d:]
            # neighbors must also be confident flip-or-already-that-team
            conf_l = np.ones(len(labels), bool)
            conf_r = np.ones(len(labels), bool)
            neighbor_ok_l = (np.abs(corr) >= float(margin)) | (labels == propose_lab)
            neighbor_ok_r = neighbor_ok_l
            conf_l[d:] = neighbor_ok_l[:-d]
            conf_r[:-d] = neighbor_ok_r[d:]
            agree &= same_l & same_r & conf_l & conf_r
        want &= agree
    want = _suppress_short_islands(want, labels, graded, int(min_island))
    labels[want] = np.where(corr_home[want], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum), int(want.sum()), corr


def visible_team_acc(gt, pred, features):
    present = np.asarray(features['ball_present'], bool)
    live = present & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
    return float(np.mean(pred[live] == gt[live])) if live.any() else float('nan')


def nested_tune(train_games, model_path, priors, seed=0):
    """Tune margin/W/min_island on a nested holdout of the 9 train matches."""
    margins = sorted({priors['margin'], 1.0, 1.5, 2.0})
    windows = sorted({priors['window'], 3, 5})
    islands = sorted({priors['min_island'], 8, 12, 18})
    # Nested holdout: last train game (stable, not the outer holdout).
    nested_hold = train_games[-1]
    fit_games = train_games[:-1]
    # Model is already fit on all outer train games; tune apply hyperparams only
    # on nested_hold predictions from that model (no refit — keep nested cheap).
    data = load_match(nested_hold)
    baseline = predict_version(data['features'], data['graded'], (2, 3))
    best = None
    for margin in margins:
        for window in windows:
            for min_island in islands:
                pred, _, _ = apply_node_corrector(
                    data['features'], data['graded'], data['nodes'], model_path,
                    margin=margin, window=window, min_island=min_island,
                    baseline=baseline)
                metrics = score_natural(data['gt'], pred, data['features'])
                acc = visible_team_acc(data['gt'], pred, data['features'])
                score = (metrics['spell_f1_mean'], acc)
                cand = dict(margin=margin, window=window, min_island=min_island,
                            spell=metrics['spell_f1_mean'], vis_acc=acc)
                if best is None or score > (best['spell'], best['vis_acc']):
                    best = cand
    best['nested_hold'] = nested_hold
    best['fit_games_unused'] = fit_games  # documented; model uses full outer train
    return best


def load_priors(path=None):
    path = Path(path or (REPORTS / 'wrong_team_audit.json'))
    if path.exists():
        data = json.loads(path.read_text())
        return data.get('priors', dict(
            margin=DEFAULT_MARGIN, window=DEFAULT_WINDOW, min_island=DEFAULT_MIN_ISLAND))
    return dict(margin=DEFAULT_MARGIN, window=DEFAULT_WINDOW, min_island=DEFAULT_MIN_ISLAND)


def crossfit(matches=None, priors=None):
    matches = matches or MATCHES
    priors = priors or load_priors()
    fold_dir = REPORTS / 'crossfit_node_corrector'
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tuned = []
    for holdout in matches:
        path = fold_dir / ('node_' + holdout + '.json')
        train_games = fold_train_games(holdout)
        if not path.exists():
            print('train fold', holdout, flush=True)
            model = train(train_games)
            joblib.dump(model, fold_dir / ('node_' + holdout + '.joblib'))
            export_model(model, path, train_games)
        params = nested_tune(train_games, path, priors)
        tuned.append(dict(holdout=holdout, **params))
        print('tune', holdout, params, flush=True)
        data = load_match(holdout)
        pred, nflips, corr = apply_node_corrector(
            data['features'], data['graded'], data['nodes'], path,
            margin=params['margin'], window=params['window'],
            min_island=params['min_island'])
        # Raw corrector (margin only, no window/island) for diagnostics
        raw, raw_flips, _ = apply_node_corrector(
            data['features'], data['graded'], data['nodes'], path,
            margin=params['margin'], window=0, min_island=0)
        metrics = score_natural(data['gt'], pred, data['features'])
        v24 = predict_version(data['features'], data['graded'], (2, 4))
        m24 = score_natural(data['gt'], v24, data['features'])
        acc = visible_team_acc(data['gt'], pred, data['features'])
        acc24 = visible_team_acc(data['gt'], v24, data['features'])
        acc_raw = visible_team_acc(data['gt'], raw, data['features'])
        rows.append(dict(
            game=holdout, candidate='node_corrector',
            flips=nflips, raw_flips=raw_flips,
            visible_team_acc=acc, visible_team_acc_raw=acc_raw,
            visible_team_acc_v24=acc24,
            spell_f1_v24=m24['spell_f1_mean'],
            margin=params['margin'], window=params['window'],
            min_island=params['min_island'],
            **metrics))
        print(holdout, 'flips', nflips, 'vis', round(acc, 4),
              'spell', round(metrics['spell_f1_mean'], 4),
              'v24_spell', round(m24['spell_f1_mean'], 4), flush=True)
    return rows, tuned


def gate_decision(rows):
    """Promote only if mean spell+vis ≥ v2.4 and no match loses >0.02 spell."""
    spell = np.mean([r['spell_f1_mean'] for r in rows])
    spell24 = np.mean([r['spell_f1_v24'] for r in rows])
    vis = np.mean([r['visible_team_acc'] for r in rows])
    vis24 = np.mean([r['visible_team_acc_v24'] for r in rows])
    deltas = [r['spell_f1_mean'] - r['spell_f1_v24'] for r in rows]
    worst = float(np.min(deltas))
    ok = (spell >= spell24 - 1e-12) and (vis >= vis24 - 1e-12) and (worst >= -0.02)
    return dict(
        pass_gate=bool(ok),
        mean_spell=float(spell), mean_spell_v24=float(spell24),
        mean_vis=float(vis), mean_vis_v24=float(vis24),
        worst_spell_delta=worst,
        per_match_spell_delta=dict((r['game'], float(r['spell_f1_mean'] - r['spell_f1_v24']))
                                   for r in rows),
    )


def majority_apply_params(tuned):
    """Ship-time defaults: mode of nested-tuned params across folds."""
    def mode(vals):
        vals = list(vals)
        return max(set(vals), key=vals.count)
    return dict(
        margin=float(mode(t['margin'] for t in tuned)),
        window=int(mode(t['window'] for t in tuned)),
        min_island=int(mode(t['min_island'] for t in tuned)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--priors', default=str(REPORTS / 'wrong_team_audit.json'))
    args = parser.parse_args()
    priors = load_priors(args.priors)
    print('priors', priors, flush=True)
    if args.train_all:
        model = train(MATCHES)
        out = REPORTS / 'node_corrector_all10'
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out / 'model.joblib')
        export_model(model, out / 'node_corrector_v1.json', MATCHES)
        print('wrote', out / 'node_corrector_v1.json', flush=True)
    if args.crossfit or not args.train_all:
        rows, tuned = crossfit(priors=priors)
        write_csv(REPORTS / 'crossfit_node_corrector.csv', rows)
        decision = gate_decision(rows)
        ship_params = majority_apply_params(tuned)
        summary = dict(
            aggregate=summarize_rows(rows),
            gate=decision,
            ship_params=ship_params,
            tuned=tuned,
            rows=rows,
            priors=priors,
        )
        Path(REPORTS / 'crossfit_node_corrector.json').write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(dict(gate=decision, ship_params=ship_params,
                              aggregate=summary['aggregate']), indent=2))


if __name__ == '__main__':
    main()
