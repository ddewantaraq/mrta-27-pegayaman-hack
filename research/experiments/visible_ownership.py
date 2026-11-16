"""Visible-ball team ownership classifier (cross-fitted HGB).

Trains on frames where the ball is present and GT is H/A. Exports NumPy trees.
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
from eval_external import write_csv
from state_machine import _merge_short_spells
from boosted_dead_model import load_boosted_model, boosted_logits

FEATURE_VERSION = 'visible_own_v1_41'
FEATURE_DIM = 41


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def design_visible_features(features):
    """48-d per-frame design for visible-ball team classification."""
    present = np.asarray(features['ball_present'], bool)
    period = features.get('period')
    bx = np.nan_to_num(features['ball_x_m'], nan=0.)
    by = np.nan_to_num(features['ball_y_m'], nan=0.)
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    dh = np.nan_to_num(features['min_dist_home'], nan=99.)
    da = np.nan_to_num(features['min_dist_away'], nan=99.)
    nd = np.nan_to_num(features['nearest_dist'], nan=99.)
    nearest = np.asarray(features['nearest_team'], dtype=np.float64)
    majority = np.asarray(features['majority_team'], dtype=np.float64)
    nh = features['n_near_home'].astype(np.float64)
    na = features['n_near_away'].astype(np.float64)
    pz_team = features['pz_team'].astype(np.float64)
    duel = features['pz_duel'].astype(np.float64)
    motion = np.asarray(features['player_motion'], dtype=np.float64)
    vx = _delta(bx, period)
    vy = _delta(by, period)
    ax = _delta(vx, period)
    ay = _delta(vy, period)
    accel = np.sqrt(ax * ax + ay * ay)
    # Closing speeds toward each team's nearest (approx via dist delta).
    close_h = -_delta(dh, period)
    close_a = -_delta(da, period)
    cols = [
        bx, by, speed, vx, vy, ax, ay, accel,
        dh, da, nd, dh - da, np.exp(-dh / 3.), np.exp(-da / 3.),
        nearest, majority, nh, na, nh - na, pz_team, duel,
        present.astype(np.float64),
        close_h, close_a, close_h - close_a,
    ]
    # First 6 motion channels (team aggregate kinematics if present).
    for i in range(min(12, motion.shape[1])):
        cols.append(np.nan_to_num(motion[:, i], nan=0.))
    # Lag-1 nearest and dist for short memory.
    nearest_lag = np.roll(nearest, 1); nearest_lag[0] = nearest[0]
    nd_lag = np.roll(nd, 1); nd_lag[0] = nd[0]
    cols.extend([nearest_lag, nd_lag, nearest - nearest_lag, nd - nd_lag])
    x = np.column_stack(cols).astype(np.float64)
    assert x.shape[1] == FEATURE_DIM, x.shape
    # Mask missing ball as NaN so trees can route on missingness if used later.
    x[~present] = np.nan
    return x


def training_matrix(games, stride=5):
    xs, ys = [], []
    for game in games:
        data = load_match(game)
        f, gt, graded = data['features'], data['gt'], data['graded']
        x = design_visible_features(f)
        take = graded & f['ball_present'] & np.isin(gt, ['H', 'A'])
        idx = np.flatnonzero(take)[::stride]
        xs.append(x[idx])
        ys.append((gt[idx] == 'H').astype(np.float64))
        print('visible features', game, len(idx), flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def train_visible(games, seed=73):
    x, y = training_matrix(games)
    model = HistGradientBoostingClassifier(
        learning_rate=.08, max_iter=200, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=5., max_bins=63,
        early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model


def export_trees(estimator, path, training_matches, correction=0.):
    trees = []
    for iteration in estimator._predictors:
        nodes = iteration[0].nodes
        if np.any(nodes['is_categorical']):
            raise ValueError('Only numeric trees can be exported')
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf',
                      missing_left='missing_go_to_left')
        trees.append({key: nodes[source].tolist() for key, source in fields.items()})
    payload = dict(
        feature_version=FEATURE_VERSION,
        baseline=float(estimator._baseline_prediction[0, 0]) + correction,
        prior_log_odds_correction=correction,
        training_matches=list(training_matches),
        trees=trees,
    )
    Path(path).write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    return path


def predict_logits_visible(features, model_path):
    from boosted_dead_model import boosted_logits
    data = json.loads(Path(model_path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported visible ownership schema')
    arrays = []
    for tree in data['trees']:
        converted = {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                     for k, v in tree.items()}
        arrays.append(converted)
    packed = dict(baseline=float(data['baseline']), trees=arrays)
    return boosted_logits(design_visible_features(features), packed)


def apply_visible_to_baseline(features, baseline, graded, logits, minimum=18,
                              confidence=0.0):
    """Replace baseline H/A on visible frames using temporally decoded logits."""
    from dead_ball_model import decode_dead_logits
    labels = np.asarray(baseline).copy()
    live = graded & np.isin(labels, ['H', 'A'])
    present = np.asarray(features['ball_present'], bool)
    # Decode over live spans; then keep only confident visible overrides.
    home = decode_dead_logits(np.nan_to_num(logits, nan=0.), live,
                              features.get('period'))
    use = live & present
    if confidence > 0:
        use = use & (np.abs(np.nan_to_num(logits, nan=0.)) >= float(confidence))
    labels[use] = np.where(home[use], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)


def make_predict_fn(model_path):
    def predict_fn(features, graded, nodes):
        base = predict_version(features, graded, (2, 3))
        logits = predict_logits_visible(features, model_path)
        return apply_visible_to_baseline(features, base, graded, logits)
    return predict_fn


def crossfit_visible(matches=None):
    matches = matches or MATCHES
    rows = []
    fold_dir = REPORTS / 'crossfit_visible'
    fold_dir.mkdir(parents=True, exist_ok=True)
    for holdout in matches:
        train_games = fold_train_games(holdout)
        path = fold_dir / ('visible_' + holdout + '.json')
        joblib_path = fold_dir / ('visible_' + holdout + '.joblib')
        if not path.exists():
            model = train_visible(train_games)
            joblib.dump(model, joblib_path)
            export_trees(model, path, train_games)
        pred_fn = make_predict_fn(path)
        data = load_match(holdout)
        pred = pred_fn(data['features'], data['graded'], data['nodes'])
        metrics = score_natural(data['gt'], pred, data['features'])
        # Visible-frame team accuracy
        gt = data['gt']
        present = data['features']['ball_present']
        live = present & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
        vis_acc = float(np.mean(pred[live] == gt[live])) if live.any() else float('nan')
        rows.append(dict(game=holdout, candidate='visible_v23', protocol='natural',
                         visible_team_acc=vis_acc, **metrics))
        print('holdout', holdout, 'spell', metrics['spell_f1_mean'],
              'vis_acc', vis_acc, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true',
                        help='train on all 10 matches for shipping')
    parser.add_argument('--out-model', default=str(REPORTS / 'visible_ownership_all10.json'))
    args = parser.parse_args()
    if args.train_all:
        model = train_visible(MATCHES)
        joblib.dump(model, REPORTS / 'visible_ownership_all10.joblib')
        export_trees(model, args.out_model, MATCHES)
        print('trained all-10', args.out_model, flush=True)
    if args.crossfit or not args.train_all:
        rows = crossfit_visible()
        write_csv(REPORTS / 'crossfit_visible.csv', rows)
        summary = summarize_rows(rows)
        Path(REPORTS / 'crossfit_visible.json').write_text(json.dumps(dict(
            aggregate=summary, rows=rows), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
