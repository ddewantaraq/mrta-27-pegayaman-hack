"""Insert-only contested H↔A switches on frozen sm_v2_3.

No delete / no retarget. Propose splits only at contested peaks with:
  - peak far from existing pred switches
  - ensemble logit polarity disagrees across the peak (left vs right)
  - both resulting islands >= min_island
  - top-K by contested*disagreement score

Nested-tune K / min_score / min_island / sep / window on train folds only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from eval_external import write_csv
from state_machine import _merge_short_spells
from pipeline_090 import v23_team_logits
from contested_band_features import contested_score, contested_peaks, ha_switches


DEFAULT_K = 8
DEFAULT_MIN_SCORE = 1.0
DEFAULT_MIN_ISLAND = 25
DEFAULT_SEP = 25
DEFAULT_WINDOW = 12
DEFAULT_MIN_DISAGREE = 0.35
MERGE_MINIMUM = 18


def _spell_bounds(labels, graded, i):
    team = labels[i]
    a = i
    while a > 0 and graded[a - 1] and labels[a - 1] == team:
        a -= 1
    b = i + 1
    n = len(labels)
    while b < n and graded[b] and labels[b] == team:
        b += 1
    return a, b


def _mean_logit(logits, lo, hi):
    lo = max(0, lo)
    hi = min(len(logits), hi)
    if hi <= lo:
        return 0.0
    return float(np.mean(np.nan_to_num(logits[lo:hi], nan=0.)))


def propose_inserts(labels, graded, features, team_logits,
                    k=DEFAULT_K, min_score=DEFAULT_MIN_SCORE,
                    min_island=DEFAULT_MIN_ISLAND, sep=DEFAULT_SEP,
                    window=DEFAULT_WINDOW, min_disagree=DEFAULT_MIN_DISAGREE):
    """Return ranked insert candidates: list of (cut, score, left_team, right_team)."""
    labels = np.asarray(labels)
    graded = np.asarray(graded, bool)
    logits = np.asarray(team_logits, float)
    score = contested_score(features)
    pred_sw = ha_switches(labels, graded)
    peaks = contested_peaks(score, graded, min_score=min_score, min_separation=sep)
    cands = []
    for cut in peaks:
        cut = int(cut)
        if cut <= 0 or cut >= len(labels) - 1:
            continue
        if labels[cut] not in ('H', 'A') or not graded[cut]:
            continue
        if labels[cut - 1] != labels[cut]:
            continue  # already a boundary
        if len(pred_sw) and np.min(np.abs(pred_sw - cut)) < sep:
            continue
        a, b = _spell_bounds(labels, graded, cut)
        left_len = cut - a
        right_len = b - cut
        if left_len < min_island or right_len < min_island:
            continue
        left_mean = _mean_logit(logits, cut - window, cut)
        right_mean = _mean_logit(logits, cut, cut + window)
        left_want = 'H' if left_mean >= 0 else 'A'
        right_want = 'H' if right_mean >= 0 else 'A'
        if left_want == right_want:
            continue  # no polarity flip across peak
        disagree = abs(left_mean - right_mean)
        if disagree < float(min_disagree):
            continue
        cur = labels[cut]
        # Only propose if exactly one side wants to leave the current team
        # (safer than rewriting both halves of a long spell).
        flip_right = (right_want != cur and left_want == cur)
        flip_left = (left_want != cur and right_want == cur)
        if not flip_right and not flip_left:
            continue
        side = 'right' if flip_right else 'left'
        rank = float(score[cut]) * (1.0 + disagree)
        cands.append((rank, cut, side, left_want, right_want, a, b))
    cands.sort(key=lambda t: -t[0])
    return cands[: int(k)]


def apply_inserts(labels, graded, features, team_logits, params, minimum=MERGE_MINIMUM):
    labels = np.asarray(labels).copy()
    graded = np.asarray(graded, bool)
    accepted = []
    remaining_k = int(params['k'])
    used = []
    for _ in range(remaining_k):
        cands = propose_inserts(
            labels, graded, features, team_logits,
            k=max(32, remaining_k * 4),
            min_score=params['min_score'],
            min_island=params['min_island'],
            sep=params['sep'],
            window=params['window'],
            min_disagree=params['min_disagree'])
        pick = None
        for rank, cut, side, left_want, right_want, a, b in cands:
            if any(abs(cut - u) < params['sep'] for u in used):
                continue
            if labels[cut] not in ('H', 'A') or labels[cut - 1] != labels[cut]:
                continue
            a2, b2 = _spell_bounds(labels, graded, cut)
            if (cut - a2) < params['min_island'] or (b2 - cut) < params['min_island']:
                continue
            pick = (cut, side, left_want, right_want, a2, b2, rank)
            break
        if pick is None:
            break
        cut, side, left_want, right_want, a, b, rank = pick
        if side == 'right':
            labels[cut:b] = right_want
        else:
            labels[a:cut] = left_want
        used.append(cut)
        accepted.append(dict(cut=int(cut), rank=float(rank), side=side,
                             left=left_want, right=right_want))
    return _merge_short_spells(labels, graded, minimum), accepted


def default_params():
    return dict(
        k=DEFAULT_K,
        min_score=DEFAULT_MIN_SCORE,
        min_island=DEFAULT_MIN_ISLAND,
        sep=DEFAULT_SEP,
        window=DEFAULT_WINDOW,
        min_disagree=DEFAULT_MIN_DISAGREE,
    )


def nested_tune(train_games, grid=None):
    """Tune insert params by mean spell Δ on the last 3 train matches."""
    grid = grid or dict(
        k=[6, 10, 16],
        min_score=[0.75, 0.9, 1.0],
        min_island=[18, 25],
        sep=[18, 25],
        window=[8, 12],
        min_disagree=[0.2, 0.35],
    )
    select_games = train_games[-3:]
    cached = []
    for game in select_games:
        data = load_match(game)
        baseline = predict_version(data['features'], data['graded'], (2, 3))
        logits = v23_team_logits(
            data['features'], data['graded'], data['nodes'], baseline)
        base_metrics = score_natural(data['gt'], baseline, data['features'])
        cached.append(dict(data=data, baseline=baseline, logits=logits,
                           base_spell=base_metrics['spell_f1_mean']))
    best = None
    best_key = None
    for k in grid['k']:
        for min_score in grid['min_score']:
            for min_island in grid['min_island']:
                for sep in grid['sep']:
                    for window in grid['window']:
                        for min_disagree in grid['min_disagree']:
                            params = dict(
                                k=k, min_score=min_score, min_island=min_island,
                                sep=sep, window=window, min_disagree=min_disagree)
                            deltas = []
                            n_ins = []
                            for item in cached:
                                pred, acc = apply_inserts(
                                    item['baseline'], item['data']['graded'],
                                    item['data']['features'], item['logits'], params)
                                metrics = score_natural(
                                    item['data']['gt'], pred, item['data']['features'])
                                deltas.append(metrics['spell_f1_mean'] - item['base_spell'])
                                n_ins.append(len(acc))
                            mean_delta = float(np.mean(deltas))
                            mean_n = float(np.mean(n_ins))
                            insert_pref = 1 if 1 <= mean_n <= 8 else (0 if mean_n < 1 else -1)
                            key = (round(mean_delta, 5), insert_pref, -abs(mean_n - 4))
                            cand = dict(params=params, delta=mean_delta,
                                        n_inserts=mean_n, select_games=select_games)
                            if best is None or key > best_key:
                                best = cand
                                best_key = key
    return best


def visible_acc(gt, pred, features):
    present = np.asarray(features['ball_present'], bool)
    live = present & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
    return float(np.mean(pred[live] == gt[live])) if live.any() else float('nan')


def crossfit(matches=None):
    matches = matches or MATCHES
    rows, tuned = [], []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        print('tune fold', holdout, flush=True)
        # Smaller grid for speed on each fold
        best = nested_tune(train_games, grid=dict(
            k=[6, 10, 16],
            min_score=[0.75, 0.9, 1.0],
            min_island=[18, 25],
            sep=[18, 25],
            window=[12],
            min_disagree=[0.2, 0.35],
        ))
        tuned.append(dict(holdout=holdout, **best))
        print(' ', best['params'], 'delta', round(best['delta'], 4),
              'inserts', best['n_inserts'], flush=True)
        data = load_match(holdout)
        baseline = predict_version(data['features'], data['graded'], (2, 3))
        logits = v23_team_logits(
            data['features'], data['graded'], data['nodes'], baseline)
        pred, accepted = apply_inserts(
            baseline, data['graded'], data['features'], logits, best['params'])
        metrics = score_natural(data['gt'], pred, data['features'])
        m23 = score_natural(data['gt'], baseline, data['features'])
        rows.append(dict(
            game=holdout, candidate='insert_only',
            n_inserts=len(accepted),
            visible_team_acc=visible_acc(data['gt'], pred, data['features']),
            visible_team_acc_v23=visible_acc(data['gt'], baseline, data['features']),
            spell_f1_v23=m23['spell_f1_mean'],
            **best['params'],
            **metrics))
        print(holdout, 'inserts', len(accepted),
              'spell', round(metrics['spell_f1_mean'], 4),
              'v23', round(m23['spell_f1_mean'], 4), flush=True)
    return rows, tuned


def gate_decision(rows):
    spell = float(np.mean([r['spell_f1_mean'] for r in rows]))
    spell23 = float(np.mean([r['spell_f1_v23'] for r in rows]))
    deltas = [r['spell_f1_mean'] - r['spell_f1_v23'] for r in rows]
    worst = float(np.min(deltas))
    ok = (spell >= spell23 - 1e-12) and (worst >= -0.02)
    return dict(
        pass_crossfit=bool(ok),
        mean_spell=spell, mean_spell_v23=spell23, worst_spell_delta=worst,
        per_match_spell_delta={r['game']: float(r['spell_f1_mean'] - r['spell_f1_v23'])
                               for r in rows},
    )


def majority_params(tuned):
    keys = ['k', 'min_score', 'min_island', 'sep', 'window', 'min_disagree']

    def mode(vals):
        vals = list(vals)
        return max(set(vals), key=vals.count)

    out = {}
    for key in keys:
        vals = [t['params'][key] for t in tuned]
        out[key] = mode(vals)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--params-json', default=None,
                        help='apply fixed params to all matches (no nested tune)')
    args = parser.parse_args()
    if args.params_json:
        params = json.loads(Path(args.params_json).read_text())
        rows = []
        for game in MATCHES:
            data = load_match(game)
            baseline = predict_version(data['features'], data['graded'], (2, 3))
            logits = v23_team_logits(
                data['features'], data['graded'], data['nodes'], baseline)
            pred, accepted = apply_inserts(
                baseline, data['graded'], data['features'], logits, params)
            metrics = score_natural(data['gt'], pred, data['features'])
            m23 = score_natural(data['gt'], baseline, data['features'])
            rows.append(dict(game=game, candidate='insert_only_fixed',
                             n_inserts=len(accepted),
                             spell_f1_v23=m23['spell_f1_mean'], **metrics))
            print(game, len(accepted), round(metrics['spell_f1_mean'], 4),
                  round(m23['spell_f1_mean'], 4), flush=True)
        Path(REPORTS / 'insert_only_fixed.json').write_text(
            json.dumps(dict(params=params, rows=rows, gate=gate_decision(rows)), indent=2))
        return
    rows, tuned = crossfit()
    write_csv(REPORTS / 'crossfit_insert_only.csv', rows)
    decision = gate_decision(rows)
    ship = majority_params(tuned)
    summary = dict(
        aggregate=summarize_rows(rows),
        gate=decision,
        ship_params=ship,
        tuned=tuned,
        rows=rows,
    )
    Path(REPORTS / 'crossfit_insert_only.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(gate=decision, ship_params=ship,
                          aggregate=summary['aggregate']), indent=2))


if __name__ == '__main__':
    main()
