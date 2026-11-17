"""P1 step 2c: event-first head on union (rule + accel) candidates.

Same HGB config, same GT lead+3 touch labels, same decoder grid as
event_first.py; only the candidate set changes (rule vs union).
LOMO over FIT matches; optional grouped SC-train folds via --with-sc.
Compares best-rule vs best-union; Track B/C/D matches never touched.
Writes evaluation/reports/touch_union_crossfit.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, predict_version
from error_corrector_model import ensemble_team_logits
from event_first import (
    ANCHOR,
    CACHE,
    FIT,
    LABEL_LEAD,
    choose,
    fit_model,
    lomo,
    skillcorner_train,
)
from event_first_model import (
    live_spans,
    touch_candidates,
    touch_design,
    viterbi_two_state,
)
from touch_supervision import accel_candidates


def union_candidates(features, graded):
    rule = touch_candidates(features, graded)[0]
    accel = accel_candidates(features, graded)
    return np.array(sorted(set(rule.tolist()) | set(accel.tolist())), dtype=np.int64)


def game_cache_union(game):
    path = CACHE / '{}_v{}_{}.npz'.format(game, *ANCHOR)
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    with np.load(path, allow_pickle=False) as z:
        ref, logits = z['reference'], z['team_logits']
    rule_idx, kin = touch_candidates(f, graded)
    idx = union_candidates(f, graded)
    x = touch_design(f, ref, logits, idx, kin)
    lead = gt[np.clip(idx + LABEL_LEAD, 0, len(gt) - 1)]
    keep = np.isin(lead, ['H', 'A'])
    rule_set = set(rule_idx.tolist())
    accel_only = np.array([i not in rule_set for i in idx.tolist()], bool)
    return dict(game=game, f=f, gt=gt, graded=graded, ref=ref, idx=idx, x=x,
                y=(lead == 'H').astype(float), keep=keep, accel_only=accel_only)


def _sign(labels):
    return np.where(labels == 'H', 1., np.where(labels == 'A', -1., 0.))


def decode_dual_margin(d, logit, cost, margin_rule, margin_accel):
    """decode_event_first with a higher bar for accel-only candidates."""
    from state_machine import _merge_short_spells
    labels = np.asarray(d['ref']).copy()
    present = np.asarray(d['f']['ball_present'], bool)
    graded = np.asarray(d['graded'], bool)
    idx = np.asarray(d['idx'])
    margins = np.where(d['accel_only'], margin_accel, margin_rule)
    out = labels.copy()
    for a, b in live_spans(labels, graded):
        inside = (idx >= a) & (idx < b)
        if not inside.any():
            continue
        ti, tl = idx[inside], logit[inside]
        tm = margins[inside]
        state = viterbi_two_state(tl, cost, _sign(labels[a:a + 1])[0])
        bounds = np.r_[ti, b]
        for k in range(len(ti)):
            team = 'H' if state[k] > 0 else 'A'
            if team == labels[ti[k]] or abs(tl[k]) < tm[k] or np.sign(tl[k]) != state[k]:
                continue
            seg = np.arange(bounds[k], bounds[k + 1])
            out[seg[present[seg]]] = team
    return _merge_short_spells(out, graded, 18)


def margin_sweep(union_items):
    """FIT LOMO over accel-only margins; rule margin fixed at shipped 1.0."""
    from crossfit_common import score_natural
    from event_first import as_arrays, packed
    from boosted_dead_model import boosted_logits
    rows = []
    for d in union_items:
        train = [e for e in union_items if e['game'] != d['game']]
        model = fit_model(train)
        logit = boosted_logits(d['x'], as_arrays(packed(model)))
        base = score_natural(d['gt'], d['ref'], d['f'])
        rows.append(dict(game=d['game'], candidate='anchor', **base))
        for margin_accel in (1.0, 2.0, 3.0, 4.0):
            labels = decode_dual_margin(d, logit, 0.0, 1.0, margin_accel)
            m = score_natural(d['gt'], labels, d['f'])
            rows.append(dict(game=d['game'],
                             candidate='ma{}'.format(margin_accel), **m))
        print(d['game'], 'anchor', round(base['spell_f1_mean'], 4),
              'best-ma', round(max(r['spell_f1_mean'] for r in rows
                                   if r['game'] == d['game']), 4), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-sc', action='store_true')
    parser.add_argument('--sc-weight', type=float, default=1.0)
    parser.add_argument('--margin-sweep', action='store_true')
    parser.add_argument('--out', default=str(REPORTS / 'touch_union_crossfit.json'))
    args = parser.parse_args()
    if args.margin_sweep:
        union_items = [game_cache_union(g) for g in FIT]
        rows = margin_sweep(union_items)
        Path(args.out).write_text(json.dumps(dict(rows=rows), indent=2) + '\n')
        by = {}
        for r in rows:
            by.setdefault(r['candidate'], []).append(r['spell_f1_mean'])
        print(json.dumps({k: round(float(np.mean(v)), 4) for k, v in by.items()},
                         indent=2))
        return
    from event_first import game_cache as rule_cache
    sc_names = skillcorner_train() if args.with_sc else []
    rule_items = [rule_cache(g) for g in FIT]
    union_items = [game_cache_union(g) for g in FIT]
    rule_sc = [rule_cache(g) for g in sc_names]
    union_sc = [game_cache_union(g) for g in sc_names]
    print('touches rule/union:',
          sum(int(d['keep'].sum()) for d in rule_items),
          sum(int(d['keep'].sum()) for d in union_items), flush=True)
    rule_rows = lomo(rule_items, rule_sc, args.sc_weight if args.with_sc else 0.0)
    union_rows = lomo(union_items, union_sc, args.sc_weight if args.with_sc else 0.0)
    rule_best, rule_table = choose(rule_rows)
    union_best, union_table = choose(union_rows)
    result = dict(fit=FIT, with_sc=args.with_sc, sc_weight=args.sc_weight,
                  rule_best=rule_best, rule_table=rule_table,
                  union_best=union_best, union_table=union_table)
    print(json.dumps(dict(rule_best=rule_best,
                          rule_row=rule_table.get(rule_best),
                          union_best=union_best,
                          union_row=union_table.get(union_best)), indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
