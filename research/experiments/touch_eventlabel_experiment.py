"""P1 follow-up: touch-team head trained on event-team labels vs GT lead+3.

Rule candidates only; labels from Sportec events_raw play-team (0.90-0.93
GT agreement) with GT lead+3 backoff beyond 12f. LOMO over the three
Sportec FIT matches (Sample_Game_3 has no events_raw equivalent and is
excluded from both variants). Same HGB config and decoder grid as
event_first.py. Writes evaluation/reports/touch_eventlabel_crossfit.json.
"""
import argparse
import json
import os
import sys
from bisect import bisect_left
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first import CACHE, LABEL_LEAD, fit_model, score_holdout
from event_first_model import touch_candidates, touch_design
from touch_supervision import parse_sportec_events_raw, sportec_frame_offset

FIT3 = ['J03WOH', 'J03WN1', 'J03WOY']
SPORTEC_ROOT = Path('data_external/sportec')
EVENT_TOL = 12


def event_labels(game, idx, gt):
    """Play-event team within EVENT_TOL frames, else GT lead+3."""
    offset = sportec_frame_offset(game, SPORTEC_ROOT)
    plays = sorted((t.frame_n - offset, t.team)
                   for t in parse_sportec_events_raw(game, SPORTEC_ROOT)
                   if t.kind == 'play' and t.team is not None)
    pos = [p[0] for p in plays]
    gt = np.asarray(gt)
    labels = []
    for i in idx:
        best, best_d = None, EVENT_TOL + 1
        j = bisect_left(pos, int(i))
        for k in (j - 1, j):
            if 0 <= k < len(pos) and abs(pos[k] - i) < best_d:
                best, best_d = plays[k][1], abs(pos[k] - i)
        if best is None:
            g = gt[min(int(i) + LABEL_LEAD, len(gt) - 1)]
            labels.append(g)
        else:
            labels.append('H' if best == 0 else 'A')
    return np.array(labels)


def game_cache(game, label_fn):
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    path = CACHE / '{}_v2_6.npz'.format(game)
    with np.load(path, allow_pickle=False) as z:
        ref, logits = z['reference'], z['team_logits']
    idx, kin = touch_candidates(f, graded)
    x = touch_design(f, ref, logits, idx, kin)
    lead = gt[np.clip(idx + LABEL_LEAD, 0, len(gt) - 1)]
    keep = np.isin(lead, ['H', 'A'])
    labels = label_fn(game, idx, gt)
    return dict(game=game, f=f, gt=gt, graded=graded, ref=ref, idx=idx, x=x,
                y=(labels == 'H').astype(float), keep=keep)


def gt_labels(game, idx, gt):
    return np.asarray(gt)[np.clip(idx + LABEL_LEAD, 0, len(gt) - 1)]


def lomo3(items):
    rows = []
    for d in items:
        train = [e for e in items if e['game'] != d['game']]
        rows += score_holdout(d, fit_model(train), 'fit')
    return rows


def choose3(rows):
    """event_first.choose over FIT3 (no Sample_Game_3, no SC domain)."""
    by = {}
    for r in rows:
        by.setdefault(r['candidate'], {})[r['game']] = r
    anchor = by['anchor']

    def mean(c, key):
        return float(np.mean([v[key] for v in by[c].values()]))

    table, best, best_spell = {}, None, mean('anchor', 'spell_f1_mean')
    for c in by:
        worst = min(by[c][g]['spell_f1_mean'] - anchor[g]['spell_f1_mean']
                    for g in FIT3)
        table[c] = dict(spell=mean(c, 'spell_f1_mean'),
                        hidden=mean(c, 'hidden_team_acc'), worst_delta=worst)
        if c == 'anchor':
            continue
        if (worst >= -0.005 and table[c]['spell'] > best_spell
                and table[c]['hidden'] + 1e-9 >= mean('anchor', 'hidden_team_acc')):
            best, best_spell = c, table[c]['spell']
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_eventlabel_crossfit.json'))
    args = parser.parse_args()
    gt_items = [game_cache(g, gt_labels) for g in FIT3]
    ev_items = [game_cache(g, event_labels) for g in FIT3]
    n_diff = sum(int((((d['y'] > .5) != (e['y'] > .5)) & d['keep']).sum())
                 for d, e in zip(gt_items, ev_items))
    print('label diffs (event vs GT):', n_diff, flush=True)
    gt_rows = lomo3(gt_items)
    ev_rows = lomo3(ev_items)
    gt_best, gt_table = choose3(gt_rows)
    ev_best, ev_table = choose3(ev_rows)
    print(json.dumps(dict(gt_best=gt_best, gt_row=gt_table.get(gt_best),
                          ev_best=ev_best, ev_row=ev_table.get(ev_best)), indent=2))
    Path(args.out).write_text(json.dumps(dict(
        fit=FIT3, label_diffs=n_diff, gt_best=gt_best, gt_table=gt_table,
        ev_best=ev_best, ev_table=ev_table), indent=2) + '\n')


if __name__ == '__main__':
    main()
