"""Duel-conditional margin: relax event-first overwrite only at duels.

Missed switches are 94.5% contested; false switches less so. At touches
where |min_dist_home - min_dist_away| <= 1m (duel), use a lower margin;
elsewhere keep shipped 1.0. Same trained touch model (no refit change),
decoder-only ablation, FIT LOMO. Writes touch_duel_margin_crossfit.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, score_natural
from event_first import FIT, as_arrays, fit_model, game_cache, packed
from event_first_model import live_spans, viterbi_two_state
from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells


def _sign(labels):
    return np.where(labels == 'H', 1., np.where(labels == 'A', -1., 0.))


def decode_duel_margin(d, logit, duel_margin, base_margin=1.0, cost=0.0):
    labels = np.asarray(d['ref']).copy()
    present = np.asarray(d['f']['ball_present'], bool)
    graded = np.asarray(d['graded'], bool)
    idx = np.asarray(d['idx'])
    f = d['f']
    dh = np.nan_to_num(np.asarray(f['min_dist_home'], np.float64), nan=99.)
    da = np.nan_to_num(np.asarray(f['min_dist_away'], np.float64), nan=99.)
    duel = np.abs(dh - da) <= 1.0
    margins = np.where(duel[idx], duel_margin, base_margin)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_duel_margin_crossfit.json'))
    args = parser.parse_args()
    items = [game_cache(g) for g in FIT]
    result = {}
    for duel_margin in (1.0, 0.5, 0.0):
        spells, hiddens, worst = [], [], []
        anchor_spells = {}
        for d in items:
            train = [e for e in items if e['game'] != d['game']]
            model = fit_model(train)
            logit = boosted_logits(d['x'], as_arrays(packed(model)))
            labels = decode_duel_margin(d, logit, duel_margin)
            m = score_natural(d['gt'], labels, d['f'])
            base = score_natural(d['gt'], d['ref'], d['f'])
            spells.append(m['spell_f1_mean']); hiddens.append(m['hidden_team_acc'])
            worst.append(m['spell_f1_mean'] - base['spell_f1_mean'])
            anchor_spells[d['game']] = base['spell_f1_mean']
        result['dm{}'.format(duel_margin)] = dict(
            spell=float(np.mean(spells)), hidden=float(np.mean(hiddens)),
            worst_delta=float(np.min(worst)))
        print('duel_margin', duel_margin, json.dumps(result['dm{}'.format(duel_margin)]), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
