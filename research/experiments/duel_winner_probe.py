"""Rank 3 probe: duel-winner head vs touch logit on duel windows only.

Sportec FIT matches: duel frames from events_raw TacklingGame, label =
WinnerTeam. Features = 22 touch features + 8 intercept features at the
duel frame. LOMO accuracy vs the touch HGB logit on the same frames.
Kill gate: duel head must beat touch logit on duel windows.
Writes evaluation/reports/duel_winner_probe.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first import CACHE, fit_model, game_cache
from event_first_model import touch_design, touch_candidates
from touch_physics_experiment import intercept_features
from touch_supervision import parse_sportec_events_raw, sportec_frame_offset

FIT3 = ['J03WOH', 'J03WN1', 'J03WOY']
ROOT = Path('data_external/sportec')


def duel_cache(game):
    d = game_cache(game)
    offset = sportec_frame_offset(game, ROOT)
    duels = [(t.frame_n - offset, t.team)
             for t in parse_sportec_events_raw(game, ROOT)
             if t.kind == 'duel' and t.team is not None]
    idx = np.array([i for i, _ in duels if 0 <= i < len(d['gt'])], dtype=np.int64)
    y = np.array([team for i, team in duels if 0 <= i < len(d['gt'])], float)
    nodes = load_match(game)['nodes']
    with np.load(CACHE / '{}_v2_6.npz'.format(game), allow_pickle=False) as z:
        ref, logits = z['reference'], z['team_logits']
    kin = touch_candidates(d['f'], d['graded'])[1]
    x = np.hstack([touch_design(d['f'], ref, logits, idx, kin),
                   intercept_features(d['f'], nodes, idx)])
    return dict(game=game, x=x, y=(y == 0).astype(float), ref=d)


def fit_duel(items):
    x = np.concatenate([d['x'] for d in items])
    y = np.concatenate([d['y'] for d in items])
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1., max_bins=127, early_stopping=False, random_state=29)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'duel_winner_probe.json'))
    args = parser.parse_args()
    items = [duel_cache(g) for g in FIT3]
    rows = []
    for d in items:
        train = [e for e in items if e['game'] != d['game']]
        duel_model = fit_duel(train)
        duel_acc = float(np.mean(
            (duel_model.predict(d['x']) > .5) == (d['y'] > .5)))
        # touch HGB baseline logit on same duel frames
        touch_items = [game_cache(e['game']) for e in train]
        touch_model = fit_model(touch_items)
        from event_first import as_arrays, packed
        from boosted_dead_model import boosted_logits
        # duel x first 22 cols are touch design
        logit = boosted_logits(d['x'][:, :22], as_arrays(packed(touch_model)))
        touch_acc = float(np.mean((logit >= 0) == (d['y'] > .5)))
        rows.append(dict(game=d['game'], n_duels=len(d['y']),
                         duel_acc=duel_acc, touch_acc=touch_acc))
        print(d['game'], 'duels', len(d['y']), 'duel-head', round(duel_acc, 4),
              'touch-logit', round(touch_acc, 4), flush=True)
    result = dict(rows=rows,
                  mean_duel=float(np.mean([r['duel_acc'] for r in rows])),
                  mean_touch=float(np.mean([r['touch_acc'] for r in rows])))
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
