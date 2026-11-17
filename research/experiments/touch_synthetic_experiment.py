"""Rank 6 gate: touch HGB real vs real+synthetic duels, FIT LOMO.

Synthetic duel snippets (control-outcome labels) are added to every
training fold with a sample weight; selection identical to event_first.
Writes evaluation/reports/touch_synthetic_crossfit.json.
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
from crossfit_common import REPORTS
from event_first import FIT, choose, game_cache, score_holdout
from synthetic_duels import synthetic_design


def fit_mixed(items, xs, ys, syn_weight):
    x = np.concatenate([d['x'][d['keep']] for d in items] + ([xs] if syn_weight else []))
    y = np.concatenate([d['y'][d['keep']] for d in items] + ([ys] if syn_weight else []))
    w = np.concatenate([np.ones(sum(int(d['keep'].sum()) for d in items)),
                        np.full(len(ys), syn_weight) if syn_weight else np.zeros(0)])
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1., max_bins=127, early_stopping=False, random_state=29)
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=w)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--syn-weight', type=float, default=1.0)
    parser.add_argument('--out', default=str(REPORTS / 'touch_synthetic_crossfit.json'))
    args = parser.parse_args()
    items = [game_cache(g) for g in FIT]
    xs, ys = synthetic_design()
    print('synthetic', xs.shape, 'real touches', sum(int(d['keep'].sum()) for d in items), flush=True)
    for weight in (0.0, args.syn_weight):
        rows = []
        for d in items:
            train = [e for e in items if e['game'] != d['game']]
            rows += score_holdout(d, fit_mixed(train, xs, ys, weight), 'fit')
        best, table = choose(rows)
        print('weight', weight, 'best', best, json.dumps(table.get(best)), flush=True)
        if weight == 0.0:
            base = table.get(best)
        else:
            mixed = table.get(best)
    Path(args.out).write_text(json.dumps(
        dict(real_best=base, syn_weight=args.syn_weight, syn_best=mixed), indent=2) + '\n')


if __name__ == '__main__':
    main()
