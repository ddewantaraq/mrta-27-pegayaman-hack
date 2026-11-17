"""Final lever: contested-weighted touch-team training.

The boundary head got contested weighting (held); the event-first touch
HGB never did. Touches within 25f of a GT switch get extra sample weight.
Same candidates/labels/decoder grid as event_first.py, FIT LOMO only.
Writes evaluation/reports/touch_contested_weight_crossfit.json.
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


def contested_flags(gt, idx, window=25):
    gt = np.asarray(gt)
    live = np.isin(gt, ['H', 'A'])
    switches = np.flatnonzero(live[:-1] & live[1:] & (gt[:-1] != gt[1:])) + 1
    flags = np.zeros(len(gt), bool)
    for s in switches:
        flags[max(0, s - window):s + window + 1] = True
    return flags[np.asarray(idx)]


def fit_weighted(items, extra):
    x = np.concatenate([d['x'][d['keep']] for d in items])
    y = np.concatenate([d['y'][d['keep']] for d in items])
    w = np.concatenate([
        1. + extra * contested_flags(d['gt'], d['idx'])[d['keep']]
        for d in items])
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1., max_bins=127, early_stopping=False, random_state=29)
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=w)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_contested_weight_crossfit.json'))
    args = parser.parse_args()
    items = [game_cache(g) for g in FIT]
    result = {}
    for extra in (0., 2., 4.):
        rows = []
        for d in items:
            train = [e for e in items if e['game'] != d['game']]
            rows += score_holdout(d, fit_weighted(train, extra), 'fit')
        best, table = choose(rows)
        result['w{}'.format(extra)] = dict(best=best, row=table.get(best))
        print('extra', extra, 'best', best, json.dumps(table.get(best)), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
