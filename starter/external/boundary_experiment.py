"""External-only test of physical pitch boundary versus inward dead-ball bands."""
import json
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_reconcile_experiment import cached_features
from eval_external import (GAME_NAMES, kwargs_for_version, run_labeller, score_one,
                           write_csv, aggregate_by_candidate, per_game_means)
from metrica_gt import build_metrica_gt
from dropout import apply_dropout, dropout_configs
from fetch_metrica import DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    rows = []
    for game in GAME_NAMES:
        feats = cached_features(game)
        gt = build_metrica_gt(game, feats['n_frames'], root=DEFAULT_OUT)
        graded = np.ones(feats['n_frames'], dtype=bool)
        for drop, hide in dropout_configs(gt, feats, seed=args.seed):
            f = apply_dropout(feats, hide)
            for name, margin in [('latest', .02), ('boundary', 0.), ('band005', .005)]:
                kw = kwargs_for_version(1, 8)
                kw['out_margin'] = margin
                pred = run_labeller(f, graded, kw)
                rows.append(dict(game=game, dropout=drop, candidate=name,
                                 **score_one(gt, pred, hide | ~feats['ball_present'])))
            print(game, drop, flush=True)
    write_csv('evaluation/reports/boundary_experiment_seed{}.csv'.format(args.seed), rows)
    print(json.dumps(dict(aggregate=aggregate_by_candidate(rows),
                          per_game=per_game_means(rows)), indent=2))


if __name__ == '__main__':
    main()
