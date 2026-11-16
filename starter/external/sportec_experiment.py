"""Independent Sportec validation; full-match reference never enters labeller.

Compare approved v1.8 and physical-boundary candidate under natural and
coordinate-independent 41% block dropout (including stoppages).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sportec_features import extract_sportec
from eval_external import (kwargs_for_version, run_labeller, score_one, write_csv,
                           aggregate_by_candidate)
from dropout import apply_dropout


def block_dropout(n, seconds=5, seed=41, probability=.41):
    rng = np.random.RandomState(seed)
    block = int(seconds*25)
    states = rng.random_sample((n+block-1)//block) < probability
    return np.repeat(states, block)[:n]


def cached_sportec(match):
    cache = 'evaluation/reports/feature_cache/sportec_' + match + '.npz'
    if os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as data:
            feats = {k: data[k] for k in data.files if k not in ('gt', 'graded')}
            gt, graded = data['gt'].copy(), data['graded'].copy()
        feats['n_frames'] = int(feats['n_frames'])
    else:
        print('extract Sportec', match, flush=True)
        feats, gt, graded = extract_sportec(match)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **feats, gt=gt, graded=graded)
    return feats, gt, graded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WOH')
    args = parser.parse_args()
    feats, gt, graded = cached_sportec(args.match)
    rows = []
    specs = [('natural', ~feats['ball_present'])]
    for sec in (2, 5, 10, 30):
        specs.append(('blocks{}s'.format(sec), block_dropout(feats['n_frames'], sec)))
    for drop, hide in specs:
        f = apply_dropout(feats, hide)
        for name, margin in [('latest', .02), ('boundary', 0.)]:
            kw = kwargs_for_version(1, 8)
            kw['out_margin'] = margin
            pred = run_labeller(f, graded, kw)
            rows.append(dict(game=args.match, dropout=drop, candidate=name,
                             **score_one(gt, pred, hide | ~feats['ball_present'])))
        print(drop, flush=True)
    write_csv('evaluation/reports/sportec_' + args.match + '.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
