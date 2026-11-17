"""Seed-ensembled touch HGB: average logits over seeds vs single seed 29.

Shipped event-first head is one HGB (seed 29). Averaging raw logits over
seeds is a pure variance reduction; NumPy export would just carry more
trees. Same candidates/labels/decoder, FIT LOMO selection.
Writes evaluation/reports/touch_seed_ensemble_crossfit.json.
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
from crossfit_common import REPORTS, score_natural
from event_first import FIT, as_arrays, choose, fit_model, game_cache, packed, score_holdout
from event_first_model import decode_event_first
from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells

SEEDS = (29, 73, 101, 202, 307)


def fit_seed(items, seed):
    x = np.concatenate([d['x'][d['keep']] for d in items])
    y = np.concatenate([d['y'][d['keep']] for d in items])
    w = np.concatenate([np.full(int(d['keep'].sum()),
                                1. if d['game'].startswith('SC_') else 1.)
                        for d in items])
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1., max_bins=127, early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=w)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_seed_ensemble_crossfit.json'))
    args = parser.parse_args()
    items = [game_cache(g) for g in FIT]
    # single-seed baseline rows via existing machinery
    base_rows = []
    for d in items:
        train = [e for e in items if e['game'] != d['game']]
        base_rows += score_holdout(d, fit_model(train), 'fit')
    base_best, base_table = choose(base_rows)
    # ensemble rows: decode with averaged logits over the decoder grid
    from event_first import MARGINS, SWITCH_COSTS
    ens_rows = []
    for d in items:
        train = [e for e in items if e['game'] != d['game']]
        logits = np.mean([boosted_logits(d['x'], as_arrays(packed(fit_seed(train, s))))
                          for s in SEEDS], axis=0)
        base = score_natural(d['gt'], d['ref'], d['f'])
        ens_rows.append(dict(game=d['game'], domain='fit', candidate='anchor',
                             touch_acc=0., **base))
        for cost in SWITCH_COSTS:
            for margin in MARGINS:
                labels = _merge_short_spells(
                    decode_event_first(d['f'], d['ref'], d['graded'], logits,
                                       d['idx'], cost, margin), d['graded'], 18)
                m = score_natural(d['gt'], labels, d['f'])
                ens_rows.append(dict(game=d['game'], domain='fit',
                                     candidate='c{}_m{}'.format(cost, margin),
                                     touch_acc=0., **m))
        print(d['game'], 'ens best', round(max(r['spell_f1_mean'] for r in ens_rows
                                              if r['game'] == d['game']), 4), flush=True)
    ens_best, ens_table = choose(ens_rows)
    result = dict(single_best=base_best, single_row=base_table.get(base_best),
                  ensemble_best=ens_best, ensemble_row=ens_table.get(ens_best))
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
