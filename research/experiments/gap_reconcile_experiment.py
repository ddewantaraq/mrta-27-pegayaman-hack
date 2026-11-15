"""Evaluate reconciling missing-ball labels after the visible/hidden splice.

Uses only external Metrica labels. Seed 17 is a new dropout robustness check;
it does not make the previously used external matches independent test games.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_external import (GAME_NAMES, aggregate_by_candidate, per_game_means,
                           kwargs_for_version, run_labeller, score_one, write_csv)
from metrica_features import extract_metrica_features
from metrica_gt import build_metrica_gt
from dropout import apply_dropout, dropout_configs
from fetch_metrica import DEFAULT_OUT
from short_spell_experiment import suppress_live_islands


def reconcile_gaps(labels, ball_present, graded, mode='mid'):
    """Reconcile a missing live stretch with its final visible endpoints.

    Same-team endpoints imply uninterrupted team possession. Opposing endpoints
    place one unknown turnover at the midpoint (or retain the existing split
    in same-only mode). D, X, period/graded boundaries are barriers.
    """
    out = np.asarray(labels).copy()
    labels = np.asarray(labels)
    present = np.asarray(ball_present, dtype=bool)
    graded = np.asarray(graded, dtype=bool)
    n = len(out)
    i = 0
    while i < n:
        if not graded[i] or present[i] or labels[i] not in ('H', 'A'):
            i += 1
            continue
        a = i
        while i < n and graded[i] and not present[i] and labels[i] in ('H', 'A'):
            i += 1
        b = i
        if a == 0 or b == n or not graded[a-1] or not graded[b]:
            continue
        if not present[a-1] or not present[b]:
            continue
        left, right = labels[a-1], labels[b]
        if left not in ('H', 'A') or right not in ('H', 'A'):
            continue
        if left == right:
            out[a:b] = left
        elif mode == 'mid':
            cut = a + (b-a)//2
            out[a:cut] = left
            out[cut:b] = right
    return out


def cached_features(game):
    directory = 'evaluation/reports/feature_cache'
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, game + '.npz')
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as data:
            feats = {key: data[key] for key in data.files}
        feats['n_frames'] = int(feats['n_frames'])
        return feats
    feats = extract_metrica_features(game, root=DEFAULT_OUT)
    np.savez_compressed(path, **feats)
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--generate', metavar='CSV')
    parser.add_argument('--mode', choices=('same', 'mid'), default='mid')
    args = parser.parse_args()
    if args.generate:
        from features import extract_features
        from state_machine import graded_mask, splice_vc_on_visible
        from predict_v1 import write_submission
        from vc_possession import live_structure
        feats = extract_features()
        graded = graded_mask(feats['n_frames'])
        pred = splice_vc_on_visible(feats, graded=graded)
        out = reconcile_gaps(pred, feats['ball_present'], graded, args.mode)
        write_submission(args.generate, out)
        print(json.dumps(live_structure(out, graded), indent=2))
        return
    rows = []
    for game in GAME_NAMES:
        print(game, flush=True)
        feats = cached_features(game)
        gt = build_metrica_gt(game, feats['n_frames'], root=DEFAULT_OUT)
        # Explicitly break decoder state across tracking periods, without GT.
        graded = np.ones(feats['n_frames'], dtype=bool)
        for name, hide in dropout_configs(gt, feats, seed=args.seed):
            dropped = apply_dropout(feats, hide)
            pred = run_labeller(dropped, graded, kwargs_for_version(1, 8))
            score_hide = hide | ~feats['ball_present']
            candidates = {'latest': pred}
            for mode in ('same', 'mid'):
                candidates[mode] = reconcile_gaps(pred, dropped['ball_present'], graded, mode)
            candidates['island25'] = suppress_live_islands(pred, graded, 25, dropped['ball_present'])
            for cand, output in candidates.items():
                rows.append(dict(game=game, dropout=name, candidate=cand,
                                 **score_one(gt, output, score_hide)))
            print(' ', name, flush=True)
    path = 'evaluation/reports/gap_reconcile_seed{}.csv'.format(args.seed)
    write_csv(path, rows)
    print(json.dumps(dict(aggregate=aggregate_by_candidate(rows),
                          per_game=per_game_means(rows)), indent=2))


if __name__ == '__main__':
    main()
