"""External-only experiment: remove brief live islands enclosed by dead ball.

No challenge ground truth is read. Outputs aggregate metrics, never tracking.
Run from the repository root with Python.
"""
import json
import argparse
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


def suppress_live_islands(labels, graded, max_frames=12, ball_present=None):
    """Replace only sub-threshold D-(H/A)-D islands, within one graded block.

    A brief apparent restart immediately followed by another stoppage may be
    edge tracking flicker. Never absorb a turnover or cross X/half boundaries.
    This is an experimental hypothesis, not an assertion about true restarts.
    """
    labels = np.asarray(labels)
    graded = np.asarray(graded, dtype=bool)
    out = labels.copy()
    i = 0
    while i < len(labels):
        j = i + 1
        while j < len(labels) and labels[j] == labels[i] and graded[j] == graded[i]:
            j += 1
        if (graded[i] and labels[i] in ('H', 'A') and j-i < max_frames
                and i > 0 and j < len(labels) and graded[i-1] and graded[j]
                and labels[i-1] == 'D' and labels[j] == 'D'
                and (ball_present is None or np.all(ball_present[i:j]))):
            out[i:j] = 'D'
        i = j
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', metavar='CSV',
                        help='regenerate the visible-island25 candidate from ATD')
    args = parser.parse_args()
    if args.generate:
        from features import extract_features
        from state_machine import graded_mask, splice_vc_on_visible
        from predict_v1 import write_submission
        feats = extract_features()
        graded = graded_mask(feats['n_frames'])
        pred = splice_vc_on_visible(feats, graded=graded)
        output = suppress_live_islands(pred, graded, 25, feats['ball_present'])
        write_submission(args.generate, output)
        from vc_possession import live_structure
        print(json.dumps(live_structure(output, graded), indent=2))
        return
    rows = []
    for game in GAME_NAMES:
        print(game, flush=True)
        feats = extract_metrica_features(game, root=DEFAULT_OUT)
        gt = build_metrica_gt(game, feats['n_frames'], root=DEFAULT_OUT)
        graded = np.ones(feats['n_frames'], dtype=bool)
        for name, hide in dropout_configs(gt, feats):
            dropped = apply_dropout(feats, hide)
            pred = run_labeller(dropped, graded, kwargs_for_version(1, 8))
            score_hide = hide | ~feats['ball_present']
            for threshold in (0, 12, 25):
                cand = 'latest' if threshold == 0 else 'visible_island{}'.format(threshold)
                output = pred if threshold == 0 else suppress_live_islands(
                    pred, graded, threshold, dropped['ball_present'])
                rows.append(dict(game=game, dropout=name, candidate=cand,
                                 **score_one(gt, output, score_hide)))
            print(' ', name, flush=True)
    os.makedirs('evaluation/reports', exist_ok=True)
    write_csv('evaluation/reports/short_spell_experiment.csv', rows)
    print(json.dumps(dict(aggregate=aggregate_by_candidate(rows),
                          per_game=per_game_means(rows)), indent=2))


if __name__ == '__main__':
    main()
