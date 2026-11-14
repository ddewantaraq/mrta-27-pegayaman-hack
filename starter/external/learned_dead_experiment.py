"""Train on external J03WOH, optionally adding Metrica Game 1; evaluate separately.

Excluded fitting matches are J03WN1 and Metrica Game 2. Reference labels are separate
from geometry features. The saved model is a small numpy parameter archive.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sportec_experiment import cached_sportec, block_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one, write_csv,
                           aggregate_by_candidate)
from dropout import apply_dropout
from dead_ball_model import design_features, fit_logistic, decode_dead, protect_hidden_live

MODEL = 'evaluation/reports/dead_model_sportec_J03WOH.npz'


def train(include_metrica=False):
    xs, ys = [], []
    datasets = [cached_sportec('J03WOH')]
    if include_metrica:
        from gap_reconcile_experiment import cached_features
        from metrica_gt import build_metrica_gt
        from fetch_metrica import DEFAULT_OUT
        f = cached_features('Sample_Game_1')
        truth = build_metrica_gt('Sample_Game_1', f['n_frames'], root=DEFAULT_OUT)
        datasets.append((f, truth, np.ones(f['n_frames'], bool)))
    for feats, gt, graded in datasets:
        for seconds in (0, 5, 15):
            hide = np.zeros(feats['n_frames'], bool) if seconds == 0 else block_dropout(
                feats['n_frames'], seconds, seed=73)
            f = apply_dropout(feats, hide)
            x = design_features(f)
            take = np.flatnonzero(graded & (gt != 'X'))[::5]
            xs.append(x[take])
            ys.append((gt[take] == 'D').astype(float))
    model = fit_logistic(np.concatenate(xs), np.concatenate(ys))
    path = MODEL.replace('.npz', '_mixed.npz') if include_metrica else MODEL
    np.savez(path, **model)
    print('training sources: J03WOH' + (' + Metrica1' if include_metrica else ''),
          sum(len(x) for x in xs), 'samples', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WN1')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--metrica', action='store_true')
    parser.add_argument('--mixed-training', action='store_true')
    parser.add_argument('--generate', metavar='CSV', help='predict challenge from ATD, without GT')
    parser.add_argument('--legacy-dropouts', action='store_true',
                        help='use original natural + 2/5/10/30s live-window dropout suite')
    parser.add_argument('--live-islands', action='store_true', help='guarded <25f live-island postprocess')
    parser.add_argument('--protect-hidden-live', action='store_true',
                        help='preserve approved H/A answers on missing-ball frames')
    args = parser.parse_args()
    model_path = MODEL.replace('.npz', '_mixed.npz') if args.mixed_training else MODEL
    if args.train or not os.path.isfile(model_path):
        model = train(args.mixed_training)
    else:
        with np.load(model_path) as data:
            model = dict(data)
    if args.generate:
        from features import extract_features
        from state_machine import graded_mask, splice_vc_on_visible
        from predict_v1 import write_submission
        from vc_possession import live_structure
        feats = extract_features()
        graded = graded_mask(feats['n_frames'])
        team = splice_vc_on_visible(feats, graded=graded, use_geometric_dead=False)
        dead = decode_dead(feats, model, graded)
        team[dead & graded] = 'D'
        if args.protect_hidden_live:
            baseline = splice_vc_on_visible(feats, graded=graded)
            team = protect_hidden_live(team, baseline, feats['ball_present'])
        if args.live_islands:
            from short_spell_experiment import suppress_live_islands
            team = suppress_live_islands(team, graded, 25, feats['ball_present'])
        write_submission(args.generate, team)
        print(json.dumps(live_structure(team, graded), indent=2))
        return
    if args.metrica:
        from gap_reconcile_experiment import cached_features
        from metrica_gt import build_metrica_gt
        from fetch_metrica import DEFAULT_OUT
        feats = cached_features(args.match)
        gt = build_metrica_gt(args.match, feats['n_frames'], root=DEFAULT_OUT)
        graded = np.ones(feats['n_frames'], bool)
    else:
        feats, gt, graded = cached_sportec(args.match)
    rows = []
    if args.legacy_dropouts:
        from dropout import dropout_configs
        specs = list(dropout_configs(gt, feats))
    else:
        specs = [(str(sec), np.zeros(feats['n_frames'], bool) if sec == 0 else
                  block_dropout(feats['n_frames'], sec)) for sec in (0, 2, 5, 10, 30)]
    for sec, hide in specs:
        f = apply_dropout(feats, hide)
        kw = kwargs_for_version(1, 8)
        baseline = run_labeller(f, graded, kw)
        kw['use_geometric_dead'] = False
        team = run_labeller(f, graded, kw)
        dead = decode_dead(f, model, graded)
        pred = team.copy()
        pred[dead & graded] = 'D'
        from short_spell_experiment import suppress_live_islands
        post = suppress_live_islands(pred, graded, 25, f['ball_present'])
        protected = protect_hidden_live(pred, baseline, f['ball_present'])
        for name, labels in [('latest', baseline), ('learned_dead', pred),
                             ('learned_dead_island25', post), ('learned_dead_protected', protected)]:
            rows.append(dict(game=args.match, dropout=str(sec), candidate=name,
                             **score_one(gt, labels, hide | ~feats['ball_present'])))
        print(args.match, sec, flush=True)
    suffix = '_mixed' if args.mixed_training else ''
    suffix += '_legacy' if args.legacy_dropouts else ''
    write_csv('evaluation/reports/learned_dead_' + args.match + suffix + '.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
