"""Nonlinear live/dead emissions; fit only the locked external training split.

Requires requirements-research.txt. No random frame-level validation split:
temporal neighbours must not leak between training and validation.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset, training_data, REPORTS
from motion_dead_model import design_motion_features
from dead_ball_model import decode_dead_logits, protect_hidden_live
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one,
                           write_csv, aggregate_by_candidate)
from state_machine import _bridge_short_dead, _merge_short_spells

MODEL = REPORTS/'boosted_motion_v1.joblib'


def train():
    x, y = training_data()
    model = HistGradientBoostingClassifier(
        learning_rate=.1, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., max_bins=63,
        early_stopping=False, class_weight='balanced', random_state=73)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    joblib.dump(model, MODEL)
    print('fit boosted motion', len(y), x.shape[1], flush=True)
    return model


def sportec_training_data():
    cache = REPORTS/'motion_training_sportec_v1.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            return archive['x'].copy(), archive['y'].copy()
    xs, ys = [], []
    for game in ('J03WOH', 'J03WN1', 'J03WOY'):
        f, gt, graded = dataset(game)
        for sec in (0, 5, 15):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec, seed=73)
            x = design_motion_features(apply_dropout(f, hide))
            take = np.flatnonzero(graded & (gt != 'X'))[::5]
            xs.append(x[take])
            ys.append((gt[take] == 'D').astype(float))
        print('direct-status training', game, flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def train_sportec():
    x, y = sportec_training_data()
    model = HistGradientBoostingClassifier(
        learning_rate=.1, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., max_bins=63,
        early_stopping=False, class_weight='balanced', random_state=73)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    joblib.dump(model, REPORTS/'boosted_motion_sportec_v1.joblib')
    print('fit direct-status boosted motion', len(y), x.shape[1], flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WN1')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--sportec-training', action='store_true',
                        help='train only WOH/WN1/WOY direct live/dead labels; exclude all Metrica proxies')
    parser.add_argument('--reference-v2', action='store_true',
                        help='Metrica only: evaluate separate explicit-stoppage proxy; do not overwrite historical scores')
    parser.add_argument('--calibrate-prior', action='store_true',
                        help='undo balanced-class prior using the training-label prevalence only')
    args = parser.parse_args()
    path = REPORTS/'boosted_motion_sportec_v1.joblib' if args.sportec_training else MODEL
    fit = train_sportec if args.sportec_training else train
    model = fit() if args.train or not path.exists() else joblib.load(path)
    prior_shift = 0.
    if args.calibrate_prior:
        cache = REPORTS/('motion_training_sportec_v1.npz' if args.sportec_training
                        else 'motion_training_v1.npz')
        with np.load(cache) as archive:
            prior = float(np.mean(archive['y']))
        prior_shift = float(np.log(prior/(1-prior)))
        print('Training-prior correction', prior, prior_shift, flush=True)
    f, gt, graded = dataset(args.match)
    if args.reference_v2:
        if not args.match.startswith('Sample_Game_'):
            parser.error('--reference-v2 applies only to Metrica')
        from metrica_reference_v2 import build_metrica_reference_v2
        gt = build_metrica_reference_v2(args.match, f['n_frames'])
    rows = []
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec)
        dropped = apply_dropout(f, hide)
        v18 = kwargs_for_version(1, 8)
        old = run_labeller(dropped, graded, v18)
        latest = run_labeller(dropped, graded, kwargs_for_version(1, 9))
        v18['use_geometric_dead'] = False
        team = run_labeller(dropped, graded, v18)
        x = design_motion_features(dropped)
        with threadpool_limits(limits=4):
            logits = model.decision_function(x)+prior_shift
        dead = decode_dead_logits(logits, graded, dropped.get('period'))
        pred = team.copy()
        pred[dead & graded] = 'D'
        protected = protect_hidden_live(pred, old, dropped['ball_present'])
        post = _merge_short_spells(_bridge_short_dead(pred, graded, 25), graded, 18)
        for name, labels in [('latest', latest), ('boosted', pred),
                             ('boosted_protected', protected), ('boosted_post', post)]:
            rows.append(dict(game=args.match, dropout=str(sec), candidate=name,
                             **score_one(gt, labels, hide | ~f['ball_present'])))
        print(args.match, sec, flush=True)
    suffix = '_sportec' if args.sportec_training else ''
    suffix += '_reference_v2' if args.reference_v2 else ''
    suffix += '_prior' if args.calibrate_prior else ''
    write_csv(str(REPORTS/('boosted_motion_'+args.match+suffix+'.csv')), rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
