"""Motion candidates trained on prior training matches; fresh test split locked.

Training: Sportec J03WOH + Metrica Game 1, never challenge labels.
Development: J03WN1 + Metrica Game 2.
Fresh validation: J03WMX + J03WPY. Reserve J03WOY/J03WQQ/J03WR9 until needed.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sportec_experiment import cached_sportec, block_dropout
from motion_features import load_external_motion
from gap_reconcile_experiment import cached_features
from metrica_gt import build_metrica_gt
from fetch_metrica import DEFAULT_OUT
from dropout import apply_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one,
                           write_csv, aggregate_by_candidate)
from dead_ball_model import fit_logistic, protect_hidden_live
from motion_dead_model import design_motion_features, predict_motion_dead

REPORTS = Path('evaluation/reports')


def dataset(game):
    if game == 'Sample_Game_3':
        from metrica3_geometry import cached_geometry, ROOT
        from metrica3_evaluation import reference
        f, _ = cached_geometry()
        truth = reference(ROOT, f['n_frames'])
        graded = np.ones(f['n_frames'], bool)
    elif game.startswith('Sample_Game_'):
        f = cached_features(game)
        truth = build_metrica_gt(game, f['n_frames'], root=DEFAULT_OUT)
        graded = np.ones(f['n_frames'], bool)
    else:
        f, truth, graded = cached_sportec(game)
    if 'player_motion' not in f:
        f['player_motion'] = load_external_motion(game)
    f['game'] = game
    return f, truth, graded


def training_data():
    cache = REPORTS/'motion_training_v1.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            return archive['x'].copy(), archive['y'].copy()
    xs, ys = [], []
    for game in ('J03WOH', 'Sample_Game_1'):
        f, gt, graded = dataset(game)
        for sec in (0, 5, 15):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec, seed=73)
            dropped = apply_dropout(f, hide)
            x = design_motion_features(dropped)
            take = np.flatnonzero(graded & (gt != 'X'))[::5]
            xs.append(x[take])
            ys.append((gt[take] == 'D').astype(float))
        print('training features', game, flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def train_models():
    x, y = training_data()
    models = {}
    for name, ball in (('joint', True), ('motion', False)):
        selected = x if ball else x[:, 41:]
        model = fit_logistic(selected, y)
        np.savez(REPORTS/('dead_motion_'+name+'.npz'), **model)
        models[name] = model
        print('fit', name, len(y), selected.shape[1], flush=True)
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WN1')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.train or not (REPORTS/'dead_motion_joint.npz').exists():
        models = train_models()
    else:
        models = {}
        for name in ('joint', 'motion'):
            with np.load(REPORTS/('dead_motion_'+name+'.npz')) as archive:
                models[name] = dict(archive)
    f, gt, graded = dataset(args.match)
    rows = []
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec)
        dropped = apply_dropout(f, hide)
        v18 = kwargs_for_version(1, 8)
        old = run_labeller(dropped, graded, v18)
        latest = run_labeller(dropped, graded, kwargs_for_version(1, 9))
        v18['use_geometric_dead'] = False
        team = run_labeller(dropped, graded, v18)
        candidates = {'latest': latest}
        for name, model in models.items():
            dead = predict_motion_dead(dropped, model, graded, include_ball=name == 'joint')
            pred = team.copy()
            pred[dead & graded] = 'D'
            candidates[name] = pred
            candidates[name+'_protected'] = protect_hidden_live(pred, old, dropped['ball_present'])
        for name, pred in candidates.items():
            rows.append(dict(game=args.match, dropout=str(sec), candidate=name,
                             **score_one(gt, pred, hide | ~f['ball_present'])))
        print(args.match, sec, flush=True)
    write_csv(str(REPORTS/('motion_dead_'+args.match+'.csv')), rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
