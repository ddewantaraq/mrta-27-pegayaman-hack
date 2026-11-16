"""Combined crossfit pipeline: visible team + dead + HSMM (+ v2.3 missing prior).

Milestone path toward 0.90 natural spellF1 / hidden accuracy.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from visible_ownership import (
    train_visible, export_trees, predict_logits_visible, design_visible_features)
from dead_crossfit import (
    train_dead, export_dead, predict_dead_logits)
from hsmm_decoder import fit_duration_from_games, decode_hsmm
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
from eval_external import write_csv
from state_machine import _merge_short_spells
import joblib


def v23_team_logits(features, graded, nodes, reference):
    residual = load_parameters(DEFAULT_RESIDUAL_MODEL_PATH)
    boundary_p = load_parameters(DEFAULT_BOUNDARY_MODEL_PATH)
    old = predict_logits(features, graded, nodes, reference, residual)
    newer, _ = predict_dual(features, graded, nodes, reference, boundary_p)
    return 0.5 * (old + newer)


def ensure_fold_models(holdout, train_games, fold_dir):
    fold_dir.mkdir(parents=True, exist_ok=True)
    vis_path = fold_dir / ('visible_' + holdout + '.json')
    dead_path = fold_dir / ('dead_' + holdout + '.json')
    dur_path = fold_dir / ('duration_' + holdout + '.npz')
    if not vis_path.exists():
        model = train_visible(train_games)
        joblib.dump(model, fold_dir / ('visible_' + holdout + '.joblib'))
        export_trees(model, vis_path, train_games)
    if not dead_path.exists():
        model, prior = train_dead(train_games)
        joblib.dump(model, fold_dir / ('dead_' + holdout + '.joblib'))
        export_dead(model, dead_path, train_games, prior=prior)
    if not dur_path.exists():
        logp = fit_duration_from_games(load_match, train_games, max_len=100)
        np.savez_compressed(dur_path, H=logp['H'], A=logp['A'], D=logp['D'])
    return vis_path, dead_path, dur_path


def load_duration(path):
    with np.load(path) as data:
        return {s: data[s].copy() for s in ('H', 'A', 'D')}


def predict_pipeline(features, graded, nodes, vis_path, dead_path, dur_path,
                     minimum=18, max_len=80):
    # Baseline reference for residual prior / missing fill.
    reference = predict_version(features, graded, (2, 3))
    vis_logits = predict_logits_visible(features, vis_path)
    dead_logits = predict_dead_logits(features, dead_path)
    miss_logits = v23_team_logits(features, graded, nodes, reference)
    present = np.asarray(features['ball_present'], bool)
    team_logits = np.where(present, np.nan_to_num(vis_logits, nan=0.), miss_logits)
    duration = load_duration(dur_path)
    labels = decode_hsmm(
        team_logits, dead_logits, graded, features.get('period'),
        duration_logp=duration, max_len=max_len, stride=5)
    return _merge_short_spells(labels, graded, minimum)


def crossfit_pipeline(matches=None):
    matches = matches or MATCHES
    fold_dir = REPORTS / 'crossfit_pipeline'
    rows = []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        print('pipeline fold', holdout, flush=True)
        vis, dead, dur = ensure_fold_models(holdout, train_games, fold_dir)
        data = load_match(holdout)
        pred = predict_pipeline(
            data['features'], data['graded'], data['nodes'], vis, dead, dur)
        metrics = score_natural(data['gt'], pred, data['features'])
        rows.append(dict(game=holdout, candidate='pipeline_090', **metrics))
        print(' ', holdout, 'spell', round(metrics['spell_f1_mean'], 4),
              'hid', metrics['hidden_team_acc'], flush=True)
    return rows


def train_all10():
    out = REPORTS / 'pipeline_090_all10'
    out.mkdir(parents=True, exist_ok=True)
    vis = train_visible(MATCHES)
    joblib.dump(vis, out / 'visible.joblib')
    export_trees(vis, out / 'visible.json', MATCHES)
    dead, prior = train_dead(MATCHES)
    joblib.dump(dead, out / 'dead.joblib')
    export_dead(dead, out / 'dead.json', MATCHES, prior=prior)
    logp = fit_duration_from_games(load_match, MATCHES, max_len=100)
    np.savez_compressed(out / 'duration.npz', H=logp['H'], A=logp['A'], D=logp['D'])
    print('trained all-10 into', out, flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    args = parser.parse_args()
    if args.train_all:
        train_all10()
    if args.crossfit or not args.train_all:
        rows = crossfit_pipeline()
        write_csv(REPORTS / 'crossfit_pipeline.csv', rows)
        summary = summarize_rows(rows)
        # Compare to v23 anchor if available
        anchor = REPORTS / 'crossfit_anchors.json'
        extra = {}
        if anchor.exists():
            extra['v23'] = json.loads(anchor.read_text())['aggregate'].get('v23')
        Path(REPORTS / 'crossfit_pipeline.json').write_text(
            json.dumps(dict(aggregate=summary, v23_anchor=extra.get('v23'),
                            rows=rows), indent=2))
        print(json.dumps(dict(pipeline=summary, v23=extra.get('v23')), indent=2))


if __name__ == '__main__':
    main()
