"""Visible-ball ownership v2: HGB on control_design (+ ball kinematics)."""
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
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from team_control_model import control_design
from visible_ownership import _delta, export_trees as _exp  # reuse pattern
from boosted_dead_model import boosted_logits
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells
from eval_external import write_csv
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual, decode_boundary
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH

FEATURE_VERSION = 'visible_own_v2_control'


def design_v2(features):
    base = control_design(features)
    period = features.get('period')
    bx = np.nan_to_num(features['ball_x_m'], nan=0.)
    by = np.nan_to_num(features['ball_y_m'], nan=0.)
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    vx, vy = _delta(bx, period), _delta(by, period)
    extra = np.column_stack([bx, by, speed, vx, vy, _delta(vx, period), _delta(vy, period)])
    return np.column_stack([base, extra]).astype(np.float64)


def training_matrix(games, stride=5):
    xs, ys = [], []
    for game in games:
        data = load_match(game)
        f, gt, graded = data['features'], data['gt'], data['graded']
        x = design_v2(f)
        take = graded & f['ball_present'] & np.isin(gt, ['H', 'A'])
        idx = np.flatnonzero(take)[::stride]
        xs.append(x[idx]); ys.append((gt[idx] == 'H').astype(float))
        print('v2 features', game, len(idx), x.shape[1], flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def train(games, seed=73):
    x, y = training_matrix(games)
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=300, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=2., max_bins=127,
        early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model


def export_model(estimator, path, training_matches):
    trees = []
    for iteration in estimator._predictors:
        nodes = iteration[0].nodes
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf',
                      missing_left='missing_go_to_left')
        trees.append({key: nodes[source].tolist() for key, source in fields.items()})
    payload = dict(feature_version=FEATURE_VERSION,
                   baseline=float(estimator._baseline_prediction[0, 0]),
                   n_features=int(estimator.n_features_in_),
                   training_matches=list(training_matches), trees=trees)
    Path(path).write_text(json.dumps(payload, separators=(',', ':')) + '\n')


def load_logits(features, path):
    data = json.loads(Path(path).read_text())
    arrays = [{k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
               for k, v in tree.items()} for tree in data['trees']]
    return boosted_logits(design_v2(features), dict(baseline=float(data['baseline']), trees=arrays))


def blend_with_v23(features, graded, nodes, vis_logits, alpha=0.25):
    reference = predict_version(features, graded, (2, 3))
    residual = load_parameters(DEFAULT_RESIDUAL_MODEL_PATH)
    boundary_p = load_parameters(DEFAULT_BOUNDARY_MODEL_PATH)
    old = predict_logits(features, graded, nodes, reference, residual)
    newer, boundary = predict_dual(features, graded, nodes, reference, boundary_p)
    miss = 0.5 * (old + newer)
    present = np.asarray(features['ball_present'], bool)
    signal = miss.copy()
    signal[present] = (1 - alpha) * miss[present] + alpha * vis_logits[present]
    live = graded & np.isin(reference, ['H', 'A'])
    home = decode_boundary(signal, boundary, live, features.get('period'))
    home = home.copy()
    home[present] = signal[present] >= 0
    labels = reference.copy()
    labels[live] = np.where(home[live], 'H', 'A')
    return _merge_short_spells(labels, graded, 18)


def crossfit(matches=None, alpha=0.25):
    matches = matches or MATCHES
    fold_dir = REPORTS / 'crossfit_visible_v2'
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        path = fold_dir / ('visible_v2_' + holdout + '.json')
        if not path.exists():
            model = train(train_games)
            joblib.dump(model, fold_dir / ('visible_v2_' + holdout + '.joblib'))
            export_model(model, path, train_games)
        data = load_match(holdout)
        logits = load_logits(data['features'], path)
        # frame accuracy vs v23
        present = data['features']['ball_present']
        gt = data['gt']
        live = present & np.isin(gt, ['H', 'A'])
        raw = np.where(logits >= 0, 'H', 'A')
        vis_acc = float(np.mean(raw[live] == gt[live])) if live.any() else float('nan')
        pred = blend_with_v23(data['features'], data['graded'], data['nodes'], logits, alpha=alpha)
        metrics = score_natural(gt, pred, data['features'])
        rows.append(dict(game=holdout, candidate='visible_v2_a{}'.format(alpha),
                         visible_team_acc=vis_acc, **metrics))
        print(holdout, 'vis_acc', round(vis_acc, 4), 'spell', round(metrics['spell_f1_mean'], 4), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--train-all', action='store_true')
    args = parser.parse_args()
    if args.train_all:
        model = train(MATCHES)
        joblib.dump(model, REPORTS / 'visible_ownership_v2_all10.joblib')
        export_model(model, REPORTS / 'visible_ownership_v2_all10.json', MATCHES)
    if args.crossfit or not args.train_all:
        rows = crossfit(alpha=args.alpha)
        write_csv(REPORTS / 'crossfit_visible_v2.csv', rows)
        summary = summarize_rows(rows)
        Path(REPORTS / 'crossfit_visible_v2.json').write_text(
            json.dumps(dict(aggregate=summary, rows=rows), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
