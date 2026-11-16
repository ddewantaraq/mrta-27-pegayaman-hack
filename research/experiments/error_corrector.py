"""Second-stage corrector: learn when v2.3's live team label is wrong.

Cross-fitted. Only flips H/A on visible live frames where the corrector is
confident the baseline is wrong.
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
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from team_control_model import control_design
from visible_ownership import _delta
from boosted_dead_model import boosted_logits
from eval_external import write_csv
from state_machine import _merge_short_spells
from pipeline_090 import v23_team_logits


FEATURE_VERSION = 'error_correct_v1'


def design_corrector(features, baseline, team_logits):
    base = control_design(features)
    present = features['ball_present'].astype(float)
    nearest = features['nearest_team'].astype(float)
    majority = features['majority_team'].astype(float)
    base_sign = np.where(baseline == 'H', 1., np.where(baseline == 'A', -1., 0.))
    agree_n = (np.where(nearest == 0, 'H', np.where(nearest == 1, 'A', 'X')) == baseline).astype(float)
    agree_m = (np.where(majority == 0, 'H', np.where(majority == 1, 'A', 'X')) == baseline).astype(float)
    logits = np.nan_to_num(team_logits, nan=0.)
    period = features.get('period')
    bx = np.nan_to_num(features['ball_x_m'], nan=0.)
    by = np.nan_to_num(features['ball_y_m'], nan=0.)
    extra = np.column_stack([
        present, base_sign, logits, np.abs(logits),
        agree_n, agree_m, nearest, majority,
        np.nan_to_num(features['nearest_dist'], nan=99.),
        np.nan_to_num(features['min_dist_home'], nan=99.),
        np.nan_to_num(features['min_dist_away'], nan=99.),
        bx, by, np.nan_to_num(features['ball_speed'], nan=0.),
        _delta(bx, period), _delta(by, period),
    ])
    return np.column_stack([base, extra]).astype(np.float64)


def training_matrix(games, stride=5):
    xs, ys = [], []
    for game in games:
        data = load_match(game)
        f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
        baseline = predict_version(f, graded, (2, 3))
        logits = v23_team_logits(f, graded, nodes, baseline)
        x = design_corrector(f, baseline, logits)
        # Predict whether GT is Home among visible live frames (relearn team).
        take = graded & f['ball_present'] & np.isin(gt, ['H', 'A']) & np.isin(baseline, ['H', 'A'])
        idx = np.flatnonzero(take)[::stride]
        xs.append(x[idx])
        ys.append((gt[idx] == 'H').astype(float))
        print('corrector features', game, len(idx), flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def train(games, seed=73):
    x, y = training_matrix(games)
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=250, max_leaf_nodes=63,
        min_samples_leaf=30, l2_regularization=1., max_bins=127,
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
    Path(path).write_text(json.dumps(dict(
        feature_version=FEATURE_VERSION,
        baseline=float(estimator._baseline_prediction[0, 0]),
        n_features=int(estimator.n_features_in_),
        training_matches=list(training_matches), trees=trees), separators=(',', ':')) + '\n')


def load_logits(features, baseline, team_logits, path):
    data = json.loads(Path(path).read_text())
    arrays = [{k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
               for k, v in tree.items()} for tree in data['trees']]
    packed = dict(baseline=float(data['baseline']), trees=arrays)
    return boosted_logits(design_corrector(features, baseline, team_logits), packed)


def apply_corrector(features, graded, nodes, model_path, margin=0.5, minimum=18):
    baseline = predict_version(features, graded, (2, 3))
    team_logits = v23_team_logits(features, graded, nodes, baseline)
    corr = load_logits(features, baseline, team_logits, model_path)
    labels = baseline.copy()
    present = features['ball_present']
    live = graded & present & np.isin(baseline, ['H', 'A'])
    # Only flip when corrector disagrees with baseline AND is confident.
    corr_home = corr >= 0
    base_home = baseline == 'H'
    disagree = live & (corr_home != base_home) & (np.abs(corr) >= margin)
    labels[disagree] = np.where(corr_home[disagree], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum), int(disagree.sum())


def crossfit(matches=None, margin=0.5):
    matches = matches or MATCHES
    fold_dir = REPORTS / 'crossfit_corrector'
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for holdout in matches:
        path = fold_dir / ('corr_' + holdout + '.json')
        train_games = fold_train_games(holdout)
        if not path.exists():
            model = train(train_games)
            joblib.dump(model, fold_dir / ('corr_' + holdout + '.joblib'))
            export_model(model, path, train_games)
        data = load_match(holdout)
        pred, nflips = apply_corrector(
            data['features'], data['graded'], data['nodes'], path, margin=margin)
        metrics = score_natural(data['gt'], pred, data['features'])
        # visible team acc after correction
        gt = data['gt']; present = data['features']['ball_present']
        live = present & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
        acc = float(np.mean(pred[live] == gt[live])) if live.any() else float('nan')
        rows.append(dict(game=holdout, candidate='corrector_m{}'.format(margin),
                         flips=nflips, visible_team_acc=acc, **metrics))
        print(holdout, 'flips', nflips, 'vis_acc', round(acc, 4),
              'spell', round(metrics['spell_f1_mean'], 4), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--train-all', action='store_true')
    args = parser.parse_args()
    if args.train_all:
        model = train(MATCHES)
        joblib.dump(model, REPORTS / 'error_corrector_all10.joblib')
        export_model(model, REPORTS / 'error_corrector_all10.json', MATCHES)
    if args.crossfit or not args.train_all:
        rows = crossfit(margin=args.margin)
        write_csv(REPORTS / 'crossfit_corrector.csv', rows)
        summary = summarize_rows(rows)
        Path(REPORTS / 'crossfit_corrector.json').write_text(
            json.dumps(dict(aggregate=summary, rows=rows), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
