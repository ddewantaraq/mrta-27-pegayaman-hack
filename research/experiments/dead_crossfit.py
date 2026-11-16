"""Cross-fitted boosted live/dead with ball stoppage cues."""
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
from motion_dead_model import design_motion_features
from dead_ball_model import decode_dead_logits
from dropout import apply_dropout
from sportec_experiment import block_dropout
from eval_external import write_csv
from state_machine import _bridge_short_dead, _merge_short_spells
from boosted_dead_model import boosted_logits
from export_boosted_model import REPORTS as _R  # noqa: F401

FEATURE_VERSION = 'dead_crossfit_v1'


def stoppage_cues(features):
    """Extra ball-based stoppage / restart cues appended to motion design."""
    present = np.asarray(features['ball_present'], bool).astype(np.float64)
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    bx = np.nan_to_num(features['ball_x_norm'], nan=0.5)
    by = np.nan_to_num(features['ball_y_norm'], nan=0.5)
    out_x = (bx < 0.02) | (bx > 0.98)
    out_y = (by < 0.02) | (by > 0.98)
    still = (speed < 0.5).astype(np.float64)
    # Simple still-run length (capped).
    run = np.zeros(len(speed))
    for i in range(1, len(speed)):
        run[i] = (run[i - 1] + 1) if still[i] else 0
    run = np.minimum(run, 250) / 250.
    return np.column_stack([
        present, still, run, out_x.astype(float), out_y.astype(float),
        (out_x | out_y).astype(float), speed,
        np.nan_to_num(features['nearest_dist'], nan=99.) / 20.,
    ]).astype(np.float32)


def design_dead_features(features):
    base = design_motion_features(features, include_ball=True)
    return np.column_stack([base, stoppage_cues(features)]).astype(np.float32)


def training_matrix(games, stride=5):
    xs, ys = [], []
    for game in games:
        data = load_match(game)
        f, gt, graded = data['features'], data['gt'], data['graded']
        for sec in (0, 5, 15):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(
                f['n_frames'], sec, seed=73)
            dropped = apply_dropout(f, hide)
            x = design_dead_features(dropped)
            take = np.flatnonzero(graded & (gt != 'X'))[::stride]
            xs.append(x[take])
            ys.append((gt[take] == 'D').astype(float))
        print('dead features', game, flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def train_dead(games, seed=73):
    x, y = training_matrix(games)
    model = HistGradientBoostingClassifier(
        learning_rate=.1, max_iter=150, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., max_bins=63,
        early_stopping=False, class_weight='balanced', random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    return model, float(np.mean(y))


def export_dead(estimator, path, training_matches, prior=None):
    correction = 0.
    if prior is not None and 0 < prior < 1:
        # Undo balanced-class prior toward training prevalence.
        correction = float(np.log(prior / (1 - prior)))
    trees = []
    for iteration in estimator._predictors:
        nodes = iteration[0].nodes
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf',
                      missing_left='missing_go_to_left')
        trees.append({key: nodes[source].tolist() for key, source in fields.items()})
    payload = dict(
        feature_version=FEATURE_VERSION,
        baseline=float(estimator._baseline_prediction[0, 0]) + correction,
        prior_log_odds_correction=correction,
        n_features=int(estimator.n_features_in_),
        training_matches=list(training_matches),
        trees=trees,
    )
    Path(path).write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    return path


def load_dead_model(path):
    data = json.loads(Path(path).read_text())
    arrays = []
    for tree in data['trees']:
        converted = {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                     for k, v in tree.items()}
        arrays.append(converted)
    return dict(baseline=float(data['baseline']), trees=arrays,
                feature_version=data['feature_version'])


def predict_dead_logits(features, model_path):
    model = load_dead_model(model_path)
    return boosted_logits(design_dead_features(features), model)


def apply_dead(features, baseline, graded, logits, minimum=18):
    labels = np.asarray(baseline).copy()
    dead = decode_dead_logits(logits, graded, features.get('period'))
    labels[dead & graded] = 'D'
    labels = _bridge_short_dead(labels, graded, 25)
    return _merge_short_spells(labels, graded, minimum)


def crossfit_dead(matches=None):
    matches = matches or MATCHES
    rows = []
    fold_dir = REPORTS / 'crossfit_dead'
    fold_dir.mkdir(parents=True, exist_ok=True)
    for holdout in matches:
        train_games = fold_train_games(holdout)
        path = fold_dir / ('dead_' + holdout + '.json')
        if not path.exists():
            model, prior = train_dead(train_games)
            joblib.dump(model, fold_dir / ('dead_' + holdout + '.joblib'))
            export_dead(model, path, train_games, prior=prior)
        data = load_match(holdout)
        # Start from geometric/vc baseline without boosted dead: use v1.8-like via
        # stripping boosted from v2.3 is heavy; instead overlay on v2.3 then
        # re-apply dead from new model after clearing D back to nearest team.
        base = predict_version(data['features'], data['graded'], (2, 3))
        # Replace D with v2.3's latent isn't available; keep non-D, re-decide D.
        liveish = base.copy()
        # For frames currently D, seed with nearest team so dead overlay can fire.
        nearest = data['features']['nearest_team']
        dead_mask = liveish == 'D'
        liveish[dead_mask] = np.where(nearest[dead_mask] == 0, 'H',
                                      np.where(nearest[dead_mask] == 1, 'A', 'X'))
        logits = predict_dead_logits(data['features'], path)
        pred = apply_dead(data['features'], liveish, data['graded'], logits)
        metrics = score_natural(data['gt'], pred, data['features'])
        rows.append(dict(game=holdout, candidate='dead_crossfit', **metrics))
        print('holdout', holdout, 'spell', metrics['spell_f1_mean'],
              'hid', metrics['hidden_team_acc'], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--out-model', default=str(REPORTS / 'dead_crossfit_all10.json'))
    args = parser.parse_args()
    if args.train_all:
        model, prior = train_dead(MATCHES)
        joblib.dump(model, REPORTS / 'dead_crossfit_all10.joblib')
        export_dead(model, args.out_model, MATCHES, prior=prior)
        print('trained', args.out_model, flush=True)
    if args.crossfit or not args.train_all:
        rows = crossfit_dead()
        write_csv(REPORTS / 'crossfit_dead.csv', rows)
        summary = summarize_rows(rows)
        Path(REPORTS / 'crossfit_dead.json').write_text(
            json.dumps(dict(aggregate=summary, rows=rows), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
