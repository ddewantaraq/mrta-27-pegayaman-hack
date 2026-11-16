"""NumPy inference for the cross-fitted v2.3 live-team error corrector."""
from pathlib import Path
import json
import numpy as np

from team_control_model import control_design
from boosted_dead_model import boosted_logits
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
from state_machine import _merge_short_spells

DEFAULT_CORRECTOR_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'error_corrector_v1.json')
DEFAULT_USE_ERROR_CORRECTOR = False  # demoted: Track A s25/mean vs sm_v2_3; keep v2.3 as LATEST
DEFAULT_CORRECTOR_MARGIN = 1.5
FEATURE_VERSION = 'error_correct_v1'


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def design_corrector_features(features, baseline, team_logits):
    base = control_design(features)
    present = features['ball_present'].astype(float)
    nearest = features['nearest_team'].astype(float)
    majority = features['majority_team'].astype(float)
    base_sign = np.where(baseline == 'H', 1., np.where(baseline == 'A', -1., 0.))
    nearest_lab = np.where(nearest == 0, 'H', np.where(nearest == 1, 'A', 'X'))
    majority_lab = np.where(majority == 0, 'H', np.where(majority == 1, 'A', 'X'))
    agree_n = (nearest_lab == baseline).astype(float)
    agree_m = (majority_lab == baseline).astype(float)
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


def load_corrector(path=DEFAULT_CORRECTOR_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported error corrector schema')
    arrays = []
    for tree in data['trees']:
        converted = {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                     for k, v in tree.items()}
        arrays.append(converted)
    return dict(baseline=float(data['baseline']), trees=arrays)


def ensemble_team_logits(features, graded, nodes, reference):
    residual = load_parameters(DEFAULT_RESIDUAL_MODEL_PATH)
    boundary_p = load_parameters(DEFAULT_BOUNDARY_MODEL_PATH)
    old = predict_logits(features, graded, nodes, reference, residual)
    newer, _ = predict_dual(features, graded, nodes, reference, boundary_p)
    return 0.5 * (old + newer)


def apply_error_corrector(features, reference, graded, nodes,
                          model_path=DEFAULT_CORRECTOR_PATH,
                          margin=DEFAULT_CORRECTOR_MARGIN, minimum=18):
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    team_logits = ensemble_team_logits(features, graded, nodes, labels)
    corr = boosted_logits(
        design_corrector_features(features, labels, team_logits),
        load_corrector(model_path))
    present = np.asarray(features['ball_present'], bool)
    live = graded & present & np.isin(labels, ['H', 'A'])
    corr_home = corr >= 0
    base_home = labels == 'H'
    disagree = live & (corr_home != base_home) & (np.abs(corr) >= float(margin))
    labels[disagree] = np.where(corr_home[disagree], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)
