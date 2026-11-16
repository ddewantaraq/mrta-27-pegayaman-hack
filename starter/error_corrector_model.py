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

# sm_v2_6 / sm_v2_7: corrector fit and margin-tuned on fit matches only.
HONEST_CORRECTOR_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'error_corrector_v2.json')
HONEST_CORRECTOR_MARGIN = 2.5
# Demoted sm_v2_4 / sm_v2_5: corrector v1 trained on Track B/C matches too.
V25_CORRECTOR_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'error_corrector_v1.json')
V25_CORRECTOR_MARGIN = 3.0
V24_CORRECTOR_MARGIN = 1.5
DEFAULT_CORRECTOR_PATH = HONEST_CORRECTOR_PATH
DEFAULT_USE_ERROR_CORRECTOR = True
DEFAULT_CORRECTOR_MARGIN = HONEST_CORRECTOR_MARGIN
DEFAULT_CORRECTOR_WINDOW = 0
DEFAULT_CORRECTOR_MIN_ISLAND = 1
DEFAULT_CORRECTOR_STRUCTURE_GUARD = False
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


def _suppress_short_islands(flip_mask, labels, graded, min_island):
    """Drop flip runs shorter than min_island unless they abut D/X/ungraded."""
    if min_island <= 1 or not np.any(flip_mask):
        return flip_mask
    out = flip_mask.copy()
    n = len(flip_mask)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_island:
            left_barrier = (i == 0) or (not graded[i - 1]) or labels[i - 1] in ('D', 'X')
            right_barrier = (j == n) or (not graded[j]) or labels[j] in ('D', 'X')
            if not (left_barrier or right_barrier):
                out[i:j] = False
        i = j
    return out


def _window_agree(propose, propose_lab, corr, labels, margin, window):
    """Require ±window neighbors to share proposed team (or also be confident)."""
    if window <= 0 or not np.any(propose):
        return propose
    want = propose.copy()
    agree = np.ones(len(labels), bool)
    neighbor_ok = (np.abs(corr) >= float(margin)) | (labels == propose_lab)
    for d in range(1, int(window) + 1):
        same_l = np.ones(len(labels), bool)
        same_r = np.ones(len(labels), bool)
        same_l[d:] = propose_lab[d:] == propose_lab[:-d]
        same_r[:-d] = propose_lab[:-d] == propose_lab[d:]
        conf_l = np.ones(len(labels), bool)
        conf_r = np.ones(len(labels), bool)
        conf_l[d:] = neighbor_ok[:-d]
        conf_r[:-d] = neighbor_ok[d:]
        agree &= same_l & same_r & conf_l & conf_r
    want &= agree
    return want


def _structure_ok(cand, ref, graded):
    from vc_possession import live_structure
    c = live_structure(cand, graded)
    r = live_structure(ref, graded)
    return (c['short12'] <= r['short12']
            and c['short25'] <= r['short25']
            and c['mean_s'] + 1e-9 >= r['mean_s']
            and c['n_spells'] <= r['n_spells'] + 2)


def _flip_islands(mask):
    islands = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        islands.append((i, j))
        i = j
    return islands


def _apply_with_structure_guard(reference, graded, want, corr_home, corr, minimum):
    """Drop lowest-|corr| flip islands until structure ≤ reference, else no-op."""
    ref = np.asarray(reference)
    graded = np.asarray(graded, bool)
    want = want.copy()
    while np.any(want):
        labels = ref.copy()
        labels[want] = np.where(corr_home[want], 'H', 'A')
        labels = _merge_short_spells(labels, graded, minimum)
        if _structure_ok(labels, ref, graded):
            return labels
        islands = _flip_islands(want)
        if not islands:
            break
        # Drop the least-confident island (lowest mean |corr|).
        scores = [float(np.mean(np.abs(corr[a:b]))) for a, b in islands]
        drop_i = int(np.argmin(scores))
        a, b = islands[drop_i]
        want[a:b] = False
    return _merge_short_spells(ref.copy(), graded, minimum)


def apply_error_corrector(features, reference, graded, nodes,
                          model_path=DEFAULT_CORRECTOR_PATH,
                          margin=DEFAULT_CORRECTOR_MARGIN,
                          window=DEFAULT_CORRECTOR_WINDOW,
                          min_island=DEFAULT_CORRECTOR_MIN_ISLAND,
                          structure_guard=DEFAULT_CORRECTOR_STRUCTURE_GUARD,
                          minimum=18):
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
    propose = live & (corr_home != base_home) & (np.abs(corr) >= float(margin))
    propose_lab = np.where(corr_home, 'H', 'A')
    want = _window_agree(propose, propose_lab, corr, labels, margin, int(window))
    want = _suppress_short_islands(want, labels, graded, int(min_island))
    if structure_guard:
        return _apply_with_structure_guard(
            labels, graded, want, corr_home, corr, minimum)
    labels[want] = np.where(corr_home[want], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)
