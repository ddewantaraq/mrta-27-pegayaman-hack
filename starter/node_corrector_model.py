"""NumPy inference for the player-node live-team corrector (sm_v2_5)."""
from pathlib import Path
import json
import numpy as np

from team_control_model import control_design
from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells
from error_corrector_model import ensemble_team_logits

DEFAULT_NODE_CORRECTOR_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'node_corrector_v1.json')
DEFAULT_USE_NODE_CORRECTOR = False  # held: failed gate vs sm_v2_4
DEFAULT_NODE_MARGIN = 1.5
DEFAULT_NODE_WINDOW = 5
DEFAULT_NODE_MIN_ISLAND = 8
FEATURE_VERSION = 'node_correct_v1'
TOP_K = 3
PLAYER_DIM = 8


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def _top_k_team(nodes_t, ball_xy, k=TOP_K):
    seen = nodes_t[:, 6] > 0.3
    pos = nodes_t[:, :2]
    vel = nodes_t[:, 2:4]
    dist = np.linalg.norm(pos - ball_xy[None, :], axis=1)
    dist = np.where(seen, dist, 99.)
    order = np.argsort(dist, kind='stable')[:k]
    out = np.zeros((k, PLAYER_DIM), np.float64)
    for i, j in enumerate(order):
        if not seen[j] or dist[j] >= 90:
            continue
        rel = pos[j] - ball_xy
        d = max(float(dist[j]), 1e-3)
        closing = -float(np.dot(rel, vel[j])) / d
        out[i] = [
            pos[j, 0], pos[j, 1], vel[j, 0], vel[j, 1],
            nodes_t[j, 4], 1., closing, d,
        ]
    return out


def node_summaries_5hz(features, nodes, stride=5):
    n = len(features['ball_present'])
    frames = np.arange(0, n, stride)
    t5 = len(frames)
    bx = np.nan_to_num(features['ball_x_norm'], nan=0.5)
    by = np.nan_to_num(features['ball_y_norm'], nan=0.5)
    home = np.zeros((t5, TOP_K, PLAYER_DIM), np.float64)
    away = np.zeros((t5, TOP_K, PLAYER_DIM), np.float64)
    for i, f in enumerate(frames):
        nn = nodes[min(i, len(nodes) - 1)]
        sign_h = nn[0, 0, 7] if nn[0, 0, 6] > 0.3 else (
            nn[0, :, 7][nn[0, :, 6] > 0.3][0] if np.any(nn[0, :, 6] > 0.3) else 1.)
        sign_a = nn[1, 0, 7] if nn[1, 0, 6] > 0.3 else (
            nn[1, :, 7][nn[1, :, 6] > 0.3][0] if np.any(nn[1, :, 6] > 0.3) else -1.)
        ball_h = np.array([bx[f] if sign_h > 0 else 1. - bx[f], by[f]])
        ball_a = np.array([bx[f] if sign_a > 0 else 1. - bx[f], by[f]])
        home[i] = _top_k_team(nn[0], ball_h)
        away[i] = _top_k_team(nn[1], ball_a)
    flat_h = home.reshape(t5, -1)
    flat_a = away.reshape(t5, -1)
    return np.concatenate([flat_h, flat_a, flat_h - flat_a], axis=1)


def upsample_period_safe(values_5hz, n_frames, period, graded, stride=5):
    out = np.zeros((n_frames, values_5hz.shape[1]), np.float64)
    period = np.asarray(period if period is not None else np.zeros(n_frames, np.int8))
    graded = np.asarray(graded, bool)
    a = 0
    while a < n_frames:
        if not graded[a]:
            a += 1
            continue
        b = a + 1
        while b < n_frames and graded[b] and period[b] == period[a]:
            b += 1
        ff = np.arange(((a + 4) // 5) * 5, b, 5)
        if len(ff):
            rows = values_5hz[np.clip(ff // 5, 0, len(values_5hz) - 1)]
            for col in range(values_5hz.shape[1]):
                out[a:b, col] = np.interp(np.arange(a, b), ff, rows[:, col])
        a = b
    return out


def design_node_corrector_features(features, baseline, team_logits, nodes,
                                   graded=None):
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
    n = len(present)
    if graded is None:
        graded = np.ones(n, bool)
    summaries = node_summaries_5hz(features, nodes)
    node_25 = upsample_period_safe(summaries, n, period, graded)
    return np.column_stack([base, extra, node_25]).astype(np.float64)


def load_node_corrector(path=DEFAULT_NODE_CORRECTOR_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported node corrector schema')
    arrays = []
    for tree in data['trees']:
        converted = {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                     for k, v in tree.items()}
        arrays.append(converted)
    return dict(baseline=float(data['baseline']), trees=arrays,
                apply=data.get('apply', {}))


def _suppress_short_islands(flip_mask, labels, graded, min_island):
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


def apply_node_corrector(features, reference, graded, nodes,
                         model_path=DEFAULT_NODE_CORRECTOR_PATH,
                         margin=None, window=None, min_island=None,
                         minimum=18):
    packed = load_node_corrector(model_path)
    apply_cfg = packed.get('apply') or {}
    if margin is None:
        margin = float(apply_cfg.get('margin', DEFAULT_NODE_MARGIN))
    if window is None:
        window = int(apply_cfg.get('window', DEFAULT_NODE_WINDOW))
    if min_island is None:
        min_island = int(apply_cfg.get('min_island', DEFAULT_NODE_MIN_ISLAND))
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    team_logits = ensemble_team_logits(features, graded, nodes, labels)
    corr = boosted_logits(
        design_node_corrector_features(features, labels, team_logits, nodes, graded),
        packed)
    present = np.asarray(features['ball_present'], bool)
    live = graded & present & np.isin(labels, ['H', 'A'])
    corr_home = corr >= 0
    base_home = labels == 'H'
    propose = live & (corr_home != base_home) & (np.abs(corr) >= float(margin))
    propose_lab = np.where(corr_home, 'H', 'A')
    want = propose.copy()
    if window > 0:
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
    want = _suppress_short_islands(want, labels, graded, int(min_island))
    labels[want] = np.where(corr_home[want], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)
