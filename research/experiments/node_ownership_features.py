"""Player-node ownership features: top-3-per-team + control/logit context."""
import numpy as np

from team_control_model import control_design


FEATURE_VERSION = 'node_correct_v1'
LENGTH = 105.
WIDTH = 68.
TOP_K = 3
# Per player: x, y, vx, vy, speed, seen, closing, dist
PLAYER_DIM = 8


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def _top_k_team(nodes_t, ball_xy, k=TOP_K):
    """nodes_t: (11, 8) attack-oriented; ball_xy: (2,) attack-oriented norm."""
    seen = nodes_t[:, 6] > 0.3
    pos = nodes_t[:, :2]
    vel = nodes_t[:, 2:4]  # already /10
    dist = np.linalg.norm(pos - ball_xy[None, :], axis=1)
    dist = np.where(seen, dist, 99.)
    order = np.argsort(dist, kind='stable')[:k]
    out = np.zeros((k, PLAYER_DIM), np.float64)
    for i, j in enumerate(order):
        if not seen[j] or dist[j] >= 90:
            continue
        rel = pos[j] - ball_xy
        d = max(float(dist[j]), 1e-3)
        closing = -float(np.dot(rel, vel[j])) / d  # >0 when approaching
        out[i] = [
            pos[j, 0], pos[j, 1], vel[j, 0], vel[j, 1],
            nodes_t[j, 4], 1., closing, d,
        ]
    return out


def node_summaries_5hz(features, nodes, stride=5):
    """Return (T5, 2*TOP_K*PLAYER_DIM + TOP_K*PLAYER_DIM diffs) at 5 Hz."""
    n = len(features['ball_present'])
    frames = np.arange(0, n, stride)
    t5 = len(frames)
    if nodes.shape[0] < t5:
        raise ValueError('nodes shorter than expected 5 Hz grid')
    bx = np.nan_to_num(features['ball_x_norm'], nan=0.5)
    by = np.nan_to_num(features['ball_y_norm'], nan=0.5)
    home = np.zeros((t5, TOP_K, PLAYER_DIM), np.float64)
    away = np.zeros((t5, TOP_K, PLAYER_DIM), np.float64)
    for i, f in enumerate(frames):
        nn = nodes[i]
        # Team signs: use slot 0 sign when seen, else +1.
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
    diffs = flat_h - flat_a
    return np.concatenate([flat_h, flat_a, diffs], axis=1)


def upsample_period_safe(values_5hz, n_frames, period, graded, stride=5):
    """Period-safe linear interp of 5 Hz rows onto 25 Hz (residual pattern)."""
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
            rows = values_5hz[ff // 5]
            for col in range(values_5hz.shape[1]):
                out[a:b, col] = np.interp(np.arange(a, b), ff, rows[:, col])
        a = b
    return out


def design_node_corrector_features(features, baseline, team_logits, nodes,
                                   graded=None):
    """Concatenate control_design extras with upsampled top-3 node summaries."""
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
