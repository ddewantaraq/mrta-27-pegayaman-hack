"""Observed-ball endpoint context, independent of training framework or labels."""
import numpy as np

# Process-local override for boundary retrain rung D. Keep False for shipped models.
USE_TRANSITION_CONTEXT = False


def _endpoint_xy(features):
    present = np.asarray(features['ball_present'], bool)
    n = len(present)
    idx = np.arange(n)
    previous = np.maximum.accumulate(np.where(present, idx, -1))
    following = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    xy = np.column_stack([features['ball_x_norm'], features['ball_y_norm']])
    before = xy[np.clip(previous, 0, n - 1)].copy()
    after = xy[np.clip(following, 0, n - 1)].copy()
    before[previous < 0], after[following == n] = .5, .5
    return before, after, present, previous, following, idx, n


def _delta(arr, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    return out


def _transition_extras(features, window=5):
    """Ball/pressure transition cues packed into 4 channels (same width as v1 extras)."""
    n = len(features['ball_present'])
    speed = np.nan_to_num(np.asarray(features.get('ball_speed'), dtype=np.float64), nan=0.)
    if speed.shape[0] != n:
        speed = np.zeros(n, np.float64)
    bx = np.nan_to_num(np.asarray(features.get('ball_x_m'), dtype=np.float64), nan=0.)
    by = np.nan_to_num(np.asarray(features.get('ball_y_m'), dtype=np.float64), nan=0.)
    dx, dy = _delta(bx), _delta(by)
    # Direction change: angle between consecutive velocity vectors, in [-1, 1].
    v0 = np.column_stack([dx[:-1], dy[:-1]])
    v1 = np.column_stack([dx[1:], dy[1:]])
    n0 = np.linalg.norm(v0, axis=1)
    n1 = np.linalg.norm(v1, axis=1)
    denom = np.maximum(n0 * n1, 1e-6)
    cos = np.clip((v0 * v1).sum(axis=1) / denom, -1., 1.)
    ang = np.zeros(n, np.float64)
    ang[1:] = np.arccos(cos) / np.pi
    speed_d = _delta(speed)
    # Nearest-opponent pressure: min away dist when home nearer, else min home.
    d_h = np.nan_to_num(np.asarray(features.get('min_dist_home'), dtype=np.float64), nan=99.)
    d_a = np.nan_to_num(np.asarray(features.get('min_dist_away'), dtype=np.float64), nan=99.)
    nearest = np.asarray(features.get('nearest_team'), dtype=np.float64)
    pressure = np.where(nearest == 0, d_a, np.where(nearest == 1, d_h, np.minimum(d_h, d_a)))
    # Rolling Δ pressure over ±window (future-minus-past mean).
    kern = np.ones(window, dtype=np.float64) / float(window)
    past = np.convolve(pressure, kern, mode='same')
    future = np.convolve(pressure[::-1], kern, mode='same')[::-1]
    pressure_d = future - past
    # Closing: approximate with -Δ nearest_dist (positive = approaching).
    nd = np.nan_to_num(np.asarray(features.get('nearest_dist'), dtype=np.float64), nan=99.)
    closing = -_delta(nd)
    # Normalize into roughly unit ranges.
    return np.column_stack([
        np.tanh(speed_d / 5.),
        np.clip(ang, 0., 1.),
        np.tanh(pressure_d / 3.),
        np.tanh(closing / 3.),
    ]).astype(np.float32)


def ball_context(features):
    before, after, present, previous, following, idx, n = _endpoint_xy(features)
    if USE_TRANSITION_CONTEXT:
        extras = _transition_extras(features)
    else:
        extras = np.column_stack([
            np.minimum(idx - previous, 750) / 750.,
            np.minimum(following - idx, 750) / 750.,
            present.astype(float),
            (idx - previous) / np.maximum(following - previous, 1),
        ]).astype(np.float32)
    return np.column_stack([before, after, extras]).astype(np.float32)
