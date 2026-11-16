"""External-supervised, team-symmetric possession emissions and temporal decode."""
import numpy as np

from dead_ball_model import _mean_window, decode_dead_logits


def control_design(features):
    present = np.asarray(features['ball_present'], bool)
    n = len(present)
    idx = np.arange(n)
    nearest = np.asarray(features['nearest_team'])
    nearest_sign = np.where(nearest == 0, 1., np.where(nearest == 1, -1., 0.))
    zone = np.asarray(features['pz_team'])
    zone_sign = np.where(zone == 0, 1., np.where(zone == 1, -1., 0.))
    difference = np.nan_to_num((features['min_dist_away']-features['min_dist_home'])/5.)
    nh, na = features['n_near_home'], features['n_near_away']
    distance = np.nan_to_num(features['nearest_dist'], nan=99.)
    raw = np.column_stack([
        nearest_sign, nearest_sign*np.exp(-distance/3.),
        np.clip(difference, -1., 1.),
        (nh.astype(float)-na)/(nh.astype(float)+na+1.), zone_sign,
    ])*present[:, None]
    previous = np.maximum.accumulate(np.where(present, idx, -1))
    following = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    a = raw[np.clip(previous, 0, n-1)].copy()
    b = raw[np.clip(following, 0, n-1)].copy()
    a[previous < 0], b[following == n] = 0., 0.
    fraction = (idx-previous)/np.maximum(following-previous, 1)
    parts = [raw, a, b, a*(1-fraction[:, None]), b*fraction[:, None]]
    weight = present.astype(float)
    for radius in (12, 25, 50, 125):
        coverage = _mean_window(weight, radius)
        parts.append(np.column_stack([
            _mean_window(raw[:, k], radius)/np.maximum(coverage, 1e-6)
            for k in range(raw.shape[1])]))
    return np.column_stack(parts).astype(np.float32)


def predict_teams(features, model, graded):
    x = control_design(features)
    logits = model['beta'][0]+((x-model['mean'])/model['scale'])@model['beta'][1:]
    home = decode_dead_logits(logits, graded, features.get('period'))
    return np.where(home, 'H', 'A')
