"""Observed-ball endpoint context, independent of training framework or labels."""
import numpy as np


def ball_context(features):
    present = np.asarray(features['ball_present'], bool)
    n = len(present)
    idx = np.arange(n)
    previous = np.maximum.accumulate(np.where(present, idx, -1))
    following = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    xy = np.column_stack([features['ball_x_norm'], features['ball_y_norm']])
    before = xy[np.clip(previous, 0, n-1)].copy()
    after = xy[np.clip(following, 0, n-1)].copy()
    before[previous < 0], after[following == n] = .5, .5
    return np.column_stack([
        before, after, np.minimum(idx-previous, 750)/750.,
        np.minimum(following-idx, 750)/750., present.astype(float),
        (idx-previous)/np.maximum(following-previous, 1),
    ]).astype(np.float32)
