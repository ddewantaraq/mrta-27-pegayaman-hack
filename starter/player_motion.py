"""Permutation-invariant player-motion features, available without the ball.

Motivated by Lang et al. (2022), doi:10.1038/s41598-022-19948-1.
Uses robust aggregates rather than requiring persistent player identities.
"""
import numpy as np


def summarize_motion(xy, period=None, fps=25., lag=5):
    """Return n x 12 features from n x players x 2 metre coordinates.

    Velocity spans 2*lag frames. Missing coordinates, period boundaries and
    implausible >12 m/s displacements are excluded, not converted into motion.
    No ball coordinates or labels are accepted by this function.
    """
    xy = np.asarray(xy, dtype=np.float32)
    n, players, _ = xy.shape
    result = np.zeros((n, 12), dtype=np.float32)
    pos_ok = np.isfinite(xy).all(axis=2)
    count_pos = pos_ok.sum(axis=1)
    result[:, 8] = np.minimum(count_pos/22., 1.5)
    for axis in range(2):
        values = np.where(pos_ok, xy[:, :, axis], 0.)
        avg = values.sum(axis=1)/np.maximum(count_pos, 1)
        variance = np.maximum(0., (values*values).sum(axis=1)/np.maximum(count_pos, 1)-avg*avg)
        result[:, 10+axis] = np.sqrt(variance)/(105. if axis == 0 else 68.)
    if n <= 2*lag:
        return result
    speed = np.full((n, players), np.nan, np.float32)
    v = np.linalg.norm(xy[2*lag:]-xy[:-2*lag], axis=2)*(fps/(2*lag))
    valid = pos_ok[lag:-lag] & np.isfinite(v) & (v <= 12.)
    if period is not None:
        period = np.asarray(period)
        same = (period[2*lag:] == period[:-2*lag]) & (period[2*lag:] == period[lag:-lag])
        valid &= same[:, None]
    speed[lag:-lag] = np.where(valid, v, np.nan)
    valid = np.isfinite(speed)
    count = valid.sum(axis=1)
    denom = np.maximum(count, 1)
    result[:, 0] = np.nansum(speed, axis=1)/denom/10.
    ordered = np.sort(np.where(valid, speed, np.inf), axis=1)
    rows = np.arange(n)
    for column, quantile in enumerate((.25, .5, .75, .9), 1):
        fractional = quantile*np.maximum(count-1, 0)
        low = np.floor(fractional).astype(int)
        high = np.ceil(fractional).astype(int)
        # Index values are finite whenever count > 0.
        a = np.where(count > 0, ordered[rows, low], 0.)
        b = np.where(count > 0, ordered[rows, high], 0.)
        result[:, column] = (a+(b-a)*(fractional-low))/10.
    result[:, 5] = np.sum(valid & (speed > 2.), axis=1)/denom
    result[:, 6] = np.sum(valid & (speed > 4.), axis=1)/denom
    result[:, 7] = np.sum(valid & (speed < .5), axis=1)/denom
    result[:, 9] = np.minimum(count/22., 1.5)
    return result
