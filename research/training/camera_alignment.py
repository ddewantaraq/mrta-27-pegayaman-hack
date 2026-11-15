"""Automatic two-camera ball alignment from geometry, never possession labels.

KD-tree trajectory retrieval proposes matches; temporal neighbourhood agreement
validates them. Fusion requires locally agreeing overlapping observations.
"""
import numpy as np


def trajectory_templates(xy, present, step=5, lag=25):
    centers = np.arange(lag, len(xy)-lag, step)
    valid = present[centers-lag] & present[centers] & present[centers+lag]
    centers = centers[valid]
    a, b, c = xy[centers-lag], xy[centers], xy[centers+lag]
    travel = np.linalg.norm(b-a, axis=1)+np.linalg.norm(c-b, axis=1)
    keep = travel >= 8.
    return centers[keep], np.column_stack([a[keep], b[keep], c[keep]])


def find_anchors(target_xy, target_present, source_xy, source_present, max_proposals=600):
    from scipy.spatial import cKDTree
    ti, tx = trajectory_templates(target_xy, target_present)
    si, sx = trajectory_templates(source_xy, source_present)
    if len(ti) == 0 or len(si) == 0:
        return np.empty((0, 3))
    distance, match = cKDTree(sx).query(tx, k=1, distance_upper_bound=10.)
    valid = np.flatnonzero(np.isfinite(distance))
    order = valid[np.argsort(distance[valid])]
    selected = []
    seen = set()
    for k in order:
        bucket = int(ti[k]//125)
        if bucket not in seen:
            seen.add(bucket)
            selected.append(k)
        if len(selected) >= max_proposals:
            break
    anchors = []
    delta = np.arange(-50, 51, 5)
    for k in selected:
        t = int(ti[k])
        source = int(si[match[k]])
        best = None
        for adjust in range(-7, 8):
            tt, ss = t+delta, source+adjust+delta
            in_bounds = (tt >= 0) & (tt < len(target_xy)) & (ss >= 0) & (ss < len(source_xy))
            tt, ss = tt[in_bounds], ss[in_bounds]
            observed = target_present[tt] & source_present[ss]
            if observed.sum() < 10:
                continue
            error = np.linalg.norm(target_xy[tt[observed]]-source_xy[ss[observed]], axis=1)
            median, high = np.quantile(error, [.5, .9])
            if median <= 3. and high <= 6. and (best is None or median < best[0]):
                best = (median, source+adjust)
        if best is not None:
            anchors.append((t, best[1], best[0]))
    return np.asarray(sorted(anchors), dtype=float).reshape(-1, 3)


def mapping_from_anchors(anchors, n_target, max_gap=1500, offset_tolerance=10):
    """Interpolate only between neighbouring anchors with consistent offsets."""
    mapping = np.full(n_target, -1, np.int64)
    if len(anchors) < 2:
        return mapping
    for left, right in zip(anchors[:-1], anchors[1:]):
        a, sa = map(int, left[:2])
        b, sb = map(int, right[:2])
        if b-a > max_gap or b <= a or abs((sb-b)-(sa-a)) > offset_tolerance:
            continue
        mapping[a:b+1] = np.rint(np.linspace(sa, sb, b-a+1)).astype(np.int64)
    return mapping


def _rolling_sum(a, radius):
    idx = np.arange(len(a))
    lo, hi = np.maximum(0, idx-radius), np.minimum(len(a), idx+radius+1)
    cumulative = np.r_[0., np.cumsum(a)]
    return cumulative[hi]-cumulative[lo]


def fuse_camera_ball(target_xy, target_present, source_xy, source_present, anchors):
    """Fill missing target points only with locally verified source geometry."""
    mapping = mapping_from_anchors(anchors, len(target_xy))
    mapped = (mapping >= 0) & (mapping < len(source_xy))
    source_index = np.clip(mapping, 0, len(source_xy)-1)
    both = mapped & target_present & source_present[source_index]
    errors = np.linalg.norm(target_xy-source_xy[source_index], axis=1)
    good = both & (errors <= 3.)
    overlap = _rolling_sum(both.astype(float), 25)
    agreeing = _rolling_sum(good.astype(float), 25)
    local = (overlap >= 10) & (agreeing >= .8*overlap)
    velocity = np.full(len(source_xy), np.inf)
    adjacent = source_present[1:] & source_present[:-1]
    velocity[1:] = np.where(adjacent, np.linalg.norm(np.diff(source_xy, axis=0), axis=1)*25., np.inf)
    fill = (mapped & ~target_present & source_present[source_index] & local
            & (velocity[source_index] <= 45.))
    xy = np.asarray(target_xy).copy()
    xy[fill] = source_xy[source_index[fill]]
    return xy, target_present | fill, fill, mapping
