"""Geometry-only, anonymous player sets for temporal possession inference."""
import numpy as np


def pack_player_nodes(xy, teams, period, stride=5, length=105., width=68.):
    """5 Hz (time, team, 11, 8) nodes: x/y, vx/vy, speed, motion, seen, sign.

    Each team is oriented toward its attacking goal using geometry alone.
    No identity/ownership/status annotations are accepted. Slot order carries
    no semantics; downstream encoders must pool players permutation-invariantly.
    """
    xy = np.asarray(xy, np.float32)
    teams = np.asarray(teams)
    period = np.asarray(period)
    n = len(xy)
    valid = np.isfinite(xy).all(axis=2)
    valid &= (xy[:, :, 0] >= -.1*length) & (xy[:, :, 0] <= 1.1*length)
    valid &= (xy[:, :, 1] >= -.1*width) & (xy[:, :, 1] <= 1.1*width)
    velocity = np.zeros_like(xy)
    motion = np.zeros(valid.shape, bool)
    lag = 5
    if n > 2*lag:
        v = (xy[2*lag:]-xy[:-2*lag])*2.5
        speed = np.linalg.norm(v, axis=2)
        same = (period[2*lag:] == period[:-2*lag]) & (period[2*lag:] == period[lag:-lag])
        ok = valid[lag:-lag] & np.isfinite(speed) & (speed <= 12.) & same[:, None]
        velocity[lag:-lag] = np.where(ok[:, :, None], v, 0.)
        motion[lag:-lag] = ok
    frames = np.arange(0, n, stride)
    out = np.zeros((len(frames), 2, 11, 8), np.float32)
    for p in np.unique(period):
        if p == 0:
            continue
        full = np.flatnonzero(period == p)
        leftmost = []
        for team in (0, 1):
            positions = xy[full][:, teams == team, 0]
            seen = valid[full][:, teams == team]
            nearest_goal = np.min(np.where(seen, positions, np.inf), axis=1)
            finite = np.isfinite(nearest_goal)
            leftmost.append(float(np.median(nearest_goal[finite])) if finite.any() else length/2)
        home_attacks_right = leftmost[0] < leftmost[1]
        selection = np.flatnonzero(period[frames] == p)
        ff = frames[selection]
        for team in (0, 1):
            ids = np.flatnonzero(teams == team)
            if len(ids) == 0:
                continue
            seen = valid[ff][:, ids]
            # Valid positions precede padding; no semantic ordering is learned.
            take = np.argsort(~seen, axis=1, kind='stable')[:, :11]
            selected_ids = ids[take]
            pos = xy[ff[:, None], selected_ids]
            vel = velocity[ff[:, None], selected_ids]
            exists = valid[ff[:, None], selected_ids]
            has_motion = motion[ff[:, None], selected_ids]
            sign = 1. if (home_attacks_right == (team == 0)) else -1.
            values = np.zeros((len(ff), take.shape[1], 8), np.float32)
            values[:, :, 0] = pos[:, :, 0]/length if sign > 0 else 1-pos[:, :, 0]/length
            values[:, :, 1] = pos[:, :, 1]/width
            values[:, :, 2] = vel[:, :, 0]*sign/10.
            values[:, :, 3] = vel[:, :, 1]/10.
            values[:, :, 4] = np.linalg.norm(vel, axis=2)/10.
            values[:, :, 5] = has_motion
            values[:, :, 6] = exists
            values[:, :, 7] = sign
            values[~exists] = 0.
            out[selection, team, :take.shape[1]] = values
    return out
