"""T27: conservative same-team pass-like travel candidates.

Feature-level detector used by state_machine._confirm_control as a pass counter.
Uses two observed ball contacts and a plausible straight-line travel speed.
A candidate is evidence only — not proof that a pass happened.
"""
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class PassCandidate:
    start_frame: int
    end_frame: int
    team: int
    distance_m: float
    speed_mps: float
    missing_frames: int


def detect_same_team_travel(features, fps=25.0, contact_radius_m=2.0,
                            min_distance_m=3.0, max_speed_mps=35.0,
                            max_gap_frames=50, min_gap_frames=1,
                            graded=None, dead_mask=None, max_intermediate_contacts=0):
    """Return candidate intervals [start_frame, end_frame] inclusive.

    Required feature arrays: ball_present, ball_x_m, ball_y_m, nearest_team,
    nearest_dist, n_frames. Team codes 0/1. Candidate endpoints are
    consecutive *contact events*, with a gap of at least min_gap_frames
    frames between them. Intervening close contacts invalidate the pair;
    this intentionally cannot infer player identities from team-only features.

    graded/dead_mask are optional boolean masks supplied by the caller (T29
    integration point). A candidate may not cross an ungraded/dead frame.
    """
    if fps <= 0 or contact_radius_m <= 0 or max_speed_mps <= 0 or min_distance_m < 0:
        raise ValueError('invalid physical parameters')
    if not (0 <= min_gap_frames <= max_gap_frames) or max_intermediate_contacts < 0:
        raise ValueError('invalid gap parameters')
    n = int(features['n_frames'])
    names = ('ball_present', 'ball_x_m', 'ball_y_m', 'nearest_team', 'nearest_dist')
    arrays = {k: np.asarray(features[k]) for k in names}
    if any(len(v) != n for v in arrays.values()):
        raise ValueError('feature array length mismatch')
    valid = np.ones(n, dtype=bool)
    for mask, label, negate in ((graded, 'graded', False), (dead_mask, 'dead_mask', True)):
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) != n:
                raise ValueError(label + ' length mismatch')
            valid &= ~mask if negate else mask
    bp, x, y, team, dist = (arrays[k] for k in names)
    contact = (bp.astype(bool) & valid & np.isfinite(x) & np.isfinite(y)
               & np.isfinite(dist) & (dist <= contact_radius_m)
               & np.isin(team, [0, 1]))
    # Identify the start and end of each continuous contact run.
    starts = np.flatnonzero(contact & ~np.r_[False, contact[:-1]])
    ends = np.flatnonzero(contact & ~np.r_[contact[1:], False])

    candidates = []

    # Only consecutive contact runs are considered.
    for i in range(len(starts) - 1):
        a = int(ends[i])
        b = int(starts[i + 1])

        gap = b - a - 1

        if gap < min_gap_frames or gap > max_gap_frames:
            continue

        if team[a] != team[b] or not valid[a:b + 1].all():
            continue

        # No contact should occur between consecutive runs.
        if np.any(contact[a + 1:b]):
            continue

        d = math.hypot(float(x[b] - x[a]), float(y[b] - y[a]))
        speed = d * fps / (b - a)

        if min_distance_m <= d and speed <= max_speed_mps:
            candidates.append(
                PassCandidate(a, b, int(team[a]), d, speed, gap)
            )

    return candidates
