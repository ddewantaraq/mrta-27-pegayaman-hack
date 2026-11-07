"""Deterministic ball-dropout masks for external validation.

Hides the ball on live-play windows (where derived GT is H or A) so we can
measure missing-ball H/A correctness. Gap lengths are a fixed grid in seconds;
no BAR-ATH-specific constants.
"""

from __future__ import print_function

import numpy as np

FPS = 25
DEFAULT_GAP_SECONDS = (2, 5, 10, 30)
DEFAULT_SEED = 0


def apply_dropout(features, hide_mask):
    """Return a shallow-copied features dict with ball hidden on hide_mask."""
    out = dict(features)
    bp = np.asarray(features["ball_present"], dtype=bool).copy()
    hide = np.asarray(hide_mask, dtype=bool)
    if len(hide) != len(bp):
        raise ValueError("hide_mask length {} != n_frames {}".format(
            len(hide), len(bp)))
    bp[hide] = False
    out["ball_present"] = bp
    # Clear ball coords / nearest on hidden frames so the state machine cannot
    # accidentally read a "present" signal from leftover fields. nearest_* are
    # only consulted when ball_present is True, but zeroing keeps audits clean.
    for key in ("ball_x_norm", "ball_y_norm", "ball_speed",
                "ball_x_m", "ball_y_m", "nearest_dist",
                "min_dist_home", "min_dist_away"):
        if key in out and out[key] is not None:
            arr = np.asarray(out[key], dtype=np.float64).copy()
            arr[hide] = np.nan
            out[key] = arr
    for key in ("nearest_team", "majority_team", "n_near_home", "n_near_away"):
        if key in out and out[key] is not None:
            arr = np.asarray(out[key]).copy()
            arr[hide] = -1 if key.endswith("team") else 0
            out[key] = arr
    return out


def natural_missing_mask(features):
    """Frames where the tracking ball was already absent."""
    return ~np.asarray(features["ball_present"], dtype=bool)


def synthetic_gap_mask(gt_labels, gap_frames, seed=DEFAULT_SEED, stride_factor=8):
    """Place non-overlapping gaps of length gap_frames on live (H/A) GT frames.

    Starts every stride_factor * gap_frames along the timeline; only places a gap
    when the whole window is H or A (alive). Deterministic via seed (used only
    to offset the first start so different gap lengths don't share edges).
    """
    n = len(gt_labels)
    mask = np.zeros(n, dtype=bool)
    if gap_frames <= 0 or gap_frames > n:
        return mask
    rng = np.random.RandomState(int(seed) + int(gap_frames))
    offset = int(rng.randint(0, max(1, gap_frames)))
    stride = max(gap_frames * int(stride_factor), gap_frames + 1)
    alive = (gt_labels == "H") | (gt_labels == "A")
    start = offset
    while start + gap_frames <= n:
        window = slice(start, start + gap_frames)
        if bool(np.all(alive[window])):
            mask[window] = True
            start += stride
        else:
            start += max(1, gap_frames // 4)
    return mask


def dropout_configs(gt_labels, features, gap_seconds=DEFAULT_GAP_SECONDS,
                    seed=DEFAULT_SEED, fps=FPS):
    """Yield (name, hide_mask) including natural missing and synthetic gaps."""
    yield "natural", natural_missing_mask(features)
    for sec in gap_seconds:
        frames = int(round(float(sec) * fps))
        name = "gap_{}s".format(sec)
        yield name, synthetic_gap_mask(gt_labels, frames, seed=seed)
