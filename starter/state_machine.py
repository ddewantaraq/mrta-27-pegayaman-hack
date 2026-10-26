"""T21: possession state machine v1 over T20 features.

Rules encoded (see CHALLENGE.md / team TECH_DOC):
  - Ungraded frames -> X
  - Ball missing != dead: hold last H/A (or X if never set)
  - Switch team only after N consecutive opposing nearest + distance threshold (control)
  - Basic dead-ball: ball visible outside pitch margin (normalized coords)

    Defaults aim for fewer public-window possessions than baseline_raw, without Smart Tagging.
"""

from __future__ import print_function

import numpy as np

from games import GAME, half_video_blocks

HOME, AWAY, NOBODY = 0, 1, -1
LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"

# Defaults (logged in RESULTS when predict_v1 runs)
DEFAULT_CONTROL_FRAMES = 15          # 0.6 s at 25 fps
DEFAULT_CONTROL_DIST_M = 3.0         # metres
DEFAULT_OUT_MARGIN = 0.02            # normalized pitch: outside [m, 1-m] => dead candidate
DEFAULT_MIN_SPELL_FRAMES = 8         # merge shorter H/A islands into neighbors


def graded_mask(n_frames, game=GAME):
    """True on frames that fall inside a published half video block."""
    mask = np.zeros(n_frames, dtype=bool)
    for _half, first, last in half_video_blocks(game):
        lo = max(0, first)
        hi = min(n_frames - 1, last)
        if lo <= hi:
            mask[lo:hi + 1] = True
    return mask


def _ball_out_of_pitch(x_norm, y_norm, margin):
    if np.isnan(x_norm) or np.isnan(y_norm):
        return False
    return (x_norm < margin or x_norm > 1.0 - margin
            or y_norm < margin or y_norm > 1.0 - margin)


def _team_letter(code):
    if code == HOME:
        return LABEL_H
    if code == AWAY:
        return LABEL_A
    return None


def label_possession(
    features,
    game=GAME,
    control_frames=DEFAULT_CONTROL_FRAMES,
    control_dist_m=DEFAULT_CONTROL_DIST_M,
    out_margin=DEFAULT_OUT_MARGIN,
    min_spell_frames=DEFAULT_MIN_SPELL_FRAMES,
):
    """Return length-n array of 'H'/'A'/'D'/'X'."""
    n = int(features["n_frames"])
    graded = graded_mask(n, game)
    ball_present = features["ball_present"]
    nearest_team = features["nearest_team"]
    nearest_dist = features["nearest_dist"]
    bx = features["ball_x_norm"]
    by = features["ball_y_norm"]

    labels = np.full(n, LABEL_X, dtype="<U1")
    last_live = None  # HOME / AWAY
    pending_team = NOBODY
    pending_count = 0

    for i in range(n):
        if not graded[i]:
            labels[i] = LABEL_X
            pending_team = NOBODY
            pending_count = 0
            continue

        # Dead-ball: visible ball outside pitch band
        if ball_present[i] and _ball_out_of_pitch(bx[i], by[i], out_margin):
            labels[i] = LABEL_D
            pending_team = NOBODY
            pending_count = 0
            # last_live kept so restart can resume same side if needed; clear so restart needs control
            last_live = None
            continue

        if not ball_present[i]:
            # Missing != dead: hold last live team
            if last_live is None:
                labels[i] = LABEL_X
            else:
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        # Ball present, on pitch
        cand = int(nearest_team[i])
        dist = nearest_dist[i]
        close_enough = (not np.isnan(dist)) and dist <= control_dist_m

        if cand == NOBODY or not close_enough:
            # Contested / far: keep last live if any, else X
            if last_live is None:
                labels[i] = LABEL_X
            else:
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        if last_live is None:
            # Establish first possession after dead / start
            if cand == pending_team:
                pending_count += 1
            else:
                pending_team = cand
                pending_count = 1
            if pending_count >= control_frames:
                last_live = cand
                labels[i] = _team_letter(last_live)
                pending_team = NOBODY
                pending_count = 0
            else:
                labels[i] = LABEL_X
            continue

        if cand == last_live:
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        # Opposing nearest: require sustained control to switch
        if cand == pending_team:
            pending_count += 1
        else:
            pending_team = cand
            pending_count = 1

        if pending_count >= control_frames:
            last_live = cand
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
        else:
            # Contested: keep previous team (CHALLENGE.md)
            labels[i] = _team_letter(last_live)

    if min_spell_frames > 1:
        labels = _merge_short_spells(labels, graded, min_spell_frames)

    return labels


def _merge_short_spells(labels, graded, min_frames):
    """Merge short H/A runs into neighboring live labels to cut flicker."""
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if not graded[i] or out[i] not in (LABEL_H, LABEL_A):
            i += 1
            continue
        j = i + 1
        while j < n and graded[j] and out[j] == out[i]:
            j += 1
        length = j - i
        if length < min_frames:
            # Prefer previous live label on graded frames; else next; else leave
            prev = out[i - 1] if i > 0 and graded[i - 1] and out[i - 1] in (LABEL_H, LABEL_A, LABEL_D) else None
            nxt = out[j] if j < n and graded[j] and out[j] in (LABEL_H, LABEL_A, LABEL_D) else None
            fill = None
            if prev in (LABEL_H, LABEL_A):
                fill = prev
            elif nxt in (LABEL_H, LABEL_A):
                fill = nxt
            elif prev == LABEL_D and nxt in (LABEL_H, LABEL_A):
                fill = nxt
            if fill is not None:
                out[i:j] = fill
        i = j
    return out


def params_dict(
    control_frames=DEFAULT_CONTROL_FRAMES,
    control_dist_m=DEFAULT_CONTROL_DIST_M,
    out_margin=DEFAULT_OUT_MARGIN,
    min_spell_frames=DEFAULT_MIN_SPELL_FRAMES,
):
    return {
        "control_frames": control_frames,
        "control_dist_m": control_dist_m,
        "out_margin": out_margin,
        "min_spell_frames": min_spell_frames,
    }
