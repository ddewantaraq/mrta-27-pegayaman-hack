"""T21: possession state machine over T20 features.

Default path = compliant sm_v1_3 (rule/physics citations; no public-GT parameter tuning).
sm_v1_3 = sm_v1_2 + contested majority-near-ball override of nearest.
sm_v1_2 = sm_v1_1 + establish-without-X + min_spell 18.
Restored sm_v1: control=15, min_spell=8, edge knobs 0, keep_last_live_on_dead=False.

Rules encoded (see CHALLENGE.md / team TECH_DOC):
  - Ungraded frames -> X
  - Ball missing != dead: hold last H/A (or X if never set)
  - Live ball with close nearest while establishing -> H/A (not X); lock after control_frames
  - Contested duel: if majority in control circle disagrees with nearest, trust majority
  - Switch team only after N consecutive opposing cand + distance threshold (control)
  - Dead-ball: out of pitch; optional near-edge missing / still-ball heuristics
  - On D, optionally freeze last_live (sm_v1_1+) or clear it (sm_v1)
  - Short H/A merge: touch without control != turnover
"""

from __future__ import print_function

import numpy as np

from games import GAME, half_video_blocks

HOME, AWAY, NOBODY = 0, 1, -1
LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"

# Compliant sm_v1_3 defaults — challenge/physics, not GT-tuned
DEFAULT_CONTROL_FRAMES = 25          # ~1.0 s: sustained control before switch
DEFAULT_CONTROL_DIST_M = 3.0         # metres
DEFAULT_OUT_MARGIN = 0.02            # normalized pitch: outside [m, 1-m] => dead candidate
DEFAULT_MIN_SPELL_FRAMES = 18        # ~0.72 s: touch without control != turnover
DEFAULT_MISSING_NEAR_EDGE_FRAMES = 50  # 2 s missing after near-edge sighting -> D
DEFAULT_STILL_NEAR_EDGE_FRAMES = 25    # 1 s slow ball near edge -> D
DEFAULT_STILL_SPEED_M = 0.5
DEFAULT_KEEP_LAST_LIVE_ON_DEAD = True  # freeze last team through stoppage / contested hold
DEFAULT_USE_MAJORITY_NEAR = True       # contested override of nearest by cluster majority
DEFAULT_CONTEST_DIST_M = 1.0           # |d_home - d_away| <= this => contested


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
    if x_norm is None or y_norm is None or np.isnan(x_norm) or np.isnan(y_norm):
        return False
    return (x_norm < margin or x_norm > 1.0 - margin
            or y_norm < margin or y_norm > 1.0 - margin)


def _ball_near_edge(x_norm, y_norm, margin):
    """Near or outside the out-of-pitch band (wider than out_margin for missing heuristic)."""
    band = margin * 3.0
    return _ball_out_of_pitch(x_norm, y_norm, band)


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
    missing_near_edge_frames=DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    still_near_edge_frames=DEFAULT_STILL_NEAR_EDGE_FRAMES,
    still_speed_m=DEFAULT_STILL_SPEED_M,
    keep_last_live_on_dead=DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
    use_majority_near=DEFAULT_USE_MAJORITY_NEAR,
    contest_dist_m=DEFAULT_CONTEST_DIST_M,
):
    """Return length-n array of 'H'/'A'/'D'/'X'.

    sm_v1: out-of-pitch dead only; clear last_live on D.
    sm_v1_1+: edge missing/still dead; freeze last_live on D.
    sm_v1_3: contested majority-near-ball can override nearest as cand.
    """
    n = int(features["n_frames"])
    graded = graded_mask(n, game)
    ball_present = features["ball_present"]
    nearest_team = features["nearest_team"]
    nearest_dist = features["nearest_dist"]
    min_dist_home = features.get("min_dist_home")
    min_dist_away = features.get("min_dist_away")
    majority_team = features.get("majority_team")
    bx = features["ball_x_norm"]
    by = features["ball_y_norm"]
    bspeed = features.get("ball_speed")
    can_majority = (
        use_majority_near
        and majority_team is not None
        and min_dist_home is not None
        and min_dist_away is not None
    )

    labels = np.full(n, LABEL_X, dtype="<U1")
    last_live = None  # HOME / AWAY
    pending_team = NOBODY
    pending_count = 0

    last_ball_x = np.nan
    last_ball_y = np.nan
    missing_streak = 0
    still_edge_streak = 0
    use_edge = (missing_near_edge_frames > 0) or (still_near_edge_frames > 0)

    for i in range(n):
        if not graded[i]:
            labels[i] = LABEL_X
            pending_team = NOBODY
            pending_count = 0
            missing_streak = 0
            still_edge_streak = 0
            continue

        if use_edge and ball_present[i]:
            last_ball_x = bx[i]
            last_ball_y = by[i]
            missing_streak = 0
            if bspeed is not None and still_near_edge_frames > 0:
                speed = bspeed[i]
                near = _ball_near_edge(bx[i], by[i], out_margin)
                slow = (not np.isnan(speed)) and speed <= still_speed_m
                if near and slow:
                    still_edge_streak += 1
                else:
                    still_edge_streak = 0
            else:
                still_edge_streak = 0
        elif use_edge:
            still_edge_streak = 0
            missing_streak += 1

        dead = False
        if ball_present[i] and _ball_out_of_pitch(bx[i], by[i], out_margin):
            dead = True
        elif (use_edge and ball_present[i]
              and still_near_edge_frames > 0
              and still_edge_streak >= still_near_edge_frames):
            dead = True
        elif (use_edge and not ball_present[i]
              and missing_near_edge_frames > 0
              and missing_streak >= missing_near_edge_frames
              and _ball_near_edge(last_ball_x, last_ball_y, out_margin)):
            dead = True

        if dead:
            labels[i] = LABEL_D
            pending_team = NOBODY
            pending_count = 0
            if not keep_last_live_on_dead:
                last_live = None
            continue

        if not ball_present[i]:
            if last_live is None:
                labels[i] = LABEL_X
            else:
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        cand = int(nearest_team[i])
        dist = nearest_dist[i]
        if can_majority:
            maj = int(majority_team[i])
            if maj != NOBODY and maj != cand:
                dh = min_dist_home[i]
                da = min_dist_away[i]
                if (not np.isnan(dh) and not np.isnan(da)
                        and abs(float(dh) - float(da)) <= contest_dist_m):
                    cand = maj
                    # Control distance still judged vs the nearest player of cand's team
                    dist = dh if cand == HOME else da
        close_enough = (not np.isnan(dist)) and dist <= control_dist_m

        if cand == NOBODY or not close_enough:
            if last_live is None:
                labels[i] = LABEL_X
            else:
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        if last_live is None:
            # Establish: live ball + close nearest -> H/A (not X); lock after control_frames
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
                labels[i] = _team_letter(cand)
            continue

        if cand == last_live:
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

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
    missing_near_edge_frames=DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    still_near_edge_frames=DEFAULT_STILL_NEAR_EDGE_FRAMES,
    still_speed_m=DEFAULT_STILL_SPEED_M,
    keep_last_live_on_dead=DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
    use_majority_near=DEFAULT_USE_MAJORITY_NEAR,
    contest_dist_m=DEFAULT_CONTEST_DIST_M,
):
    return {
        "control_frames": control_frames,
        "control_dist_m": control_dist_m,
        "out_margin": out_margin,
        "min_spell_frames": min_spell_frames,
        "missing_near_edge_frames": missing_near_edge_frames,
        "still_near_edge_frames": still_near_edge_frames,
        "still_speed_m": still_speed_m,
        "keep_last_live_on_dead": keep_last_live_on_dead,
        "use_majority_near": use_majority_near,
        "contest_dist_m": contest_dist_m,
    }
