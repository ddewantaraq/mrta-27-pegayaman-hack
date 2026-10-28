"""T21/T23: possession state machine over T20 features (+ optional Smart Tagging priors).

Rules encoded (see CHALLENGE.md / team TECH_DOC):
  - Ungraded frames -> X
  - Ball missing != dead by default: hold last H/A (or X if never set)
  - Switch team only after N consecutive opposing nearest + distance threshold (control)
  - Dead-ball v1.1: out of pitch; long missing near edge; still ball near edge
  - On D, keep last_live frozen (reduces graded X after restarts)
  - Soft priors: capped SET PIECES -> D; team-tagged phases = weak vote only (never copy spans)
  - T24: sandwich-only short-spell merge; shorter opposing switch soon after leaving D
  - Short D blips (< min_dead_frames) sandwiched by same team -> fill as that team
  - After a long D (>= restart_reset_dead_frames), clear last_live and re-establish
"""

from __future__ import print_function

import numpy as np

from games import GAME, half_video_blocks

HOME, AWAY, NOBODY = 0, 1, -1
LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"

# Defaults: soft set-piece priors + v1.1 hysteresis (37/25 spell-regressed on public window)
DEFAULT_CONTROL_FRAMES = 25          # 1.0 s at 25 fps
DEFAULT_CONTROL_DIST_M = 3.0         # metres
DEFAULT_OUT_MARGIN = 0.02            # normalized pitch: outside [m, 1-m] => dead candidate
DEFAULT_MIN_SPELL_FRAMES = 12        # merge shorter H/A islands (~0.5 s)
DEFAULT_MISSING_NEAR_EDGE_FRAMES = 50  # 2 s missing after last ball near edge -> D
DEFAULT_STILL_NEAR_EDGE_FRAMES = 25    # 1 s slow ball near edge -> D
DEFAULT_STILL_SPEED_M = 0.5            # m/s
DEFAULT_RESTART_CONTROL_FRAMES = 12  # used when restart_window_frames > 0
DEFAULT_RESTART_WINDOW_FRAMES = 0    # off by default (T24: 50 + restart=12 spell-regressed)
DEFAULT_MIN_DEAD_FRAMES = 50         # fill D islands shorter than this when same team on both sides
DEFAULT_RESTART_RESET_DEAD_FRAMES = 0  # off until sm_v1_10 keep decision; 50 enables post-stoppage reset
DEFAULT_RESTART_ESTABLISH_FRAMES = 12  # frames to establish after a long-D reset


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
    restart_control_frames=DEFAULT_RESTART_CONTROL_FRAMES,
    restart_window_frames=DEFAULT_RESTART_WINDOW_FRAMES,
    min_dead_frames=DEFAULT_MIN_DEAD_FRAMES,
    restart_reset_dead_frames=DEFAULT_RESTART_RESET_DEAD_FRAMES,
    restart_establish_frames=DEFAULT_RESTART_ESTABLISH_FRAMES,
    smart_priors=None,
):
    """Return length-n array of 'H'/'A'/'D'/'X'.

    smart_priors: optional dict from smart_priors.build_smart_priors with
    set_piece_dead (bool) and phase_team (int8 HOME/AWAY/NOBODY).

    After leaving dead ball, opposing switches use restart_control_frames for
    restart_window_frames (faster restarts); mid-play switches keep control_frames.

    After a dead stretch of restart_reset_dead_frames or more, last_live is cleared
    and possession is re-established over restart_establish_frames.
    """
    n = int(features["n_frames"])
    graded = graded_mask(n, game)
    ball_present = features["ball_present"]
    nearest_team = features["nearest_team"]
    nearest_dist = features["nearest_dist"]
    bx = features["ball_x_norm"]
    by = features["ball_y_norm"]
    bspeed = features["ball_speed"]

    set_piece_dead = None
    phase_team = None
    if smart_priors is not None:
        set_piece_dead = smart_priors["set_piece_dead"]
        phase_team = smart_priors["phase_team"]

    labels = np.full(n, LABEL_X, dtype="<U1")
    last_live = None  # HOME / AWAY — kept across short D
    pending_team = NOBODY
    pending_count = 0
    prev_set_piece = False
    frames_since_dead = restart_window_frames + 1  # mid-play until first D exit
    was_dead = False
    dead_stretch = 0
    post_reset_establish = False

    last_ball_x = np.nan
    last_ball_y = np.nan
    missing_streak = 0
    still_edge_streak = 0

    for i in range(n):
        if not graded[i]:
            labels[i] = LABEL_X
            pending_team = NOBODY
            pending_count = 0
            missing_streak = 0
            still_edge_streak = 0
            prev_set_piece = False
            was_dead = False
            dead_stretch = 0
            post_reset_establish = False
            frames_since_dead = restart_window_frames + 1
            continue

        in_set_piece = bool(set_piece_dead is not None and set_piece_dead[i])
        soft_team = int(phase_team[i]) if phase_team is not None else NOBODY

        # Leaving a set-piece window: restart needs fresh control
        if prev_set_piece and not in_set_piece:
            pending_team = NOBODY
            pending_count = 0
        prev_set_piece = in_set_piece

        # Update last-known ball position when visible
        if ball_present[i]:
            last_ball_x = bx[i]
            last_ball_y = by[i]
            missing_streak = 0
            speed = bspeed[i]
            near = _ball_near_edge(bx[i], by[i], out_margin)
            slow = (not np.isnan(speed)) and speed <= still_speed_m
            if near and slow:
                still_edge_streak += 1
            else:
                still_edge_streak = 0
        else:
            still_edge_streak = 0
            missing_streak += 1

        dead = in_set_piece
        if ball_present[i] and _ball_out_of_pitch(bx[i], by[i], out_margin):
            dead = True
        elif (ball_present[i]
              and still_edge_streak >= still_near_edge_frames):
            dead = True
        elif (not ball_present[i]
              and missing_streak >= missing_near_edge_frames
              and _ball_near_edge(last_ball_x, last_ball_y, out_margin)):
            dead = True

        if dead:
            labels[i] = LABEL_D
            pending_team = NOBODY
            pending_count = 0
            was_dead = True
            frames_since_dead = 0
            dead_stretch += 1
            # keep last_live frozen through short dead-ball / set pieces
            continue

        if was_dead:
            # first live frame after a dead stretch
            pending_team = NOBODY
            pending_count = 0
            frames_since_dead = 0
            if (restart_reset_dead_frames > 0
                    and dead_stretch >= restart_reset_dead_frames):
                last_live = None
                post_reset_establish = True
            else:
                post_reset_establish = False
            dead_stretch = 0
            was_dead = False
        else:
            frames_since_dead += 1

        if not ball_present[i]:
            if last_live is None:
                # Weak prior only when no live team yet
                if soft_team != NOBODY:
                    labels[i] = _team_letter(soft_team)
                else:
                    labels[i] = LABEL_X
            else:
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        # Ball present, on pitch (not dead)
        cand = int(nearest_team[i])
        dist = nearest_dist[i]
        close_enough = (not np.isnan(dist)) and dist <= control_dist_m

        if cand == NOBODY or not close_enough:
            if last_live is None:
                if soft_team != NOBODY:
                    labels[i] = _team_letter(soft_team)
                else:
                    labels[i] = LABEL_X
            else:
                # Never let phase_team overwrite an established opposing possession
                labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        in_restart = frames_since_dead < restart_window_frames
        switch_need = restart_control_frames if in_restart else control_frames

        if last_live is None:
            # Establish first possession / post-stoppage reset
            need = restart_establish_frames if post_reset_establish else switch_need
            if soft_team == cand:
                need = max(1, need // 2)
            if cand == pending_team:
                pending_count += 1
            else:
                pending_team = cand
                pending_count = 1
            if pending_count >= need:
                last_live = cand
                labels[i] = _team_letter(last_live)
                pending_team = NOBODY
                pending_count = 0
                post_reset_establish = False
            else:
                # Prefer nearest / soft vote over X (cold-start + post-reset)
                if soft_team != NOBODY:
                    labels[i] = _team_letter(soft_team)
                else:
                    labels[i] = _team_letter(cand)
            continue

        if cand == last_live:
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            continue

        # Opposing nearest: require sustained control to switch (phase cannot force it)
        if cand == pending_team:
            pending_count += 1
        else:
            pending_team = cand
            pending_count = 1

        if pending_count >= switch_need:
            last_live = cand
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
        else:
            labels[i] = _team_letter(last_live)

    if min_spell_frames > 1:
        labels = _merge_short_spells(labels, graded, min_spell_frames)
    if min_dead_frames > 0:
        labels = _fill_short_dead(labels, graded, min_dead_frames)

    return labels


def _merge_short_spells(labels, graded, min_frames):
    """Absorb short opposing H/A flicker only when sandwiched by the same live team.

    Does not fill from prev alone, and does not wipe a short post-dead spell via nxt.
    """
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
            prev = out[i - 1] if i > 0 and graded[i - 1] else None
            nxt = out[j] if j < n and graded[j] else None
            if (prev in (LABEL_H, LABEL_A)
                    and nxt in (LABEL_H, LABEL_A)
                    and prev == nxt
                    and prev != out[i]):
                out[i:j] = prev
        i = j
    return out


def _fill_short_dead(labels, graded, min_frames):
    """Fill short D islands sandwiched by the same live team (tracking noise, not stoppages).

    Ungraded / X neighbours block the fill. Real restarts are typically >> 2 s.
    """
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if not graded[i] or out[i] != LABEL_D:
            i += 1
            continue
        j = i + 1
        while j < n and graded[j] and out[j] == LABEL_D:
            j += 1
        length = j - i
        if length < min_frames:
            prev = out[i - 1] if i > 0 and graded[i - 1] else None
            nxt = out[j] if j < n and graded[j] else None
            if (prev in (LABEL_H, LABEL_A)
                    and nxt in (LABEL_H, LABEL_A)
                    and prev == nxt):
                out[i:j] = prev
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
    restart_control_frames=DEFAULT_RESTART_CONTROL_FRAMES,
    restart_window_frames=DEFAULT_RESTART_WINDOW_FRAMES,
    min_dead_frames=DEFAULT_MIN_DEAD_FRAMES,
    restart_reset_dead_frames=DEFAULT_RESTART_RESET_DEAD_FRAMES,
    restart_establish_frames=DEFAULT_RESTART_ESTABLISH_FRAMES,
):
    return {
        "control_frames": control_frames,
        "control_dist_m": control_dist_m,
        "out_margin": out_margin,
        "min_spell_frames": min_spell_frames,
        "missing_near_edge_frames": missing_near_edge_frames,
        "still_near_edge_frames": still_near_edge_frames,
        "still_speed_m": still_speed_m,
        "restart_control_frames": restart_control_frames,
        "restart_window_frames": restart_window_frames,
        "min_dead_frames": min_dead_frames,
        "restart_reset_dead_frames": restart_reset_dead_frames,
        "restart_establish_frames": restart_establish_frames,
    }
