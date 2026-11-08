"""T21: possession state machine over T20 features.

Default path = compliant sm_v1_6 (sm_v1_5 + half-time clear + opposite 2nd kickoff).
sm_v1_5 = sm_v1_3 + missing-ball H/A (backdate + gap-bridge mid; Smart Tagging off).
sm_v1_4 = sm_v1_3 + missing-aware soft SET PIECES -> D (archived; --smart-tagging).
sm_v1_3 = sm_v1_2 + contested majority-near-ball override of nearest.
sm_v1_2 = sm_v1_1 + establish-without-X + min_spell 18.
Restored sm_v1: control=15, min_spell=8, edge knobs 0, keep_last_live_on_dead=False.

Rules encoded (see CHALLENGE.md / team TECH_DOC):
  - Ungraded frames -> X; with half-kickoff-reset, also clear last_live (halftime)
  - Ball missing != dead: hold last H/A (or X if never set), unless soft set-piece prior
  - Live ball with close nearest while establishing -> H/A (not X); lock after control_frames
  - Contested duel: if majority in control circle disagrees with nearest, trust majority
  - Switch team only after N consecutive opposing cand + distance threshold (control)
  - Optional backdate of pending frames once a switch confirms (Link & Hoernig)
  - Optional gap-bridge: when team after a live missing gap differs from before, place
    the switch inside the gap (start/mid/end) rather than after it
  - Half 2 graded before published 2nd kickoff -> D; seed opposite of 1st kickoff team
  - Dead-ball: out of pitch; optional near-edge missing / still-ball; soft SET PIECES prior
  - On D, optionally freeze last_live (sm_v1_1+) or clear it (sm_v1)
  - Short H/A merge: touch without control != turnover
"""

from __future__ import print_function

import numpy as np

from games import GAME, half_video_blocks
from load_smart_tagging import KICKOFF_VIDEO_FRAMES

HOME, AWAY, NOBODY = 0, 1, -1
LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"

# Compliant sm_v1_6 defaults — challenge/physics + external-game selection; not GT-tuned
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
DEFAULT_USE_SMART_TAGGING = False      # sm_v1_5+ = sm_v1_3 base; ST archived via --smart-tagging
DEFAULT_BACKDATE_SWITCH = True         # external winner: backdate pending on confirm
DEFAULT_GAP_BRIDGE = "mid"             # external winner: switch at mid of live missing gap
DEFAULT_LONG_GAP_DEAD_FRAMES = 0       # C dropped (hurt hidden-frame team acc)
DEFAULT_HALF_KICKOFF_RESET = True      # clear at HT; D pre-2nd KO; seed opposite team


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


def _letter_to_code(letter):
    if letter == LABEL_H:
        return HOME
    if letter == LABEL_A:
        return AWAY
    return NOBODY


def _opposite_team(code):
    if code == HOME:
        return AWAY
    if code == AWAY:
        return HOME
    return NOBODY


def _half_masks(n_frames, game=GAME):
    """Bool masks for half 1 and half 2 video blocks (False outside)."""
    half1 = np.zeros(n_frames, dtype=bool)
    half2 = np.zeros(n_frames, dtype=bool)
    for half, first, last in half_video_blocks(game):
        lo = max(0, first)
        hi = min(n_frames - 1, last)
        if lo > hi:
            continue
        if half == 1:
            half1[lo:hi + 1] = True
        elif half == 2:
            half2[lo:hi + 1] = True
    return half1, half2


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
    set_piece_dead=None,
    graded=None,
    backdate_switch=DEFAULT_BACKDATE_SWITCH,
    gap_bridge=DEFAULT_GAP_BRIDGE,
    long_gap_dead_frames=DEFAULT_LONG_GAP_DEAD_FRAMES,
    half_kickoff_reset=DEFAULT_HALF_KICKOFF_RESET,
    kickoff_frames=None,
):
    """Return length-n array of 'H'/'A'/'D'/'X'.

    sm_v1: out-of-pitch dead only; clear last_live on D.
    sm_v1_1+: edge missing/still dead; freeze last_live on D.
    sm_v1_3: contested majority-near-ball can override nearest as cand.
    sm_v1_4: optional set_piece_dead mask forces D (soft Smart Tagging prior).
    sm_v1_5: optional backdate_switch / gap_bridge / long_gap_dead (external-chosen).
    sm_v1_6: half_kickoff_reset — clear last_live on ungraded; D before 2nd kickoff;
             seed opposite of first-half kickoff team at published 2nd kickoff frame.

    graded: optional bool mask; default = BAR-ATH half video blocks. External
    games pass an all-True (or period) mask so graded_mask(game) is not used.
    kickoff_frames: dict with '1st Half' / '2nd Half' video frames; default
    KICKOFF_VIDEO_FRAMES when half_kickoff_reset is on; pass explicitly None with
    half_kickoff_reset False for sm_v1_5-compatible behaviour.
    """
    n = int(features["n_frames"])
    if graded is None:
        graded = graded_mask(n, game)
    else:
        graded = np.asarray(graded, dtype=bool)
        if len(graded) != n:
            raise ValueError("graded length {} != n_frames {}".format(len(graded), n))
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
    sp_dead = None
    if set_piece_dead is not None:
        sp_dead = np.asarray(set_piece_dead, dtype=bool)
        if len(sp_dead) != n:
            raise ValueError(
                "set_piece_dead length {} != n_frames {}".format(len(sp_dead), n))
    gap_mode = (gap_bridge or "off").lower()
    if gap_mode not in ("off", "start", "mid", "end"):
        raise ValueError("gap_bridge must be off|start|mid|end, got {!r}".format(gap_bridge))

    use_kickoff = bool(half_kickoff_reset)
    if use_kickoff:
        if kickoff_frames is None:
            kickoff_frames = KICKOFF_VIDEO_FRAMES
        try:
            kickoff_2nd = int(kickoff_frames["2nd Half"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "kickoff_frames must map '2nd Half' to an int frame, got {!r}".format(
                    kickoff_frames))
        half1_mask, half2_mask = _half_masks(n, game)
    else:
        kickoff_2nd = None
        half1_mask = half2_mask = None

    labels = np.full(n, LABEL_X, dtype="<U1")
    last_live = None  # HOME / AWAY
    pending_team = NOBODY
    pending_count = 0
    pending_start = -1
    kickoff_team_1 = None  # first locked last_live in half 1

    last_ball_x = np.nan
    last_ball_y = np.nan
    missing_streak = 0
    still_edge_streak = 0
    use_edge = (missing_near_edge_frames > 0) or (still_near_edge_frames > 0)
    use_long_gap = int(long_gap_dead_frames) > 0

    for i in range(n):
        if not graded[i]:
            labels[i] = LABEL_X
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
            missing_streak = 0
            still_edge_streak = 0
            if use_kickoff:
                last_live = None
            continue

        # Half 2 graded before published 2nd kickoff: play not restarted -> D
        if (use_kickoff and half2_mask is not None and half2_mask[i]
                and i < kickoff_2nd):
            labels[i] = LABEL_D
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
            last_live = None
            missing_streak = 0
            still_edge_streak = 0
            continue

        # Seed opposite of first-half kickoff team at 2nd kickoff frame
        if (use_kickoff and i == kickoff_2nd and kickoff_team_1 is not None
                and last_live is None):
            last_live = _opposite_team(kickoff_team_1)
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
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
        elif use_edge or use_long_gap:
            still_edge_streak = 0
            if not ball_present[i]:
                missing_streak += 1
            else:
                missing_streak = 0
        elif not ball_present[i]:
            missing_streak += 1
        else:
            missing_streak = 0

        dead = False
        if sp_dead is not None and sp_dead[i]:
            dead = True
        elif ball_present[i] and _ball_out_of_pitch(bx[i], by[i], out_margin):
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
        elif (use_long_gap and not ball_present[i]
              and missing_streak >= int(long_gap_dead_frames)):
            dead = True

        if dead:
            labels[i] = LABEL_D
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
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
            pending_start = -1
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
            pending_start = -1
            continue

        if last_live is None:
            # Establish: live ball + close nearest -> H/A (not X); lock after control_frames
            if cand == pending_team:
                pending_count += 1
            else:
                pending_team = cand
                pending_count = 1
                pending_start = i
            if pending_count >= control_frames:
                last_live = cand
                labels[i] = _team_letter(last_live)
                if (use_kickoff and kickoff_team_1 is None
                        and half1_mask is not None and half1_mask[i]):
                    kickoff_team_1 = last_live
                pending_team = NOBODY
                pending_count = 0
                pending_start = -1
            else:
                labels[i] = _team_letter(cand)
            continue

        if cand == last_live:
            labels[i] = _team_letter(last_live)
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
            continue

        if cand == pending_team:
            pending_count += 1
        else:
            pending_team = cand
            pending_count = 1
            pending_start = i

        if pending_count >= control_frames:
            last_live = cand
            letter = _team_letter(last_live)
            labels[i] = letter
            if (use_kickoff and kickoff_team_1 is None
                    and half1_mask is not None and half1_mask[i]):
                kickoff_team_1 = last_live
            if backdate_switch and pending_start >= 0:
                # Relabel the pending run to the new team (switch at first control).
                lo = pending_start
                hi = i + 1
                for k in range(lo, hi):
                    if graded[k] and labels[k] in (LABEL_H, LABEL_A):
                        labels[k] = letter
            pending_team = NOBODY
            pending_count = 0
            pending_start = -1
        else:
            labels[i] = _team_letter(last_live)

    if gap_mode != "off":
        labels = _apply_gap_bridge(labels, ball_present, graded, gap_mode)

    if min_spell_frames > 1:
        labels = _merge_short_spells(labels, graded, min_spell_frames)

    return labels


def _apply_gap_bridge(labels, ball_present, graded, mode):
    """If team after a live missing gap differs from before, place switch in the gap."""
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if not graded[i] or ball_present[i] or out[i] == LABEL_D:
            i += 1
            continue
        # Start of a missing (non-D) run
        g0 = i
        while (i < n and graded[i] and (not ball_present[i])
               and out[i] != LABEL_D):
            i += 1
        g1 = i  # exclusive
        if g1 <= g0:
            continue
        # Team before gap
        before = None
        k = g0 - 1
        while k >= 0 and graded[k]:
            if out[k] in (LABEL_H, LABEL_A):
                before = out[k]
                break
            if out[k] == LABEL_D:
                break
            k -= 1
        # Confirmed team after gap (first H/A after g1)
        after = None
        k = g1
        while k < n and graded[k]:
            if out[k] in (LABEL_H, LABEL_A):
                after = out[k]
                break
            if out[k] == LABEL_D:
                break
            k += 1
        if before is None or after is None or before == after:
            continue
        # Place switch inside the gap
        length = g1 - g0
        if mode == "start":
            cut = g0
        elif mode == "end":
            cut = g1  # all gap stays `before`; switch at reappear (no-op on gap)
        else:  # mid
            cut = g0 + length // 2
        if cut > g0:
            out[g0:cut] = before
        if cut < g1:
            out[cut:g1] = after
    return out


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
    use_smart_tagging=DEFAULT_USE_SMART_TAGGING,
    set_piece_dead_cap_frames=50,
    backdate_switch=DEFAULT_BACKDATE_SWITCH,
    gap_bridge=DEFAULT_GAP_BRIDGE,
    long_gap_dead_frames=DEFAULT_LONG_GAP_DEAD_FRAMES,
    half_kickoff_reset=DEFAULT_HALF_KICKOFF_RESET,
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
        "use_smart_tagging": use_smart_tagging,
        "set_piece_dead_cap_frames": set_piece_dead_cap_frames,
        "backdate_switch": backdate_switch,
        "gap_bridge": gap_bridge,
        "long_gap_dead_frames": long_gap_dead_frames,
        "half_kickoff_reset": half_kickoff_reset,
    }
