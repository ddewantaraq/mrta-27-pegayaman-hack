"""Vidal-Codina et al. 2022 possession step, mapped onto team labels.

Paper: Ferran Vidal-Codina, Nicolas Evans, Benjamin El Fakir, Jonathan Billingham,
"Automatic event detection in football using tracking data", Sports Engineering, 2022.
https://arxiv.org/abs/2202.00804  (section 2.3 only).

Per frame:
  - Possession zone: ball within R_pz of a player -> that player (closest if several
    teammates).
  - Duel zone: if both teams have a player within R_dz, the frame is a duel.
    The paper says the duel prevails, so nobody is given possession.
  - A possession gain on a same-player run requires a ball-path change: direction
    change from the start of the run to the end, or a speed change on any frame.
    A straight fly-by is not a gain (touch / deflection != control).

Team label: from a validated gain, that team keeps the ball through loose and
void frames until the other team records a gain. Contested frames stay with the
last team because a duel never starts a gain.

Direction/speed thresholds are the published databallpy defaults of this method
(10 degrees, 5 m/s). The paper's numeric epsilons live in its supplement, which
is not used here. Radii are the paper's published values (0.5 m / 1.0 m, duel
zone 1.0 m) or databallpy's 1.5 m possession zone. They are not fit on BAR-ATH.

Noisy ball coordinates are smoothed with a Savitzky-Golay window of 7 frames,
polynomial order 2, on contiguous visible stretches, as the paper specifies.
"""

from __future__ import print_function

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1

# databallpy published defaults for the Vidal-Codina gain test
DEFAULT_BV_THRESHOLD_M_S = 5.0
DEFAULT_BA_THRESHOLD_DEG = 10.0
# Paper section 3: provider A uses 0.5 m, providers B and C use 1.0 m; duel zone 1.0 m.
DEFAULT_PZ_RADIUS_M = 1.0
DEFAULT_DZ_RADIUS_M = 1.0
DEFAULT_FPS = 25.0

# Savitzky-Golay smoothing coefficients, window 7, polynomial order 2.
_SG7 = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0], dtype=np.float64) / 21.0
_ANGLE_EPS_M = 1e-3


def zone_control(player_xy, teams, player_ids, ball_xy, pz_radius_m, dz_radius_m):
    """One frame: (player_id or -1, team or NOBODY, is_duel).

    Duel (both teams inside the duel zone) wins over a possession-zone hit.
    Otherwise the closest player inside the possession zone is the candidate.
    """
    bx, by = ball_xy
    pz2 = float(pz_radius_m) * float(pz_radius_m)
    dz2 = float(dz_radius_m) * float(dz_radius_m)
    best_d2 = None
    best_id = -1
    best_team = NOBODY
    home_dz = False
    away_dz = False
    for xy, team, pid in zip(player_xy, teams, player_ids):
        if xy is None:
            continue
        dx = xy[0] - bx
        dy = xy[1] - by
        d2 = dx * dx + dy * dy
        t = int(team)
        if d2 <= dz2:
            if t == HOME:
                home_dz = True
            elif t == AWAY:
                away_dz = True
        if d2 <= pz2 and (best_d2 is None or d2 < best_d2):
            best_d2 = d2
            best_id = int(pid)
            best_team = t
    if home_dz and away_dz:
        return -1, NOBODY, True
    if best_id < 0:
        return -1, NOBODY, False
    return best_id, best_team, False


def _smooth_visible(x, y, present):
    """SG-smooth contiguous visible stretches. Edges of each stretch stay raw."""
    xs = np.array(x, dtype=np.float64, copy=True)
    ys = np.array(y, dtype=np.float64, copy=True)
    present = np.asarray(present, dtype=bool)
    idx = np.flatnonzero(present)
    if len(idx) < 7:
        return xs, ys
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    for s, e in zip(starts, ends):
        frames = idx[s:e + 1]
        if len(frames) < 7:
            continue
        # Reflect-pad so the Savitzky-Golay window does not invent a kink at the edges.
        pad = 3
        px = np.pad(xs[frames], pad, mode="reflect")
        py = np.pad(ys[frames], pad, mode="reflect")
        cx = np.convolve(px, _SG7, mode="valid")
        cy = np.convolve(py, _SG7, mode="valid")
        # valid on a series padded by 3 on each side with a length-7 kernel
        # returns exactly len(frames) samples.
        xs[frames] = cx
        ys[frames] = cy
    return xs, ys


def _step_speed(x, y, present, fps):
    """Speed at frame i is the step from i-1 to i, metres per second. NaN if broken."""
    speed = np.full(len(x), np.nan, dtype=np.float64)
    if len(x) < 2:
        return speed
    both = np.asarray(present[:-1], dtype=bool) & np.asarray(present[1:], dtype=bool)
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx * dx + dy * dy) * float(fps)
    speed[1:][both] = step[both]
    return speed


def _angle_deg(v_in, v_out):
    n0 = float(np.hypot(v_in[0], v_in[1]))
    n1 = float(np.hypot(v_out[0], v_out[1]))
    if n0 < _ANGLE_EPS_M or n1 < _ANGLE_EPS_M:
        return 0.0
    cos = (v_in[0] * v_out[0] + v_in[1] * v_out[1]) / (n0 * n1)
    if cos > 1.0:
        cos = 1.0
    elif cos < -1.0:
        cos = -1.0
    return float(np.degrees(np.arccos(cos)))


def _is_gain(start, end, x, y, speed, present, bv_threshold, ba_threshold_deg):
    """Paper: direction change over the run, or speed change on any frame."""
    n = len(x)
    last = end if end + 1 < n else end - 1
    for i in range(start, last + 1):
        s0 = speed[i]
        s1 = speed[i + 1] if i + 1 < n else np.nan
        if s0 == s0 and s1 == s1 and abs(float(s0) - float(s1)) > bv_threshold:
            return True
    if start - 1 >= 0 and end + 1 < n and present[start - 1] and present[start] \
            and present[end] and present[end + 1]:
        v_in = (x[start] - x[start - 1], y[start] - y[start - 1])
        v_out = (x[end + 1] - x[end], y[end + 1] - y[end])
        if _angle_deg(v_in, v_out) > ba_threshold_deg:
            return True
    return False


def reappearance_angles(ball_x, ball_y, ball_present):
    """Angle in degrees at the first visible frame after each missing run.

    Compares the ball direction on the last visible frame with the direction on
    the first visible frame. NaN when either side has no step. Uses the same
    Savitzky-Golay smooth as the gain test. Hidden coordinates must already be
    removed by the caller; this does not look inside the gap.
    """
    present = np.asarray(ball_present, dtype=bool)
    xs, ys = _smooth_visible(ball_x, ball_y, present)
    n = len(xs)
    angles = np.full(n, np.nan, dtype=np.float64)
    i = 0
    prev_end = None
    while i < n:
        if not present[i]:
            i += 1
            continue
        j = i + 1
        while j < n and present[j]:
            j += 1
        if (prev_end is not None and i > prev_end + 1
                and prev_end - 1 >= 0 and present[prev_end - 1]
                and i + 1 < n and present[i + 1]):
            v_in = (xs[prev_end] - xs[prev_end - 1], ys[prev_end] - ys[prev_end - 1])
            v_out = (xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            angles[i] = _angle_deg(v_in, v_out)
        prev_end = j - 1
        i = j
    return angles


def team_possession_from_zones(
    pz_player,
    pz_team,
    ball_x,
    ball_y,
    ball_present,
    reset_mask=None,
    bv_threshold=DEFAULT_BV_THRESHOLD_M_S,
    ba_threshold_deg=DEFAULT_BA_THRESHOLD_DEG,
    fps=DEFAULT_FPS,
):
    """Length-n int8 team codes. NOBODY until a validated gain, then hold.

    reset_mask frames clear the running team (halftime / ungraded). Gains are
    not allowed to start on a reset frame, and a same-player run cannot cross one.
    """
    player = np.asarray(pz_player)
    team = np.asarray(pz_team)
    present = np.asarray(ball_present, dtype=bool)
    n = len(player)
    if len(team) != n or len(ball_x) != n or len(ball_y) != n or len(present) != n:
        raise ValueError("vc zone arrays must share n_frames")
    if reset_mask is None:
        reset = np.zeros(n, dtype=bool)
    else:
        reset = np.asarray(reset_mask, dtype=bool)
        if len(reset) != n:
            raise ValueError("reset_mask length {} != {}".format(len(reset), n))

    xs, ys = _smooth_visible(ball_x, ball_y, present)
    speed = _step_speed(xs, ys, present, fps)

    gain_frame = np.zeros(n, dtype=bool)
    gain_team = np.full(n, NOBODY, dtype=np.int8)
    i = 0
    while i < n:
        if reset[i] or (not present[i]) or int(player[i]) < 0:
            i += 1
            continue
        pid = int(player[i])
        j = i + 1
        while (j < n and not reset[j] and present[j] and int(player[j]) == pid):
            j += 1
        end = j - 1
        if _is_gain(i, end, xs, ys, speed, present, bv_threshold, ba_threshold_deg):
            gain_frame[i] = True
            gain_team[i] = int(team[i])
        i = j

    out = np.full(n, NOBODY, dtype=np.int8)
    current = NOBODY
    for i in range(n):
        if reset[i]:
            current = NOBODY
            continue
        if gain_frame[i]:
            current = int(gain_team[i])
        out[i] = current
    return out


def live_structure(labels, graded, fps=DEFAULT_FPS):
    """Non-GT structure on graded frames: spells, short islands, X-after-D."""
    labels = np.asarray(labels)
    graded = np.asarray(graded, dtype=bool)
    n = len(labels)
    n_spells = 0
    lengths = []
    short12 = 0
    short25 = 0
    i = 0
    while i < n:
        if (not graded[i]) or labels[i] not in ("H", "A"):
            i += 1
            continue
        j = i + 1
        team = labels[i]
        while j < n and graded[j] and labels[j] == team:
            j += 1
        length = j - i
        n_spells += 1
        lengths.append(length)
        if length < 12:
            short12 += 1
        if length < 25:
            short25 += 1
        i = j
    x_after_d = 0
    for i in range(1, n):
        if graded[i] and labels[i] == "X" and graded[i - 1] and labels[i - 1] == "D":
            x_after_d += 1
    on = labels[graded]
    counts = {lab: int(np.sum(on == lab)) for lab in ("H", "A", "D", "X")}
    mean_s = float(np.mean(lengths)) / float(fps) if lengths else 0.0
    return {
        "counts": counts,
        "n_spells": n_spells,
        "mean_s": mean_s,
        "short12": short12,
        "short25": short25,
        "x_after_d": x_after_d,
    }
