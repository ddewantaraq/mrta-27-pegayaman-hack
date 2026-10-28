"""Smart Tagging soft priors (not labels).

Builds per-frame arrays from the tactical Smart Tagging XML via load_smart_tagging.load:
  - set_piece_dead[bool]: SET PIECES -> encourage dead ball (capped; tags are ~35s envelopes)
  - phase_team[int8]: weak team vote from selected attacking phases (HOME/AWAY/NOBODY)

Never copy phase intervals wholesale as H/A/D. Team=N/A is ignored for phase votes.
"""

from __future__ import print_function

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME, source_frames  # noqa: E402
from load_smart_tagging import load  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1

SET_PIECE_CODE = "set pieces"

# Default: only the first 2s of each SET PIECES tag (tags are mostly fixed ~35s windows).
DEFAULT_SET_PIECE_DEAD_CAP_FRAMES = 50

# Attacking / progression phases used only as weak team votes (not as possession spans).
# Dropped ball-in-final-third / ball-in-the-box: too much pitch coverage for a weak vote.
PHASE_VOTE_CODES = frozenset({
    "build up",
    "progression",
    "attacking transition",
    "creating chances",
})


def _norm(s):
    return (s or "").strip().lower()


def team_code_from_tag(team_text):
    """Barcelona -> HOME, Atletic Club -> AWAY, else NOBODY (includes N/A)."""
    t = _norm(team_text)
    if not t or t in ("n/a", "na", "none"):
        return NOBODY
    if "barcel" in t:
        return HOME
    if "atlet" in t or "athletic" in t:
        return AWAY
    return NOBODY


def build_smart_priors(
    n_frames=None,
    game=GAME,
    data_dir=None,
    events=None,
    set_piece_dead_cap_frames=DEFAULT_SET_PIECE_DEAD_CAP_FRAMES,
    use_phase_votes=True,
):
    """Return dict with set_piece_dead, phase_team, and summary counts.

    set_piece_dead_cap_frames: mark only the first N frames of each SET PIECES
    event as dead. None = full event span (legacy T23 / --full-set-piece-dead).
    """
    n_frames = n_frames or source_frames("tactical", game)
    set_piece_dead = np.zeros(n_frames, dtype=bool)
    phase_team = np.full(n_frames, NOBODY, dtype=np.int8)

    if events is None:
        events = load(game=game, data_dir=data_dir)

    n_set = 0
    n_phase = 0
    for event in events:
        code = _norm(event.get("code"))
        start = int(event["start_frame"])
        end = int(event["end_frame"])  # half-open
        start = max(0, start)
        end = min(n_frames, end)
        if start >= end:
            continue

        if code == SET_PIECE_CODE:
            if set_piece_dead_cap_frames is None:
                dead_end = end
            else:
                dead_end = min(end, start + int(set_piece_dead_cap_frames))
            set_piece_dead[start:dead_end] = True
            n_set += 1
            continue

        if use_phase_votes and code in PHASE_VOTE_CODES:
            team = team_code_from_tag(event.get("tags", {}).get("Team"))
            if team == NOBODY:
                continue
            # Last overlapping phase wins on a frame (file order).
            phase_team[start:end] = team
            n_phase += 1

    return {
        "set_piece_dead": set_piece_dead,
        "phase_team": phase_team,
        "n_frames": n_frames,
        "n_set_piece_events": n_set,
        "n_phase_vote_events": n_phase,
        "n_set_piece_frames": int(set_piece_dead.sum()),
        "n_phase_vote_frames": int((phase_team != NOBODY).sum()),
        "set_piece_dead_cap_frames": set_piece_dead_cap_frames,
        "use_phase_votes": bool(use_phase_votes),
    }


def summarize_priors(priors):
    cap = priors.get("set_piece_dead_cap_frames")
    cap_txt = "full" if cap is None else str(cap)
    print(
        "smart priors: set_piece events={} frames={} (cap={}); "
        "phase-vote events={} frames={} (on={})".format(
            priors["n_set_piece_events"],
            priors["n_set_piece_frames"],
            cap_txt,
            priors["n_phase_vote_events"],
            priors["n_phase_vote_frames"],
            priors.get("use_phase_votes", True),
        )
    )


if __name__ == "__main__":
    p = build_smart_priors()
    summarize_priors(p)
