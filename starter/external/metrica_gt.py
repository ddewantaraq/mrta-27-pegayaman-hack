"""Event-derived per-frame H/A/D/X labels for Metrica sample games.

Follows Link & Hoernig 2017 at team level:
  - team possession continues through ball-in-air / void phases
  - switches when the opponent first gains control (on-ball event)
  - no possession while the ball is out of play (BALL OUT -> D until SET PIECE)

On-ball events that set team: PASS, RECOVERY, SHOT, SET PIECE, BALL LOST.
CHALLENGE is ignored (duel, not control).
X before first kickoff and between periods (halftime).
"""

from __future__ import print_function

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fetch_metrica import DEFAULT_OUT, game_paths  # noqa: E402

LABEL_H, LABEL_A, LABEL_D, LABEL_X = "H", "A", "D", "X"

ON_BALL_TYPES = frozenset({
    "PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST",
})
# Also treat CARD / FAULT lightly? Plan lists only the set above.


def _team_letter(team_str):
    t = (team_str or "").strip().lower()
    if t == "home":
        return LABEL_H
    if t == "away":
        return LABEL_A
    return None


def read_events(events_path):
    """List of event dicts with int Start Frame / End Frame / Period."""
    with open(events_path, "r") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            try:
                start = int(float(row["Start Frame"]))
            except (KeyError, ValueError, TypeError):
                continue
            end_raw = row.get("End Frame") or "0"
            try:
                end = int(float(end_raw))
            except (ValueError, TypeError):
                end = start
            if end <= 0:
                end = start
            try:
                period = int(float(row.get("Period") or 0))
            except (ValueError, TypeError):
                period = 0
            rows.append({
                "Team": (row.get("Team") or "").strip(),
                "Type": (row.get("Type") or "").strip().upper(),
                "Subtype": (row.get("Subtype") or "").strip().upper(),
                "Period": period,
                "Start Frame": start,
                "End Frame": end,
            })
    return rows


def labels_from_events(events, n_frames, frame_offset=1):
    """Per-frame labels. frame_offset=1 means Metrica Frame 1 -> index 0."""
    labels = np.full(n_frames, LABEL_X, dtype="<U1")
    if not events:
        return labels

    # Period spans from tracking-aligned event frames
    period_first = {}
    period_last = {}
    for ev in events:
        p = ev["Period"]
        s = ev["Start Frame"]
        e = max(ev["End Frame"], s)
        if p not in period_first or s < period_first[p]:
            period_first[p] = s
        if p not in period_last or e > period_last[p]:
            period_last[p] = e

    # Live-play windows: fill with last on-ball team; then overlay D for outages.
    # First pass: walk events in order, write H/A from each on-ball Start Frame
    # forward until the next on-ball of either team (or period end).
    on_ball = [ev for ev in events if ev["Type"] in ON_BALL_TYPES]
    on_ball.sort(key=lambda e: (e["Period"], e["Start Frame"]))

    for i, ev in enumerate(on_ball):
        letter = _team_letter(ev["Team"])
        if letter is None:
            continue
        start_idx = ev["Start Frame"] - frame_offset
        if start_idx < 0 or start_idx >= n_frames:
            continue
        # End at next on-ball start, or period last frame
        if i + 1 < len(on_ball) and on_ball[i + 1]["Period"] == ev["Period"]:
            end_frame = on_ball[i + 1]["Start Frame"]
        else:
            end_frame = period_last.get(ev["Period"], ev["Start Frame"]) + 1
        end_idx = min(n_frames, end_frame - frame_offset)
        if end_idx > start_idx:
            labels[start_idx:end_idx] = letter

    # Dead: BALL OUT start -> next SET PIECE start (same period), else period end
    ball_outs = [ev for ev in events if ev["Type"] == "BALL OUT"]
    set_pieces = [ev for ev in events if ev["Type"] == "SET PIECE"]
    for out in ball_outs:
        start_idx = out["Start Frame"] - frame_offset
        if start_idx < 0 or start_idx >= n_frames:
            continue
        end_frame = period_last.get(out["Period"], out["Start Frame"]) + 1
        for sp in set_pieces:
            if (sp["Period"] == out["Period"]
                    and sp["Start Frame"] > out["Start Frame"]):
                end_frame = sp["Start Frame"]
                break
        end_idx = min(n_frames, end_frame - frame_offset)
        if end_idx > start_idx:
            labels[start_idx:end_idx] = LABEL_D

    # X: frames never touched stay X (pre-kickoff). Also force X between periods.
    periods = sorted(period_first)
    for a, b in zip(periods, periods[1:]):
        # Gap from end of period a to start of period b
        gap_lo = period_last[a] - frame_offset + 1
        gap_hi = period_first[b] - frame_offset
        lo = max(0, gap_lo)
        hi = min(n_frames, gap_hi)
        if hi > lo:
            labels[lo:hi] = LABEL_X

    return labels


def build_metrica_gt(game_name, n_frames, root=DEFAULT_OUT):
    """Labels array for one sample game."""
    paths = game_paths(game_name, root)
    if not os.path.isfile(paths["events"]):
        raise IOError(
            "missing {}; run starter/external/fetch_metrica.py".format(paths["events"]))
    events = read_events(paths["events"])
    return labels_from_events(events, n_frames)


def summarize_labels(labels):
    n = len(labels)
    for lab in (LABEL_H, LABEL_A, LABEL_D, LABEL_X):
        c = int(np.sum(labels == lab))
        print("  {}: {} ({:.1f}%)".format(lab, c, 100.0 * c / max(1, n)))


if __name__ == "__main__":
    import argparse
    from metrica_features import extract_metrica_features  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="Sample_Game_1")
    parser.add_argument("--root", default=DEFAULT_OUT)
    args = parser.parse_args()
    feats = extract_metrica_features(args.game, root=args.root)
    labels = build_metrica_gt(args.game, feats["n_frames"], root=args.root)
    print("gt labels for {} (n={}):".format(args.game, feats["n_frames"]))
    summarize_labels(labels)
