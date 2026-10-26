"""T20: per-frame features from tactical ATD (numpy only).

Streams the same FIFA/EPTS raw format as starter/baseline.py. Does not dump coordinates to
stdout beyond aggregate counts.

    .venv/bin/python starter/features.py
"""

from __future__ import print_function

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1


def team_of_each_track(meta_path):
    """Array over track slots: 0 home, 1 away. Same convention as baseline.py."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def _parse_xyz(group):
    """Return (x, y, speed) floats from 'x,y,speed', or None if missing."""
    if not group or group[0] == MISSING:
        return None
    c1 = group.index(",")
    c2 = group.index(",", c1 + 1)
    return float(group[:c1]), float(group[c1 + 1:c2]), float(group[c2 + 1:])


def extract_features(game=GAME, data_dir=None):
    """Dict of length-n_frames arrays for the tactical ATD.

    Keys:
      ball_present   bool
      ball_x_norm, ball_y_norm, ball_speed   float (NaN if missing); x/y in 0-1
      ball_x_m, ball_y_m                    float metres (NaN if missing)
      nearest_team   int8  0=home 1=away -1=none
      min_dist_home, min_dist_away, nearest_dist   float metres (NaN if unavailable)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)

    ball_present = np.zeros(n_frames, dtype=bool)
    ball_x_norm = np.full(n_frames, np.nan, dtype=np.float64)
    ball_y_norm = np.full(n_frames, np.nan, dtype=np.float64)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float64)
    ball_x_m = np.full(n_frames, np.nan, dtype=np.float64)
    ball_y_m = np.full(n_frames, np.nan, dtype=np.float64)
    nearest_team = np.full(n_frames, NOBODY, dtype=np.int8)
    min_dist_home = np.full(n_frames, np.nan, dtype=np.float64)
    min_dist_away = np.full(n_frames, np.nan, dtype=np.float64)
    nearest_dist = np.full(n_frames, np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < 0 or frame >= n_frames:
                continue

            ball = _parse_xyz(parts[2].strip())
            if ball is None:
                continue

            bx_n, by_n, bsp = ball
            ball_present[frame] = True
            ball_x_norm[frame] = bx_n
            ball_y_norm[frame] = by_n
            ball_speed[frame] = bsp
            bx = bx_n * length
            by = by_n * width
            ball_x_m[frame] = bx
            ball_y_m[frame] = by

            best_d2 = None
            best_team = NOBODY
            d_home = None
            d_away = None
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                parsed = _parse_xyz(group)
                if parsed is None:
                    continue
                px, py, _ = parsed
                dx = px * length - bx
                dy = py * width - by
                d2 = dx * dx + dy * dy
                dist = d2 ** 0.5
                team = int(team_of_track[slot]) if slot < len(team_of_track) else NOBODY
                if team == HOME:
                    if d_home is None or dist < d_home:
                        d_home = dist
                elif team == AWAY:
                    if d_away is None or dist < d_away:
                        d_away = dist
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_team = team

            if d_home is not None:
                min_dist_home[frame] = d_home
            if d_away is not None:
                min_dist_away[frame] = d_away
            if best_d2 is not None:
                nearest_team[frame] = best_team
                nearest_dist[frame] = best_d2 ** 0.5

    return {
        "ball_present": ball_present,
        "ball_x_norm": ball_x_norm,
        "ball_y_norm": ball_y_norm,
        "ball_speed": ball_speed,
        "ball_x_m": ball_x_m,
        "ball_y_m": ball_y_m,
        "nearest_team": nearest_team,
        "min_dist_home": min_dist_home,
        "min_dist_away": min_dist_away,
        "nearest_dist": nearest_dist,
        "n_frames": n_frames,
        "pitch": (length, width),
        "game": game,
    }


def summarize(features):
    """Print aggregate counts only (safe for logs / cloud chat)."""
    n = features["n_frames"]
    bp = int(features["ball_present"].sum())
    nt = features["nearest_team"]
    print("frames: {}".format(n))
    print("ball_present: {} ({:.1f}%)".format(bp, 100.0 * bp / n))
    print("nearest_team home/away/none: {}/{}/{}".format(
        int((nt == HOME).sum()), int((nt == AWAY).sum()), int((nt == NOBODY).sum())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    args = parser.parse_args()
    print("extracting features for {} ...".format(args.game))
    feats = extract_features(game=args.game)
    summarize(feats)
