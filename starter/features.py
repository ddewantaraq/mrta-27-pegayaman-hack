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
from vc_possession import (  # noqa: E402
    DEFAULT_DZ_RADIUS_M,
    DEFAULT_PZ_RADIUS_M,
    zone_control,
)

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1

# Same circle as state_machine control_dist_m: players inside "control" range
DEFAULT_CLUSTER_RADIUS_M = 3.0


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


def proximity_to_ball(player_xy_m, team_of_player, ball_xy_m, cluster_radius_m):
    """Nearest / cluster stats for one frame.

    player_xy_m: iterable of (x_m, y_m) or None for missing players.
    team_of_player: parallel iterable of HOME/AWAY/NOBODY.
    ball_xy_m: (x_m, y_m).
    Returns dict with nearest_team, nearest_dist, min_dist_home, min_dist_away,
    n_near_home, n_near_away, majority_team.
    """
    bx, by = ball_xy_m
    r2 = float(cluster_radius_m) * float(cluster_radius_m)
    best_d2 = None
    best_team = NOBODY
    d_home = None
    d_away = None
    nh = 0
    na = 0
    for xy, team in zip(player_xy_m, team_of_player):
        if xy is None:
            continue
        px, py = xy
        dx = px - bx
        dy = py - by
        d2 = dx * dx + dy * dy
        dist = d2 ** 0.5
        t = int(team)
        if t == HOME:
            if d_home is None or dist < d_home:
                d_home = dist
            if d2 <= r2:
                nh += 1
        elif t == AWAY:
            if d_away is None or dist < d_away:
                d_away = dist
            if d2 <= r2:
                na += 1
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_team = t
    maj = NOBODY
    if nh > na:
        maj = HOME
    elif na > nh:
        maj = AWAY
    return {
        "nearest_team": best_team,
        "nearest_dist": (best_d2 ** 0.5) if best_d2 is not None else np.nan,
        "min_dist_home": d_home if d_home is not None else np.nan,
        "min_dist_away": d_away if d_away is not None else np.nan,
        "n_near_home": nh,
        "n_near_away": na,
        "majority_team": maj,
    }


def empty_feature_arrays(n_frames):
    """Allocate the standard length-n feature arrays (NaN / False / NOBODY)."""
    return {
        "ball_present": np.zeros(n_frames, dtype=bool),
        "ball_x_norm": np.full(n_frames, np.nan, dtype=np.float64),
        "ball_y_norm": np.full(n_frames, np.nan, dtype=np.float64),
        "ball_speed": np.full(n_frames, np.nan, dtype=np.float64),
        "ball_x_m": np.full(n_frames, np.nan, dtype=np.float64),
        "ball_y_m": np.full(n_frames, np.nan, dtype=np.float64),
        "nearest_team": np.full(n_frames, NOBODY, dtype=np.int8),
        "min_dist_home": np.full(n_frames, np.nan, dtype=np.float64),
        "min_dist_away": np.full(n_frames, np.nan, dtype=np.float64),
        "nearest_dist": np.full(n_frames, np.nan, dtype=np.float64),
        "n_near_home": np.zeros(n_frames, dtype=np.int8),
        "n_near_away": np.zeros(n_frames, dtype=np.int8),
        "majority_team": np.full(n_frames, NOBODY, dtype=np.int8),
        # Vidal-Codina zone hit, before the gain test. -1 / NOBODY / False if none.
        "pz_player": np.full(n_frames, -1, dtype=np.int32),
        "pz_team": np.full(n_frames, NOBODY, dtype=np.int8),
        "pz_duel": np.zeros(n_frames, dtype=bool),
    }


def extract_features(
    game=GAME,
    data_dir=None,
    cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
    pz_radius_m=DEFAULT_PZ_RADIUS_M,
    dz_radius_m=DEFAULT_DZ_RADIUS_M,
):
    """Dict of length-n_frames arrays for the tactical ATD.

    Keys:
      ball_present   bool
      ball_x_norm, ball_y_norm, ball_speed   float (NaN if missing); x/y in 0-1
      ball_x_m, ball_y_m                    float metres (NaN if missing)
      nearest_team   int8  0=home 1=away -1=none
      min_dist_home, min_dist_away, nearest_dist   float metres (NaN if unavailable)
      n_near_home, n_near_away   int8  players within cluster_radius_m of ball
      majority_team  int8  team with more near players, or -1 on tie / none
      pz_player, pz_team, pz_duel  Vidal-Codina zone (before gain test)
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    team_of_track = team_of_each_track(meta_path)

    arrays = empty_feature_arrays(n_frames)
    ball_present = arrays["ball_present"]
    ball_x_norm = arrays["ball_x_norm"]
    ball_y_norm = arrays["ball_y_norm"]
    ball_speed = arrays["ball_speed"]
    ball_x_m = arrays["ball_x_m"]
    ball_y_m = arrays["ball_y_m"]
    nearest_team = arrays["nearest_team"]
    min_dist_home = arrays["min_dist_home"]
    min_dist_away = arrays["min_dist_away"]
    nearest_dist = arrays["nearest_dist"]
    n_near_home = arrays["n_near_home"]
    n_near_away = arrays["n_near_away"]
    majority_team = arrays["majority_team"]
    pz_player = arrays["pz_player"]
    pz_team = arrays["pz_team"]
    pz_duel = arrays["pz_duel"]

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

            player_xy = []
            teams = []
            pids = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                pids.append(slot)
                parsed = _parse_xyz(group)
                if parsed is None:
                    player_xy.append(None)
                    teams.append(NOBODY)
                    continue
                px, py, _ = parsed
                player_xy.append((px * length, py * width))
                teams.append(
                    int(team_of_track[slot]) if slot < len(team_of_track) else NOBODY)

            prox = proximity_to_ball(player_xy, teams, (bx, by), cluster_radius_m)
            zpid, zteam, zduel = zone_control(
                player_xy, teams, pids, (bx, by), pz_radius_m, dz_radius_m)
            pz_player[frame] = zpid
            pz_team[frame] = zteam
            pz_duel[frame] = zduel
            nearest_team[frame] = prox["nearest_team"]
            nearest_dist[frame] = prox["nearest_dist"]
            min_dist_home[frame] = prox["min_dist_home"]
            min_dist_away[frame] = prox["min_dist_away"]
            n_near_home[frame] = prox["n_near_home"]
            n_near_away[frame] = prox["n_near_away"]
            majority_team[frame] = prox["majority_team"]

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
        "n_near_home": n_near_home,
        "n_near_away": n_near_away,
        "majority_team": majority_team,
        "pz_player": pz_player,
        "pz_team": pz_team,
        "pz_duel": pz_duel,
        "n_frames": n_frames,
        "pitch": (length, width),
        "game": game,
        "cluster_radius_m": float(cluster_radius_m),
        "pz_radius_m": float(pz_radius_m),
        "dz_radius_m": float(dz_radius_m),
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
