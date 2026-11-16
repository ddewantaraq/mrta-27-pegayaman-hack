"""Build the same feature dict as starter.features from Metrica CSV tracking.

Pitch 105x68 m, coords normalised 0-1 (Metrica convention). Ball speed is
derived from consecutive positions (Metrica tracking has no speed column).
"""

from __future__ import print_function

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.dirname(HERE)
ROOT = os.path.dirname(STARTER)
sys.path.insert(0, STARTER)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from features import (  # noqa: E402
    DEFAULT_CLUSTER_RADIUS_M,
    HOME,
    AWAY,
    NOBODY,
    empty_feature_arrays,
    proximity_to_ball,
)
from vc_possession import (  # noqa: E402
    DEFAULT_DZ_RADIUS_M,
    DEFAULT_PZ_RADIUS_M,
    zone_control,
)
from fetch_metrica import DEFAULT_OUT, game_paths  # noqa: E402

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
FPS = 25.0


def _parse_player_columns(header):
    """Return (players, ball_x, ball_y) from a Metrica tracking header.

    Sample CSV header (after two meta rows):
      Period,Frame,Time [s],Player11,,Player1,,...,Ball,
    Each player (and Ball) occupies a named x column followed by an empty y column.
    """
    cols = list(header)
    ball_x = ball_y = None
    players = []  # (ix, iy) pairs
    i = 0
    while i < len(cols):
        name = (cols[i] or "").strip()
        lower = name.lower()
        if lower in ("period", "frame", "time [s]", "time"):
            i += 1
            continue
        if lower in ("ball", "ball_x", "ballx"):
            ball_x = i
            ball_y = i + 1 if i + 1 < len(cols) else None
            i += 2
            continue
        if lower in ("ball_y", "bally"):
            ball_y = i
            i += 1
            continue
        if not name:
            # Orphan empty column; skip
            i += 1
            continue
        # Named player column + following y (empty or duplicate name)
        if i + 1 < len(cols):
            players.append((i, i + 1))
            i += 2
        else:
            i += 1
    return players, ball_x, ball_y


def _read_tracking_csv(path):
    """Return (header, rows as list of string lists), skipping first 2 meta rows."""
    with open(path, "r") as handle:
        # Two metadata rows then a header row.
        next(handle)
        next(handle)
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row]
    return header, rows


def _float_or_nan(value):
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def extract_metrica_features(
    game_name,
    root=DEFAULT_OUT,
    cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
    pz_radius_m=DEFAULT_PZ_RADIUS_M,
    dz_radius_m=DEFAULT_DZ_RADIUS_M,
    include_motion=False,
):
    """Feature dict compatible with label_possession for one Metrica sample game."""
    paths = game_paths(game_name, root)
    for key in ("home", "away"):
        if not os.path.isfile(paths[key]):
            raise IOError(
                "missing {}; run starter/external/fetch_metrica.py".format(paths[key]))

    home_header, home_rows = _read_tracking_csv(paths["home"])
    away_header, away_rows = _read_tracking_csv(paths["away"])
    if len(home_rows) != len(away_rows):
        raise ValueError("home/away row count mismatch: {} vs {}".format(
            len(home_rows), len(away_rows)))

    home_players, ball_x_h, ball_y_h = _parse_player_columns(home_header)
    away_players, ball_x_a, ball_y_a = _parse_player_columns(away_header)
    # Prefer ball from home file; fall back to away.
    ball_x_col = ball_x_h if ball_x_h is not None else ball_x_a
    ball_y_col = ball_y_h if ball_y_h is not None else ball_y_a
    if ball_x_col is None or ball_y_col is None:
        # Some dumps put Ball as last two unnamed columns of home.
        raise ValueError("could not find Ball_x/Ball_y in {}".format(paths["home"]))

    # Frame index column
    try:
        frame_col = home_header.index("Frame")
    except ValueError:
        frame_col = 1
    try:
        period_col = home_header.index("Period")
    except ValueError:
        period_col = 0

    n_frames = len(home_rows)
    arrays = empty_feature_arrays(n_frames)
    periods = np.zeros(n_frames, dtype=np.int8)
    length, width = PITCH_LENGTH_M, PITCH_WIDTH_M

    # Build team lists once (all home players HOME, all away AWAY)
    home_teams = [HOME] * len(home_players)
    away_teams = [AWAY] * len(away_players)

    prev_bx = prev_by = np.nan
    prev_t = None

    for i, (hrow, arow) in enumerate(zip(home_rows, away_rows)):
        periods[i] = int(float(hrow[period_col])) if hrow[period_col] else 0
        # Prefer reading by Frame column for sanity; index i is dense 0..n-1
        # after skip of meta; Metrica Frame starts at 1.
        bx_n = _float_or_nan(hrow[ball_x_col] if ball_x_h is not None else arow[ball_x_a])
        by_n = _float_or_nan(hrow[ball_y_col] if ball_y_h is not None else arow[ball_y_a])
        if np.isnan(bx_n) or np.isnan(by_n):
            prev_bx = prev_by = np.nan
            prev_t = None
            continue

        arrays["ball_present"][i] = True
        arrays["ball_x_norm"][i] = bx_n
        arrays["ball_y_norm"][i] = by_n
        bx = bx_n * length
        by = by_n * width
        arrays["ball_x_m"][i] = bx
        arrays["ball_y_m"][i] = by

        # Time from column if present, else i / fps
        try:
            tcol = home_header.index("Time [s]")
            t = _float_or_nan(hrow[tcol])
        except ValueError:
            t = i / FPS
        if prev_t is not None and not np.isnan(prev_bx):
            dt = t - prev_t if t is not None and not np.isnan(t) else (1.0 / FPS)
            if dt > 0:
                dx = bx - prev_bx
                dy = by - prev_by
                arrays["ball_speed"][i] = ((dx * dx + dy * dy) ** 0.5) / dt
        prev_bx, prev_by, prev_t = bx, by, t

        player_xy = []
        teams = []
        pids = []
        for ix, iy in home_players:
            px = _float_or_nan(hrow[ix]) if ix < len(hrow) else np.nan
            py = _float_or_nan(hrow[iy]) if iy < len(hrow) else np.nan
            if np.isnan(px) or np.isnan(py):
                player_xy.append(None)
            else:
                player_xy.append((px * length, py * width))
            teams.append(HOME)
            pids.append(len(pids))
        for ix, iy in away_players:
            px = _float_or_nan(arow[ix]) if ix < len(arow) else np.nan
            py = _float_or_nan(arow[iy]) if iy < len(arow) else np.nan
            if np.isnan(px) or np.isnan(py):
                player_xy.append(None)
            else:
                player_xy.append((px * length, py * width))
            teams.append(AWAY)
            pids.append(10000 + len(pids))

        prox = proximity_to_ball(player_xy, teams, (bx, by), cluster_radius_m)
        zpid, zteam, zduel = zone_control(
            player_xy, teams, pids, (bx, by), pz_radius_m, dz_radius_m)
        arrays["pz_player"][i] = zpid
        arrays["pz_team"][i] = zteam
        arrays["pz_duel"][i] = zduel
        arrays["nearest_team"][i] = prox["nearest_team"]
        arrays["nearest_dist"][i] = prox["nearest_dist"]
        arrays["min_dist_home"][i] = prox["min_dist_home"]
        arrays["min_dist_away"][i] = prox["min_dist_away"]
        arrays["n_near_home"][i] = prox["n_near_home"]
        arrays["n_near_away"][i] = prox["n_near_away"]
        arrays["majority_team"][i] = prox["majority_team"]

    result = {
        "ball_present": arrays["ball_present"],
        "ball_x_norm": arrays["ball_x_norm"],
        "ball_y_norm": arrays["ball_y_norm"],
        "ball_speed": arrays["ball_speed"],
        "ball_x_m": arrays["ball_x_m"],
        "ball_y_m": arrays["ball_y_m"],
        "nearest_team": arrays["nearest_team"],
        "min_dist_home": arrays["min_dist_home"],
        "min_dist_away": arrays["min_dist_away"],
        "nearest_dist": arrays["nearest_dist"],
        "n_near_home": arrays["n_near_home"],
        "n_near_away": arrays["n_near_away"],
        "majority_team": arrays["majority_team"],
        "pz_player": arrays["pz_player"],
        "pz_team": arrays["pz_team"],
        "pz_duel": arrays["pz_duel"],
        "n_frames": n_frames,
        "pitch": (length, width),
        "game": game_name,
        "cluster_radius_m": float(cluster_radius_m),
        "pz_radius_m": float(pz_radius_m),
        "dz_radius_m": float(dz_radius_m),
        "period": periods,
    }
    if include_motion:
        from external.motion_features import metrica_motion
        result['player_motion'] = metrica_motion(game_name, root=root)
    return result


if __name__ == "__main__":
    import argparse
    from features import summarize  # noqa: E402

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default="Sample_Game_1")
    parser.add_argument("--root", default=DEFAULT_OUT)
    args = parser.parse_args()
    feats = extract_metrica_features(args.game, root=args.root)
    summarize(feats)
    print("periods unique: {}".format(sorted(set(int(p) for p in feats["period"]))))
