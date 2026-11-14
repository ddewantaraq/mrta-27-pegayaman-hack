"""Ball-independent motion cache for external sources and challenge ATD."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from player_motion import summarize_motion


def metrica_motion(game, root=None):
    from metrica_features import _read_tracking_csv, _parse_player_columns
    from fetch_metrica import game_paths, DEFAULT_OUT
    paths = game_paths(game, root or DEFAULT_OUT)
    chunks = []
    period = None
    for side in ('home', 'away'):
        header, rows = _read_tracking_csv(paths[side])
        players, _, _ = _parse_player_columns(header)
        if period is None:
            period = np.array([int(float(r[0])) for r in rows], dtype=np.int8)
        xy = np.full((len(rows), len(players), 2), np.nan, np.float32)
        for k, (ix, iy) in enumerate(players):
            for axis, col in enumerate((ix, iy)):
                xy[:, k, axis] = [float(r[col]) if r[col].strip() else np.nan for r in rows]
                xy[:, k, axis] *= 105. if axis == 0 else 68.
        chunks.append(xy)
    return summarize_motion(np.concatenate(chunks, axis=1), period)


def challenge_motion(game='BAR-ATH', include_nodes=False):
    from features import team_of_each_track, _parse_xyz
    from games import require, source_frames, GAMES, half_video_blocks
    meta, raw = require(['atd_meta', 'atd_raw'], game)
    teams = team_of_each_track(meta)
    n = source_frames('tactical', game)
    length, width = GAMES[game]['pitch']
    xy = np.full((n, len(teams), 2), np.nan, np.float32)
    with open(raw) as handle:
        for line in handle:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            i = int(parts[0])-1
            if not 0 <= i < n:
                continue
            for k, group in enumerate(parts[1].split(';')):
                if k >= len(teams):
                    break
                point = _parse_xyz(group)
                if point is not None:
                    xy[i, k] = (point[0]*length, point[1]*width)
    period = np.zeros(n, np.int8)
    for half, a, b in half_video_blocks(game):
        period[a:b+1] = half
    motion = summarize_motion(xy, period)
    if include_nodes:
        from player_nodes import pack_player_nodes
        nodes = pack_player_nodes(xy, teams, period, length=length, width=width)
        return motion, nodes, period
    return motion


def load_external_motion(game):
    path = Path('evaluation/reports/feature_cache') / (game+'_motion_v1.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data['motion'].copy()
    if game.startswith('Sample_Game_'):
        motion = metrica_motion(game)
    else:
        from sportec_features import extract_sportec
        features, _, _ = extract_sportec(game, include_motion=True)
        motion = features['player_motion']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, motion=motion)
    return motion
