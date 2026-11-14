"""External player geometry caches; deliberately independent of event labels."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from player_nodes import pack_player_nodes


def sportec_nodes(game,stride=5):
    from sportec_features import ROOT
    metadata = ET.parse(next(ROOT.glob('*matchinformation*'+game+'.xml'))).getroot()
    general = metadata.find('.//General')
    team_ids = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    tracks, ball_first, ball_last = [], None, None
    for _, e in ET.iterparse(next(ROOT.glob('*positions*'+game+'.xml')), events=('end',)):
        if e.tag != 'FrameSet':
            continue
        frames = list(e.iter('Frame'))
        if not frames or e.get('GameSection') not in ('firstHalf', 'secondHalf'):
            e.clear()
            continue
        ns = np.array([int(f.get('N')) for f in frames], np.int64)
        if 'BallStatus' in frames[0].attrib:
            # Only frame numbers identify alignment. Annotation values are not read.
            ball_first = int(ns.min()) if ball_first is None else min(ball_first, int(ns.min()))
            ball_last = int(ns.max()) if ball_last is None else max(ball_last, int(ns.max()))
        elif e.get('TeamId') in team_ids:
            xy = np.array([(float(f.get('X')), float(f.get('Y'))) for f in frames], np.float32)
            tracks.append((e.get('PersonId'), team_ids.index(e.get('TeamId')), ns, xy,
                           1 if e.get('GameSection') == 'firstHalf' else 2))
        e.clear()
    if ball_first is None:
        raise ValueError('Missing ball frame alignment')
    people = sorted({t[0] for t in tracks})
    lookup = {pid: k for k, pid in enumerate(people)}
    teams = np.zeros(len(people), np.int8)
    n = ball_last-ball_first+1
    xy_all = np.full((n, len(people), 2), np.nan, np.float32)
    period = np.zeros(n, np.int8)
    for pid, team, ns, xy, half in tracks:
        idx = ns-ball_first
        keep = (idx >= 0) & (idx < n)
        idx, xy = idx[keep], xy[keep]
        xy_all[idx, lookup[pid]] = xy+[52.5, 34.]
        period[idx] = half
        teams[lookup[pid]] = team
    return pack_player_nodes(xy_all, teams, period,stride=stride)


def metrica_nodes(game, return_raw=False,stride=5):
    from metrica_features import _read_tracking_csv, _parse_player_columns
    from fetch_metrica import game_paths, DEFAULT_OUT
    parts, teams, period = [], [], None
    paths = game_paths(game, DEFAULT_OUT)
    for team, key in enumerate(('home', 'away')):
        header, rows = _read_tracking_csv(paths[key])
        players, _, _ = _parse_player_columns(header)
        if period is None:
            period = np.array([int(float(r[0])) for r in rows], np.int8)
        xy = np.full((len(rows), len(players), 2), np.nan, np.float32)
        for k, columns in enumerate(players):
            for axis, column in enumerate(columns):
                xy[:, k, axis] = [float(r[column]) if r[column].strip() else np.nan for r in rows]
                xy[:, k, axis] *= 105. if axis == 0 else 68.
        parts.append(xy)
        teams.extend([team]*len(players))
    xy, teams = np.concatenate(parts, axis=1), np.asarray(teams)
    if return_raw:
        return xy, teams, period
    return pack_player_nodes(xy, teams, period,stride=stride)


def load_player_nodes(game,stride=5):
    suffix='_nodes_v1.npz' if stride==5 else '_nodes_stride'+str(stride)+'_v1.npz'
    path = Path('evaluation/reports/feature_cache')/(game+suffix)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return data['nodes'].copy()
    if game == 'Sample_Game_3':
        from metrica3_geometry import geometry
        _, _, nodes = geometry(return_nodes=True,node_stride=stride)
    else:
        nodes = metrica_nodes(game,stride=stride) if game.startswith('Sample_Game_') else sportec_nodes(game,stride=stride)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, nodes=nodes)
    return nodes
