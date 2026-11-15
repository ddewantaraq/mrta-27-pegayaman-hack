"""Geometry-only loader for the reserved Metrica third sample match.

Reads metadata/tracking only, never events. Per-range EPTS channel mappings
are essential: substitutions change raw slot assignments in this match.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import empty_feature_arrays
from player_motion import summarize_motion

ROOT = Path(__file__).resolve().parents[2]/'data_external'/'metrica'/'Sample_Game_3'


def geometry(root=ROOT, return_nodes=False,node_stride=5):
    root = Path(root)
    meta = ET.parse(root/'Sample_Game_3_metadata.xml').getroot()
    players = list(meta.iter('Player'))
    player_index = {p.get('id'): i for i, p in enumerate(players)}
    home_id = next(meta.iter('Team')).get('id')
    teams = np.array([0 if p.get('teamId') == home_id else 1 for p in players], np.int8)
    channel = {p.get('id'): p.get('playerId') for p in meta.iter('PlayerChannel')}
    schedules = []
    for spec in meta.iter('DataFormatSpecification'):
        ids = [channel[p.get('playerChannelId')] for p in spec.iter('PlayerChannelRef')
               if p.get('playerChannelId').endswith('_x')]
        schedules.append((int(spec.get('startFrame'))-1, int(spec.get('endFrame')),
                          np.array([player_index[p] for p in ids], dtype=int)))
    schedules.sort(key=lambda row: row[0])
    n = max(s[1] for s in schedules)
    width = float(meta.find('.//FieldSize/Width').text)
    height = float(meta.find('.//FieldSize/Height').text)
    period = np.zeros(n, np.int8)
    params = {p.findtext('Name'): p.findtext('Value') for p in meta.iter('ProviderParameter')}
    for half, prefix in ((1, 'first_half'), (2, 'second_half')):
        a, b = int(params[prefix+'_start'])-1, int(params[prefix+'_end'])
        period[a:b] = half
    f = empty_feature_arrays(n)
    xy = np.full((n, len(players), 2), np.nan, np.float32)
    schedule = 0
    seen = np.zeros(n, bool)
    with (root/'Sample_Game_3_tracking.txt').open() as handle:
        for line in handle:
            fields = line.strip().split(':')
            if len(fields) != 3:
                raise ValueError('Unexpected EPTS frame format')
            frame = int(fields[0])-1
            if not 0 <= frame < n or seen[frame]:
                raise ValueError('Invalid or duplicated tracking frame')
            seen[frame] = True
            while schedule+1 < len(schedules) and frame >= schedules[schedule][1]:
                schedule += 1
            a, b, ids = schedules[schedule]
            if not a <= frame < b:
                raise ValueError('Frame outside EPTS mapping schedule')
            values = np.fromstring(fields[1].replace(';', ','), sep=',')
            if len(values) != 2*len(ids):
                raise ValueError('Player group count disagrees with metadata')
            xy[frame, ids] = values.reshape(-1, 2)*[width, height]
            ball = np.fromstring(fields[2], sep=',')
            if len(ball) != 2:
                raise ValueError('Unexpected ball coordinate fields')
            if np.isfinite(ball).all():
                f['ball_present'][frame] = True
                f['ball_x_norm'][frame], f['ball_y_norm'][frame] = ball
                f['ball_x_m'][frame], f['ball_y_m'][frame] = ball*[width, height]
    if not seen.all():
        raise ValueError('Tracking is not dense on declared EPTS timeline')
    ball_xy = np.column_stack([f['ball_x_m'], f['ball_y_m']])
    distance = np.linalg.norm(xy-ball_xy[:, None], axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    dh, da = distance[:, teams == 0].min(axis=1), distance[:, teams == 1].min(axis=1)
    nearest = distance.argmin(axis=1)
    closest = distance[np.arange(n), nearest]
    visible = f['ball_present'] & np.isfinite(closest)
    f['nearest_team'][visible] = teams[nearest[visible]]
    f['nearest_dist'][visible] = closest[visible]
    f['min_dist_home'] = np.where(np.isfinite(dh), dh, np.nan)
    f['min_dist_away'] = np.where(np.isfinite(da), da, np.nan)
    nh = np.sum(distance[:, teams == 0] <= 3., axis=1).astype(np.int8)
    na = np.sum(distance[:, teams == 1] <= 3., axis=1).astype(np.int8)
    f['n_near_home'], f['n_near_away'] = nh, na
    f['majority_team'] = np.where(nh > na, 0, np.where(na > nh, 1, -1)).astype(np.int8)
    duel = (dh <= 1.) & (da <= 1.)
    controlled = visible & (closest <= 1.) & ~duel
    f['pz_duel'] = duel
    f['pz_player'][controlled] = nearest[controlled]
    f['pz_team'][controlled] = teams[nearest[controlled]]
    valid = f['ball_present'][1:] & f['ball_present'][:-1] & (period[1:] == period[:-1])
    speed = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1)*25.
    f['ball_speed'][1:] = np.where(valid, speed, np.nan)
    f['player_motion'] = summarize_motion(xy, period)
    f.update(n_frames=n, period=period, pitch=(width, height), game='Sample_Game_3')
    graded = period > 0
    graded[1:] &= period[1:] == period[:-1]
    if return_nodes:
        from player_nodes import pack_player_nodes
        return f, graded, pack_player_nodes(xy, teams, period,stride=node_stride, length=width, width=height)
    return f, graded


def cached_geometry():
    path = Path('evaluation/reports/feature_cache/metrica3_geometry_v1.npz')
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            f = {key: data[key].copy() for key in data.files if key != 'graded'}
            graded = data['graded'].copy()
        f['n_frames'] = int(f['n_frames'])
        return f, graded
    f, graded = geometry()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **f, graded=graded)
    return f, graded


if __name__ == '__main__':
    f, graded = cached_geometry()
    print('Reserved geometry loaded:', f['n_frames'], 'frames;',
          int(f['ball_present'].sum()), 'ball observations. No events loaded.')
