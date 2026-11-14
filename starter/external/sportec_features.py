"""Sportec Open DFL coordinates and independent frame labels.

Bassek et al. 2025, doi:10.1038/s41597-025-04505-y, CC BY 4.0.
BallStatus/BallPossession are read ONLY into returned evaluation ground truth,
never into model features. Coordinates are centered metres in source XML.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from features import empty_feature_arrays

ROOT = HERE.parents[1] / 'data_external' / 'sportec'


def extract_sportec(match='J03WOH', root=ROOT, include_motion=False):
    root = Path(root)
    meta = next(root.glob('*matchinformation*' + match + '.xml'))
    positions = next(root.glob('*positions*' + match + '.xml'))
    metadata = ET.parse(meta).getroot()
    general = metadata.find('.//General')
    home = general.get('HomeTeamId')
    away = general.get('GuestTeamId')
    env = metadata.find('.//Environment')
    length, width = float(env.get('PitchX')), float(env.get('PitchY'))
    tracks = []
    balls = []
    for _, element in ET.iterparse(positions, events=('end',)):
        if element.tag != 'FrameSet':
            continue
        section = element.get('GameSection')
        frames = list(element.iter('Frame'))
        if not frames or section not in ('firstHalf', 'secondHalf'):
            element.clear()
            continue
        ns = np.array([int(f.get('N')) for f in frames], dtype=np.int64)
        xy = np.array([(float(f.get('X')), float(f.get('Y'))) for f in frames], dtype=np.float32)
        period = 1 if section == 'firstHalf' else 2
        if 'BallStatus' in frames[0].attrib:
            status = np.array([int(f.get('BallStatus', '-1')) for f in frames], dtype=np.int8)
            poss = np.array([int(f.get('BallPossession', '0')) for f in frames], dtype=np.int8)
            balls.append((ns, xy, period, status, poss))
        elif element.get('TeamId') in (home, away):
            team = 0 if element.get('TeamId') == home else 1
            tracks.append((element.get('PersonId'), team, ns, xy, period))
        element.clear()
    if not balls:
        raise ValueError('No labelled ball frames found')
    offset = min(int(b[0].min()) for b in balls)
    last = max(int(b[0].max()) for b in balls)
    n = last-offset+1
    f = empty_feature_arrays(n)
    period = np.zeros(n, dtype=np.int8)
    gt = np.full(n, 'X', dtype='<U1')
    seen = np.zeros(n, bool)
    for ns, xy, p, status, poss in balls:
        idx = ns-offset
        if np.any(seen[idx]):
            raise ValueError('Overlapping ball frame numbers across periods')
        seen[idx] = True
        period[idx] = p
        present = np.isfinite(xy).all(axis=1)
        f['ball_present'][idx] = present
        f['ball_x_m'][idx] = xy[:, 0]+length/2
        f['ball_y_m'][idx] = xy[:, 1]+width/2
        gt[idx[status == 0]] = 'D'
        gt[idx[(status == 1) & (poss == 1)]] = 'H'
        gt[idx[(status == 1) & (poss == 2)]] = 'A'
    f['ball_x_norm'] = f['ball_x_m']/length
    f['ball_y_norm'] = f['ball_y_m']/width
    valid_step = f['ball_present'][1:] & f['ball_present'][:-1] & (period[1:] == period[:-1])
    speed = np.hypot(np.diff(f['ball_x_m']), np.diff(f['ball_y_m']))*25
    f['ball_speed'][1:] = np.where(valid_step, speed, np.nan)
    dh = np.full(n, np.inf)
    da = np.full(n, np.inf)
    best = np.full(n, np.inf)
    best_id = np.full(n, -1, dtype=np.int32)
    best_team = np.full(n, -1, dtype=np.int8)
    nh = np.zeros(n, np.int8)
    na = np.zeros(n, np.int8)
    identifiers = {pid: i for i, pid in enumerate(sorted({t[0] for t in tracks}))}
    for pid, team, ns, xy, p in tracks:
        idx = ns-offset
        keep = (idx >= 0) & (idx < n)
        idx, xy = idx[keep], xy[keep]
        d = np.hypot(xy[:, 0]+length/2-f['ball_x_m'][idx],
                     xy[:, 1]+width/2-f['ball_y_m'][idx])
        d = np.where(period[idx] == p, d, np.nan)
        target = dh if team == 0 else da
        target[idx] = np.fmin(target[idx], d)
        near = nh if team == 0 else na
        near[idx] += (d <= 3.).astype(np.int8)
        better = d < best[idx]
        select = idx[better]
        best[select] = d[better]
        best_id[select] = identifiers[pid]
        best_team[select] = team
    available = np.isfinite(best) & f['ball_present']
    f['nearest_team'][available] = best_team[available]
    f['nearest_dist'][available] = best[available]
    f['min_dist_home'] = np.where(np.isfinite(dh), dh, np.nan)
    f['min_dist_away'] = np.where(np.isfinite(da), da, np.nan)
    f['n_near_home'], f['n_near_away'] = nh, na
    f['majority_team'] = np.where(nh > na, 0, np.where(na > nh, 1, -1)).astype(np.int8)
    duel = (dh <= 1.) & (da <= 1.)
    hit = available & (best <= 1.) & ~duel
    f['pz_player'][hit] = best_id[hit]
    f['pz_team'][hit] = best_team[hit]
    f['pz_duel'] = duel
    f.update(n_frames=n, pitch=(length, width), period=period)
    if include_motion:
        from player_motion import summarize_motion
        xy_all = np.full((n, len(identifiers), 2), np.nan, dtype=np.float32)
        for pid, _team, ns, xy, p in tracks:
            idx = ns-offset
            keep = (idx >= 0) & (idx < n)
            idx, xy = idx[keep], xy[keep]
            keep = period[idx] == p
            xy_all[idx[keep], identifiers[pid]] = xy[keep]
        f['player_motion'] = summarize_motion(xy_all, period)
    graded = period > 0
    # A coordinate-only reset at each tracking period boundary, not a GT mask.
    graded[1:] &= period[1:] == period[:-1]
    gt[~graded] = 'X'
    return f, gt, graded
