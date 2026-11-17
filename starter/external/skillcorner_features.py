"""SkillCorner Open Data broadcast tracking as 25 fps external features.

Source: https://github.com/SkillCorner/opendata (MIT, SkillCorner / PySport).
Tracking is 10 fps in centred metres; it is resampled to the 25 fps timeline
the rest of the pipeline assumes. possession.group is read ONLY into the
returned ground truth, never into features. Frames without a possession group
are 'X' (ignored by scoring and training): SkillCorner gives no dead-ball label.
Match ids are 'SC_<skillcorner id>'.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from features import empty_feature_arrays

ROOT = HERE.parents[1] / 'data_external' / 'skillcorner'
PREFIX = 'SC_'
SOURCE_FPS = 10.
TARGET_FPS = 25.


def is_skillcorner(game):
    return str(game).startswith(PREFIX)


def _paths(game, root=ROOT):
    sid = str(game)[len(PREFIX):]
    folder = Path(root) / sid
    return folder / (sid + '_match.json'), folder / (sid + '_tracking_extrapolated.jsonl')


def read_source(game, root=ROOT):
    """10 fps arrays: period, ball xy, ball detected, group, player xy, teams, pitch."""
    meta_path, track_path = _paths(game, root)
    meta = json.loads(meta_path.read_text())
    home = meta['home_team']['id']
    players = sorted(p['id'] for p in meta['players'])
    slot = {pid: k for k, pid in enumerate(players)}
    team_of = {p['id']: 0 if p['team_id'] == home else 1 for p in meta['players']}
    teams = np.array([team_of[pid] for pid in players], np.int8)
    rows = [json.loads(line) for line in track_path.read_text().splitlines() if line.strip()]
    n = len(rows)
    period = np.zeros(n, np.int8)
    ball = np.full((n, 2), np.nan, np.float32)
    detected = np.zeros(n, bool)
    group = np.zeros(n, np.int8)
    xy = np.full((n, len(players), 2), np.nan, np.float32)
    for i, row in enumerate(rows):
        if row.get('period') not in (1, 2):
            continue
        period[i] = row['period']
        b = row['ball_data']
        if b.get('x') is not None and b.get('y') is not None:
            ball[i] = (b['x'], b['y'])
            detected[i] = bool(b.get('is_detected'))
        g = row['possession']['group']
        group[i] = 1 if g == 'home team' else (2 if g == 'away team' else 0)
        for p in row['player_data']:
            k = slot.get(p['player_id'])
            if k is not None and p.get('x') is not None:
                xy[i, k] = (p['x'], p['y'])
    length, width = float(meta['pitch_length']), float(meta['pitch_width'])
    shift = np.array([length / 2, width / 2], np.float32)
    return dict(period=period, ball=ball + shift, detected=detected, group=group,
                xy=xy + shift, teams=teams, pitch=(length, width))


def resample(src):
    """Linear interpolation of coordinates within a period; nearest for labels."""
    n10 = len(src['period'])
    n = int(np.floor((n10 - 1) * TARGET_FPS / SOURCE_FPS)) + 1
    s = np.arange(n) * (SOURCE_FPS / TARGET_FPS)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, n10 - 1)
    a = (s - i0).astype(np.float32)
    near = np.where(a < .5, i0, i1)
    same = src['period'][i0] == src['period'][i1]

    def interp(values):
        lo, hi = values[i0], values[i1]
        w = a.reshape((-1,) + (1,) * (values.ndim - 1))
        mixed = lo * (1 - w) + hi * w
        ok = np.isfinite(lo) & np.isfinite(hi) & same.reshape(w.shape)
        return np.where(ok, mixed, values[near])

    return dict(period=src['period'][near], ball=interp(src['ball']),
                detected=src['detected'][i0] & src['detected'][i1] & same,
                group=src['group'][near], xy=interp(src['xy']),
                teams=src['teams'], pitch=src['pitch'])


def features_from_arrays(ball, present, xy, teams, period, pitch, include_motion=False):
    """Same feature definitions as extract_sportec, from aligned arrays."""
    length, width = pitch
    n = len(period)
    f = empty_feature_arrays(n)
    present = present & np.isfinite(ball).all(axis=1) & (period > 0)
    f['ball_present'] = present
    f['ball_x_m'] = np.where(present, ball[:, 0], np.nan).astype(np.float64)
    f['ball_y_m'] = np.where(present, ball[:, 1], np.nan).astype(np.float64)
    f['ball_x_norm'] = f['ball_x_m'] / length
    f['ball_y_norm'] = f['ball_y_m'] / width
    valid_step = present[1:] & present[:-1] & (period[1:] == period[:-1])
    speed = np.hypot(np.diff(f['ball_x_m']), np.diff(f['ball_y_m'])) * TARGET_FPS
    f['ball_speed'][1:] = np.where(valid_step, speed, np.nan)
    d = np.hypot(xy[:, :, 0] - f['ball_x_m'][:, None], xy[:, :, 1] - f['ball_y_m'][:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    home, away = teams == 0, teams == 1
    dh = d[:, home].min(axis=1) if home.any() else np.full(n, np.inf)
    da = d[:, away].min(axis=1) if away.any() else np.full(n, np.inf)
    nh = (d[:, home] <= 3.).sum(axis=1).astype(np.int8)
    na = (d[:, away] <= 3.).sum(axis=1).astype(np.int8)
    best_id = np.argmin(d, axis=1).astype(np.int32)
    best = d[np.arange(n), best_id]
    best_team = teams[best_id].astype(np.int8)
    available = np.isfinite(best) & present
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
        f['player_motion'] = summarize_motion(xy, period)
    return f


def extract_skillcorner(game, root=ROOT, include_motion=False):
    """Return (features, gt, graded) on a 25 fps timeline."""
    r = resample(read_source(game, root))
    f = features_from_arrays(r['ball'], r['detected'], r['xy'], r['teams'],
                             r['period'], r['pitch'], include_motion=include_motion)
    graded = r['period'] > 0
    graded[1:] &= r['period'][1:] == r['period'][:-1]
    gt = np.full(len(graded), 'X', dtype='<U1')
    gt[r['group'] == 1] = 'H'
    gt[r['group'] == 2] = 'A'
    gt[~graded] = 'X'
    return f, gt, graded


def skillcorner_geometry(game, root=ROOT):
    """25 fps player xy (metres, 0..L/W), team per slot, period."""
    r = resample(read_source(game, root))
    return r['xy'], r['teams'], r['period']


def cached_skillcorner(game):
    cache = Path('evaluation/reports/feature_cache') / ('skillcorner_' + game + '.npz')
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            feats = {k: data[k] for k in data.files if k not in ('gt', 'graded')}
            gt, graded = data['gt'].copy(), data['graded'].copy()
        feats['n_frames'] = int(feats['n_frames'])
        return feats, gt, graded
    print('extract SkillCorner', game, flush=True)
    feats, gt, graded = extract_skillcorner(game, include_motion=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **feats, gt=gt, graded=graded)
    return feats, gt, graded
