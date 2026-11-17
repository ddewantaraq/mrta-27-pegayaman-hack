"""Touch-level supervision from unused external event files (P1 step 1).

Parses Sportec ``events_raw`` XML and SkillCorner ``dynamic_events.csv`` --
both downloaded and hash-verified by the fetch scripts but never parsed --
into touch/duel markers aligned to the 25 fps feature timeline.

Training/evaluation supervision only; never an inference input on BAR-ATH.
Event annotations are read ONLY into returned markers, never into features.
"""
import csv
import json
import xml.etree.ElementTree as ET
from bisect import bisect_left
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np

Touch = namedtuple('Touch', ['source', 'frame_n', 'frame10', 'frame25', 'kind',
                             'team', 'possession_change', 'possession_loss',
                             'event_id'])

SC_TOUCH_TYPES = {'player_possession': 'touch', 'on_ball_engagement': 'duel'}


def _first(root, pattern):
    found = next(Path(root).glob(pattern), None)
    if found is None:
        raise ValueError('Missing file for pattern {}'.format(pattern))
    return found


def _parse_time(value):
    return datetime.fromisoformat(value).timestamp()


def _sportec_frame_clock(positions_path):
    """(N, T) timeline from the ball FrameSets; N matches extract_sportec."""
    clock = []
    seen = set()
    for _, element in ET.iterparse(positions_path, events=('end',)):
        if element.tag != 'FrameSet':
            continue
        frames = list(element.iter('Frame'))
        is_ball = frames and 'BallStatus' in frames[0].attrib
        section = element.get('GameSection')
        if is_ball and section not in seen:
            seen.add(section)
            for frame in frames:
                if frame.get('N') is not None and frame.get('T') is not None:
                    clock.append((int(frame.get('N')), _parse_time(frame.get('T'))))
            if seen >= {'firstHalf', 'secondHalf'}:
                element.clear()
                break
        element.clear()
    if not clock:
        raise ValueError('No ball N/T frames in {}'.format(positions_path))
    clock.sort(key=lambda pair: pair[1])
    return clock


def sportec_frame_offset(match, root):
    """Feature index 0 == this tracking frame N (mirrors extract_sportec)."""
    clock = _sportec_frame_clock(_first(Path(root), '*positions*' + match + '.xml'))
    return min(n for n, _ in clock)


def parse_sportec_events_raw(match, root):
    """Touch/duel markers for one Sportec match, keyed by tracking frame N."""
    root = Path(root)
    meta = ET.parse(_first(root, '*matchinformation*' + match + '.xml')).getroot()
    general = meta.find('.//General')
    home, away = general.get('HomeTeamId'), general.get('GuestTeamId')
    clock = _sportec_frame_clock(_first(root, '*positions*' + match + '.xml'))
    times = [t for _, t in clock]
    ns = [n for n, _ in clock]
    touches = []
    events = ET.parse(_first(root, '*events_raw*' + match + '.xml')).getroot()
    for event in events.iter('Event'):
        stamp = event.get('EventTime')
        if stamp is None:
            continue
        pos = bisect_left(times, _parse_time(stamp))
        if pos >= len(times):
            pos = len(times) - 1
        elif pos > 0 and times[pos] - _parse_time(stamp) > _parse_time(stamp) - times[pos - 1]:
            pos -= 1
        frame_n = ns[pos]
        for play in event.iter('Play'):
            team = play.get('Team')
            touches.append(Touch('sportec', frame_n, None, None, 'play',
                                 0 if team == home else (1 if team == away else None),
                                 False, False, event.get('EventId')))
        for duel in event.iter('TacklingGame'):
            team = duel.get('WinnerTeam')
            touches.append(Touch('sportec', frame_n, None, None, 'duel',
                                 0 if team == home else (1 if team == away else None),
                                 duel.get('PossessionChange') == 'true', False,
                                 event.get('EventId')))
    touches.sort(key=lambda t: t.frame_n)
    return touches


def frame25_from_frame10(frame10):
    """Inverse of skillcorner_features.resample: 10 fps row -> 25 fps frame."""
    return int(round(frame10 * 2.5))


def parse_skillcorner_dynamic(game, root):
    """Touch/duel markers for one SkillCorner match on the 25 fps timeline."""
    sid = str(game)[len('SC_'):]
    folder = Path(root) / sid
    meta = json.loads((folder / (sid + '_match.json')).read_text())
    home = meta['home_team']['id']
    team_of = {p['id']: p['team_id'] for p in meta.get('players', [])}
    touches = []
    with open(folder / (sid + '_dynamic_events.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            kind = SC_TOUCH_TYPES.get(row['event_type'])
            if kind is None:
                continue
            if kind == 'duel':
                # team_id is the pressing side; the possessing side comes
                # from the player_in_possession_id (verified: never equal).
                possessor = row.get('player_in_possession_id') or ''
                poss_team = team_of.get(int(possessor)) if possessor else None
                team = None if poss_team is None else (0 if poss_team == home else 1)
            else:
                team_id = row.get('team_id') or ''
                team = 0 if team_id == str(home) else (1 if team_id else None)
            frame10 = int(row['frame_start'])
            touches.append(Touch(
                'skillcorner', None, frame10, frame25_from_frame10(frame10), kind,
                team, False,
                (row.get('team_possession_loss_in_phase') or '') == 'True',
                row.get('event_id') or ''))
    touches.sort(key=lambda t: t.frame10)
    return touches


def accel_candidates(features, graded, accel_thresh=10., slowdown=3.,
                     dist_m=2.0, lag=2, fps=25.):
    """ELASTIC-style kick/receive candidates from ball acceleration.

    Kick: acceleration peak near a player. Receive: sharp slowdown into a
    player. Thresholds are physics-motivated constants (kicks register
    10-50 m/s^2 on tracking data), not tuned on any evaluation match.
    Returns first frames of runs, mirroring touch_candidates.
    """
    bx = np.asarray(features['ball_x_m'], np.float64)
    by = np.asarray(features['ball_y_m'], np.float64)
    present = np.asarray(features['ball_present'], bool)
    period = np.asarray(features.get('period', np.zeros(len(bx))), np.float64)
    ok = present.copy()
    for k in range(1, lag + 1):
        for shifted, ref in ((np.roll(present, k), np.roll(period, k)),
                             (np.roll(present, -k), np.roll(period, -k))):
            ok &= shifted & (np.roll(period, 0) == ref)
    ok[:lag] = False
    ok[-lag:] = False
    scale = fps / lag
    vbx = (bx - np.roll(bx, lag)) * scale
    vby = (by - np.roll(by, lag)) * scale
    vax = (np.roll(bx, -lag) - bx) * scale
    vay = (np.roll(by, -lag) - by) * scale
    sb = np.hypot(vbx, vby)
    sa = np.hypot(vax, vay)
    accel = np.hypot(vax - vbx, vay - vby) * scale
    near = (np.nan_to_num(np.asarray(features['nearest_dist'], np.float64),
                          nan=99.) <= dist_m)
    hit = (ok & np.asarray(graded, bool) & near
           & ((accel >= accel_thresh) | ((sb - sa) >= slowdown)))
    start = hit & ~np.r_[False, hit[:-1]]
    return np.flatnonzero(start)


def align_hits(candidates, references, tol):
    """Greedy one-to-one hits within +-tol; returns (precision, recall, hits)."""
    cands = sorted(int(c) for c in candidates)
    refs = sorted(int(r) for r in references)
    used = [False] * len(refs)
    hits = 0
    for cand in cands:
        best = None
        for i, ref in enumerate(refs):
            if used[i] or abs(ref - cand) > tol:
                continue
            if best is None or abs(ref - cand) < abs(refs[best] - cand):
                best = i
        if best is not None:
            used[best] = True
            hits += 1
    precision = hits / len(cands) if cands else 0.0
    recall = hits / len(refs) if refs else 0.0
    return precision, recall, hits
