"""Conservative external reference repair for explicit foul/offside stoppages.

Keeps the historical reference immutable. Add D only when an explicit stopped
event is followed by a SET PIECE as the next on-ball event in the same period.
Never infer D from a challenge/advantage alone. This remains an event proxy,
not frame-exact official live/dead ground truth like Sportec BallStatus.
"""
import numpy as np

from metrica_gt import read_events, labels_from_events, ON_BALL_TYPES
from fetch_metrica import game_paths, DEFAULT_OUT


def labels_with_explicit_stoppages(events, n_frames):
    result = labels_from_events(events, n_frames)
    ordered = sorted(events, key=lambda e: (e['Period'], e['Start Frame']))
    on_ball = [e for e in ordered if e['Type'] in ON_BALL_TYPES]
    for event in ordered:
        explicit = event['Type'] in ('FAULT RECEIVED', 'CARD') or (
            event['Type'] == 'BALL LOST' and event['Subtype'] == 'OFFSIDE')
        if not explicit:
            continue
        following = [e for e in on_ball if e['Period'] == event['Period']
                     and e['Start Frame'] > event['Start Frame']]
        if following and following[0]['Type'] == 'SET PIECE':
            a = max(0, event['Start Frame']-1)
            b = min(n_frames, following[0]['Start Frame']-1)
            result[a:b] = 'D'
    return result


def build_metrica_reference_v2(game, n_frames, root=DEFAULT_OUT):
    events = read_events(game_paths(game, root)['events'])
    return labels_with_explicit_stoppages(events, n_frames)
