"""Event-first possession: detect ball touches, classify the touching team,
decode team possession over touches with a two-state Viterbi.

Only the team of visible live frames inside the reference's live spans is
re-decoded; D/X and hidden frames keep the reference. Inference is NumPy
over exported boosted trees (training: research/experiments/event_first.py).
"""
import json
from pathlib import Path

import numpy as np

from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells

DEFAULT_EVENT_FIRST_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'event_first_v1.json')
DEFAULT_USE_EVENT_FIRST = True  # sm_v2_7
FEATURE_VERSION = 'event_first_v1'
TOUCH_DIST_M = 1.5
TURN_DEG = 25.
SPEED_JUMP = 2.
LAG = 2
FPS = 25.


def _shift(values, k, fill=np.nan):
    out = np.full_like(values, fill, dtype=np.float64)
    if k > 0:
        out[k:] = values[:-k]
    elif k < 0:
        out[:k] = values[-k:]
    else:
        out[:] = values
    return out


def _ball_kinematics(features):
    bx = np.asarray(features['ball_x_m'], np.float64)
    by = np.asarray(features['ball_y_m'], np.float64)
    present = np.asarray(features['ball_present'], bool)
    period = np.asarray(features.get('period', np.zeros(len(bx))), np.float64)
    ok = present.copy()
    for k in range(1, LAG + 1):
        ok &= (_shift(present.astype(float), k, 0.) > 0) & (_shift(present.astype(float), -k, 0.) > 0)
        ok &= (_shift(period, k) == period) & (_shift(period, -k) == period)
    scale = FPS / LAG
    vbx, vby = (bx - _shift(bx, LAG)) * scale, (by - _shift(by, LAG)) * scale
    vax, vay = (_shift(bx, -LAG) - bx) * scale, (_shift(by, -LAG) - by) * scale
    sb, sa = np.hypot(vbx, vby), np.hypot(vax, vay)
    cos = (vbx * vax + vby * vay) / np.maximum(sb * sa, 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos, -1., 1.)))
    angle = np.where((sb > .5) & (sa > .5), angle, 0.)
    return ok, sb, sa, angle


def touch_candidates(features, graded):
    """First frame of each run of plausible touches (turn or speed jump near a player)."""
    ok, sb, sa, angle = _ball_kinematics(features)
    near = np.nan_to_num(np.asarray(features['nearest_dist'], np.float64), nan=99.) <= TOUCH_DIST_M
    hit = ok & np.asarray(graded, bool) & near & (
        (angle >= TURN_DEG) | (np.abs(sa - sb) >= SPEED_JUMP))
    start = hit & ~np.r_[False, hit[:-1]]
    return np.flatnonzero(start), (sb, sa, angle)


def _sign(labels):
    return np.where(labels == 'H', 1., np.where(labels == 'A', -1., 0.))


def touch_design(features, reference, team_logits, idx, kin):
    sb, sa, angle = kin
    f = features
    sign = _sign(np.asarray(reference))
    n = len(sign)
    logits = np.nan_to_num(np.asarray(team_logits, np.float64), nan=0.)

    def at(values, offset=0, fill=np.nan):
        j = np.clip(idx + offset, 0, n - 1)
        return np.asarray(values, np.float64)[j]

    gap_prev = np.diff(np.r_[idx[:1] - 250, idx]).astype(np.float64)
    gap_next = np.diff(np.r_[idx, idx[-1:] + 250]).astype(np.float64) if len(idx) else gap_prev
    cols = [
        at(f['nearest_team']), at(np.nan_to_num(f['nearest_dist'], nan=99.)),
        at(np.nan_to_num(f['min_dist_home'], nan=99.)), at(np.nan_to_num(f['min_dist_away'], nan=99.)),
        at(f['n_near_home']), at(f['n_near_away']),
        at(f['nearest_team'], 5), at(f['nearest_team'], -5),
        at(angle), at(sa - sb), at(sb), at(sa),
        at(np.nan_to_num(f['ball_x_m'], nan=0.)), at(np.nan_to_num(f['ball_y_m'], nan=0.)),
        at(logits), at(logits, 10), at(logits, -10),
        at(sign), at(sign, 10), at(sign, -10),
        np.minimum(gap_prev, 250.), np.minimum(gap_next, 250.),
    ]
    return np.column_stack(cols).astype(np.float64) if len(idx) else np.zeros((0, len(cols)))


def load_event_first(path=DEFAULT_EVENT_FIRST_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported event-first schema')
    trees = [{k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
              for k, v in tree.items()} for tree in data['trees']]
    return dict(baseline=float(data['baseline']), trees=trees,
                switch_cost=float(data['switch_cost']), margin=float(data['margin']))


def viterbi_two_state(logit, switch_cost, start_sign=0.):
    """Most likely H(+)/A(-) sequence given per-touch home log-odds."""
    k = len(logit)
    if k == 0:
        return np.zeros(0)
    emit = np.column_stack([-np.logaddexp(0., -logit), -np.logaddexp(0., logit)])
    score = emit[0] + np.array([.5, -.5]) * start_sign
    back = np.zeros((k, 2), np.int8)
    for t in range(1, k):
        stay, move = score, score[::-1] - switch_cost
        back[t] = np.where(stay >= move, [0, 1], [1, 0])
        score = np.maximum(stay, move) + emit[t]
    state = np.zeros(k, np.int8)
    state[-1] = int(np.argmax(score))
    for t in range(k - 1, 0, -1):
        state[t - 1] = back[t, state[t]]
    return np.where(state == 0, 1., -1.)


def live_spans(labels, graded):
    live = np.asarray(graded, bool) & np.isin(labels, ['H', 'A'])
    edges = np.flatnonzero(np.diff(np.r_[0, live.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def decode_event_first(features, reference, graded, touch_logit, idx, switch_cost, margin):
    labels = np.asarray(reference).copy()
    present = np.asarray(features['ball_present'], bool)
    out = labels.copy()
    for a, b in live_spans(labels, graded):
        inside = (idx >= a) & (idx < b)
        if not inside.any():
            continue
        ti, tl = idx[inside], touch_logit[inside]
        state = viterbi_two_state(tl, switch_cost, _sign(labels[a:a + 1])[0])
        bounds = np.r_[ti, b]
        for k in range(len(ti)):
            team = 'H' if state[k] > 0 else 'A'
            if team == labels[ti[k]] or abs(tl[k]) < margin or np.sign(tl[k]) != state[k]:
                continue
            seg = np.arange(bounds[k], bounds[k + 1])
            out[seg[present[seg]]] = team
    return out


def apply_event_first(features, reference, graded, nodes, model_path=DEFAULT_EVENT_FIRST_PATH,
                      minimum=18, team_logits=None):
    from error_corrector_model import ensemble_team_logits
    graded = np.asarray(graded, bool)
    reference = np.asarray(reference)
    model = load_event_first(model_path)
    idx, kin = touch_candidates(features, graded)
    if team_logits is None:
        team_logits = ensemble_team_logits(features, graded, nodes, reference)
    logit = boosted_logits(touch_design(features, reference, team_logits, idx, kin), model)
    labels = decode_event_first(features, reference, graded, logit, idx,
                                model['switch_cost'], model['margin'])
    return _merge_short_spells(labels, graded, minimum)
