"""Insert-only contested H↔A switch proposals (research / optional sm_v2_5)."""
from pathlib import Path
import json
import numpy as np

from state_machine import _merge_short_spells
from error_corrector_model import ensemble_team_logits

DEFAULT_INSERT_PARAMS_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'insert_only_v1.json')
DEFAULT_USE_INSERT_ONLY = False


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def contested_score(features, contest_dist=1.0, near_m=5.0):
    present = np.asarray(features['ball_present'], bool)
    dh = np.nan_to_num(features['min_dist_home'], nan=99.)
    da = np.nan_to_num(features['min_dist_away'], nan=99.)
    duel = (np.abs(dh - da) <= float(contest_dist)) & (dh <= near_m) & (da <= near_m)
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    period = features.get('period')
    dspeed = np.abs(_delta(speed, period))
    activity = np.clip(speed / 15., 0., 1.) + np.clip(dspeed / 10., 0., 1.)
    closing = np.clip((-_delta(dh, period) - _delta(da, period)) / 4., 0., 1.)
    score = duel.astype(float) + 0.5 * activity + 0.5 * closing
    return np.where(present, score, 0.).astype(np.float64)


def ha_switches(labels, graded):
    labels = np.asarray(labels)
    graded = np.asarray(graded, bool)
    live = graded & np.isin(labels, ['H', 'A'])
    switch = np.zeros(len(labels), bool)
    switch[1:] = (
        live[1:] & live[:-1] & (labels[1:] != labels[:-1])
        & np.isin(labels[1:], ['H', 'A']) & np.isin(labels[:-1], ['H', 'A']))
    return np.flatnonzero(switch)


def contested_peaks(score, graded, min_score=1.0, min_separation=25):
    score = np.asarray(score, float)
    graded = np.asarray(graded, bool)
    peaks = []
    for i in range(1, len(score) - 1):
        if graded[i] and score[i] >= min_score and score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_separation:
                peaks.append(i)
            elif score[i] > score[peaks[-1]]:
                peaks[-1] = i
    return np.asarray(peaks, np.int32)


def _spell_bounds(labels, graded, i):
    team = labels[i]
    a = i
    while a > 0 and graded[a - 1] and labels[a - 1] == team:
        a -= 1
    b = i + 1
    while b < len(labels) and graded[b] and labels[b] == team:
        b += 1
    return a, b


def load_params(path=DEFAULT_INSERT_PARAMS_PATH):
    data = json.loads(Path(path).read_text())
    return data.get('params', data)


def apply_insert_only(features, reference, graded, nodes, params=None,
                      params_path=DEFAULT_INSERT_PARAMS_PATH, minimum=18):
    if params is None:
        params = load_params(params_path)
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    logits = ensemble_team_logits(features, graded, nodes, labels)
    score = contested_score(features)
    used = []
    for _ in range(int(params['k'])):
        pred_sw = ha_switches(labels, graded)
        peaks = contested_peaks(
            score, graded, min_score=float(params['min_score']),
            min_separation=int(params['sep']))
        window = int(params['window'])
        min_island = int(params['min_island'])
        min_disagree = float(params['min_disagree'])
        cands = []
        for cut in peaks:
            cut = int(cut)
            if cut <= 0 or cut >= len(labels) - 1:
                continue
            if labels[cut] not in ('H', 'A') or labels[cut - 1] != labels[cut]:
                continue
            if len(pred_sw) and np.min(np.abs(pred_sw - cut)) < int(params['sep']):
                continue
            if any(abs(cut - u) < int(params['sep']) for u in used):
                continue
            a, b = _spell_bounds(labels, graded, cut)
            if (cut - a) < min_island or (b - cut) < min_island:
                continue
            left_mean = float(np.mean(np.nan_to_num(logits[max(0, cut - window):cut], nan=0.)))
            right_mean = float(np.mean(np.nan_to_num(logits[cut:min(len(logits), cut + window)], nan=0.)))
            left_want = 'H' if left_mean >= 0 else 'A'
            right_want = 'H' if right_mean >= 0 else 'A'
            if left_want == right_want:
                continue
            disagree = abs(left_mean - right_mean)
            if disagree < min_disagree:
                continue
            cur = labels[cut]
            flip_right = (right_want != cur and left_want == cur)
            flip_left = (left_want != cur and right_want == cur)
            if not flip_right and not flip_left:
                continue
            rank = float(score[cut]) * (1.0 + disagree)
            cands.append((rank, cut, 'right' if flip_right else 'left',
                          left_want, right_want, a, b))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        rank, cut, side, left_want, right_want, a, b = cands[0]
        if side == 'right':
            labels[cut:b] = right_want
        else:
            labels[a:cut] = left_want
        used.append(cut)
    return _merge_short_spells(labels, graded, minimum)
