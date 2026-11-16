"""NumPy inference for the sparse H↔A switch editor (sm_v2_5 candidate)."""
from pathlib import Path
import json
import numpy as np

from state_machine import _merge_short_spells
from error_corrector_model import ensemble_team_logits
from numpy_set_temporal import load_parameters
from numpy_boundary_ownership import predict_dual
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
from team_control_model import control_design

DEFAULT_SWITCH_EDITOR_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'switch_editor_v1.json')
DEFAULT_USE_SWITCH_EDITOR = False
DEFAULT_MARGIN = 0.55
DEFAULT_MIN_ISLAND = 18
FEATURE_VERSION = 'switch_editor_v1'


def _delta(arr, period, fill=0.):
    out = np.full_like(arr, fill, dtype=np.float64)
    out[1:] = arr[1:] - arr[:-1]
    if period is not None:
        change = period[1:] != period[:-1]
        out[1:][change] = fill
    return out


def ha_switches(labels, graded):
    labels = np.asarray(labels)
    graded = np.asarray(graded, bool)
    live = graded & np.isin(labels, ['H', 'A'])
    switch = np.zeros(len(labels), bool)
    switch[1:] = (
        live[1:] & live[:-1] & (labels[1:] != labels[:-1])
        & np.isin(labels[1:], ['H', 'A']) & np.isin(labels[:-1], ['H', 'A']))
    return np.flatnonzero(switch)


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


def contested_peaks(score, graded, min_score=0.75, min_separation=18):
    score = np.asarray(score, float)
    graded = np.asarray(graded, bool)
    n = len(score)
    peaks = []
    for i in range(1, n - 1):
        if graded[i] and score[i] >= min_score and score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_separation:
                peaks.append(i)
            elif score[i] > score[peaks[-1]]:
                peaks[-1] = i
    return np.asarray(peaks, np.int32)


def _event_row(features, baseline, team_logits, boundary_logits, i, kind):
    present = float(features['ball_present'][i])
    nearest = float(features['nearest_team'][i])
    majority = float(features['majority_team'][i])
    base = baseline[i]
    base_sign = 1. if base == 'H' else (-1. if base == 'A' else 0.)
    nearest_lab = 'H' if nearest == 0 else ('A' if nearest == 1 else 'X')
    majority_lab = 'H' if majority == 0 else ('A' if majority == 1 else 'X')
    logits = float(np.nan_to_num(team_logits[i], nan=0.))
    blogits = float(np.nan_to_num(boundary_logits[i], nan=0.))
    dh = float(np.nan_to_num(features['min_dist_home'][i], nan=99.))
    da = float(np.nan_to_num(features['min_dist_away'][i], nan=99.))
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    period = features.get('period')
    bx = np.nan_to_num(features['ball_x_m'], nan=0.)
    by = np.nan_to_num(features['ball_y_m'], nan=0.)
    dspeed = _delta(speed, period)
    dbx = _delta(bx, period)
    dby = _delta(by, period)
    ddh = _delta(np.nan_to_num(features['min_dist_home'], nan=99.), period)
    dda = _delta(np.nan_to_num(features['min_dist_away'], nan=99.), period)
    score = contested_score(features)
    a = max(0, i - 5)
    b = min(len(score), i + 6)
    local_max = float(np.max(score[a:b]))
    ctx = np.asarray([
        present, base_sign, logits, abs(logits), blogits, abs(blogits),
        1. if nearest_lab == base else 0.,
        1. if majority_lab == base else 0.,
        nearest, majority,
        float(np.nan_to_num(features['nearest_dist'][i], nan=99.)),
        dh, da, abs(dh - da), min(dh, da),
        float(speed[i]), float(dspeed[i]), float(dbx[i]), float(dby[i]),
        float(ddh[i]), float(dda[i]), float(score[i]), local_max,
        float(bx[i]), float(by[i]),
    ], dtype=np.float64)
    cd = control_design(features)[i, :16]
    return np.concatenate([cd, ctx, [float(kind)]])


def collect_events(features, graded, baseline, team_logits, boundary_logits, R):
    pred_sw = ha_switches(baseline, graded)
    score = contested_score(features)
    peaks = contested_peaks(score, graded, min_score=0.75, min_separation=max(12, R // 2))
    insert = []
    for p in peaks:
        if len(pred_sw) == 0 or np.min(np.abs(pred_sw - p)) > R:
            insert.append(int(p))
    insert = np.asarray(insert, np.int32)
    x_pred = np.asarray([
        _event_row(features, baseline, team_logits, boundary_logits, int(i), 0)
        for i in pred_sw], dtype=np.float64) if len(pred_sw) else np.zeros((0, 43))
    x_ins = np.asarray([
        _event_row(features, baseline, team_logits, boundary_logits, int(i), 1)
        for i in insert], dtype=np.float64) if len(insert) else np.zeros((0, 43))
    return pred_sw, insert, x_pred, x_ins


def _walk_tree(x, tree):
    n = len(x)
    active = np.zeros(n, np.int32)
    values = np.zeros(n, np.float64)
    pending = np.ones(n, bool)
    feat = tree['feature']
    thr = tree['threshold']
    val = tree['value']
    left = tree['left']
    right = tree['right']
    leaf = tree['leaf']
    missing_left = tree['missing_left']
    while pending.any():
        idx = np.flatnonzero(pending)
        node = active[idx]
        is_leaf = leaf[node] != 0
        values[idx[is_leaf]] = val[node[is_leaf]]
        pending[idx[is_leaf]] = False
        go = idx[~is_leaf]
        if len(go) == 0:
            break
        node = active[go]
        f = feat[node]
        xv = x[go, f]
        miss = ~np.isfinite(xv)
        to_left = np.where(miss, missing_left[node] != 0, xv <= thr[node])
        active[go] = np.where(to_left, left[node], right[node])
    return values


def multiclass_logits(x, packed):
    n = len(x)
    n_classes = len(packed['baseline'])
    out = np.tile(np.asarray(packed['baseline'], float), (n, 1))
    for iteration in packed['trees']:
        for c, tree in enumerate(iteration):
            out[:, c] += _walk_tree(x, tree)
    return out


def load_switch_editor(path=DEFAULT_SWITCH_EDITOR_PATH):
    data = json.loads(Path(path).read_text())
    if data.get('feature_version') != FEATURE_VERSION:
        raise ValueError('Unsupported switch editor schema')

    def convert(block):
        trees = []
        for iteration in block['trees']:
            converted = []
            for tree in iteration:
                converted.append({
                    k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                    for k, v in tree.items()})
            trees.append(converted)
        return dict(baseline=block['baseline'], trees=trees)

    return dict(
        R=int(data['R']),
        switch=convert(data['switch']),
        insert=convert(data['insert']),
        apply=data.get('apply') or {},
    )


def _decode_switch_class(cls, R):
    cls = int(cls)
    if cls == 0:
        return 'delete', 0
    if cls == 1:
        return 'keep', 0
    if 2 <= cls <= R + 1:
        return 'retarget', -(cls - 1)
    return 'retarget', cls - (R + 1)


def _spell_bounds(labels, graded, i):
    team = labels[i]
    a = i
    while a > 0 and graded[a - 1] and labels[a - 1] == team:
        a -= 1
    b = i + 1
    n = len(labels)
    while b < n and graded[b] and labels[b] == team:
        b += 1
    return a, b


def apply_retarget(labels, graded, cut, offset):
    labels = labels.copy()
    n = len(labels)
    new_cut = int(np.clip(cut + offset, 1, n - 1))
    if new_cut == cut:
        return labels
    left_team = labels[cut - 1]
    right_team = labels[cut]
    if left_team not in ('H', 'A') or right_team not in ('H', 'A'):
        return labels
    lo, hi = (cut, new_cut) if new_cut > cut else (new_cut, cut)
    if new_cut > cut:
        labels[lo:hi] = left_team
    else:
        labels[lo:hi] = right_team
    return labels


def apply_delete(labels, graded, cut):
    labels = labels.copy()
    if cut <= 0 or cut >= len(labels):
        return labels
    left = labels[cut - 1]
    right = labels[cut]
    if left not in ('H', 'A') or right not in ('H', 'A') or left == right:
        return labels
    a_l, b_l = _spell_bounds(labels, graded, cut - 1)
    a_r, b_r = _spell_bounds(labels, graded, cut)
    if (b_l - a_l) >= (b_r - a_r):
        labels[a_r:b_r] = left
    else:
        labels[a_l:b_l] = right
    return labels


def apply_insert(labels, graded, cut, min_island):
    labels = labels.copy()
    n = len(labels)
    cut = int(np.clip(cut, 1, n - 1))
    if labels[cut] not in ('H', 'A') or labels[cut - 1] not in ('H', 'A'):
        return labels
    if labels[cut] != labels[cut - 1]:
        return labels
    team = labels[cut]
    a, b = _spell_bounds(labels, graded, cut)
    left_len = cut - a
    right_len = b - cut
    if left_len < min_island or right_len < min_island:
        return labels
    other = 'A' if team == 'H' else 'H'
    if right_len <= left_len:
        labels[cut:b] = other
    else:
        labels[a:cut] = other
    return labels


def apply_switch_editor(features, reference, graded, nodes,
                        model_path=DEFAULT_SWITCH_EDITOR_PATH,
                        margin=None, min_island=None, minimum=18):
    packed = load_switch_editor(model_path)
    apply_cfg = packed.get('apply') or {}
    R = int(apply_cfg.get('R', packed['R']))
    if margin is None:
        margin = float(apply_cfg.get('margin', DEFAULT_MARGIN))
    if min_island is None:
        min_island = int(apply_cfg.get('min_island', DEFAULT_MIN_ISLAND))
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    team_logits = ensemble_team_logits(features, graded, nodes, labels)
    _, boundary = predict_dual(
        features, graded, nodes, labels,
        load_parameters(DEFAULT_BOUNDARY_MODEL_PATH))
    pred_sw, insert, x_pred, x_ins = collect_events(
        features, graded, labels, team_logits, boundary, R)

    if len(x_pred):
        logits = multiclass_logits(x_pred, packed['switch'])
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        proba = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
        pred_cls = np.argmax(proba, axis=1)
        conf = np.max(proba, axis=1)
        for i, cut in enumerate(pred_sw):
            if conf[i] < margin:
                continue
            action, offset = _decode_switch_class(int(pred_cls[i]), R)
            if action == 'delete':
                labels = apply_delete(labels, graded, int(cut))
            elif action == 'retarget' and offset != 0:
                labels = apply_retarget(labels, graded, int(cut), int(offset))

    if len(x_ins):
        logits = multiclass_logits(x_ins, packed['insert'])
        if logits.shape[1] == 1:
            conf = 1 / (1 + np.exp(-logits[:, 0]))
            want = conf >= margin
        else:
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            proba = e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)
            conf = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            want = conf >= margin
        for i, cut in enumerate(insert):
            if want[i]:
                labels = apply_insert(labels, graded, int(cut), int(min_island))

    return _merge_short_spells(labels, graded, minimum)
