"""Contested-band and switch-event features for the sparse switch editor."""
import numpy as np

from team_control_model import control_design
from visible_ownership import _delta

FEATURE_VERSION = 'switch_editor_v1'
DEFAULT_CONTEST_DIST = 1.0
DEFAULT_NEAR_M = 5.0


def ha_switches(labels, graded):
    """Frame indices where live H/A label changes."""
    labels = np.asarray(labels)
    graded = np.asarray(graded, bool)
    live = graded & np.isin(labels, ['H', 'A'])
    switch = np.zeros(len(labels), bool)
    switch[1:] = (
        live[1:] & live[:-1]
        & (labels[1:] != labels[:-1])
        & np.isin(labels[1:], ['H', 'A'])
        & np.isin(labels[:-1], ['H', 'A'])
    )
    return np.flatnonzero(switch)


def contested_score(features, contest_dist=DEFAULT_CONTEST_DIST, near_m=DEFAULT_NEAR_M):
    """Soft contested score in [0, ~2+]: duel proximity + ball activity."""
    present = np.asarray(features['ball_present'], bool)
    dh = np.nan_to_num(features['min_dist_home'], nan=99.)
    da = np.nan_to_num(features['min_dist_away'], nan=99.)
    duel = (np.abs(dh - da) <= float(contest_dist)) & (dh <= near_m) & (da <= near_m)
    speed = np.nan_to_num(features['ball_speed'], nan=0.)
    period = features.get('period')
    dspeed = np.abs(_delta(speed, period))
    # Normalize activity roughly into [0,1]
    activity = np.clip(speed / 15., 0., 1.) + np.clip(dspeed / 10., 0., 1.)
    closing = np.clip((-_delta(dh, period) - _delta(da, period)) / 4., 0., 1.)
    score = duel.astype(float) + 0.5 * activity + 0.5 * closing
    score = np.where(present, score, 0.)
    return score.astype(np.float64)


def contested_peaks(score, graded, min_score=0.75, min_separation=18):
    """Local maxima of contested score on graded frames, sparsified."""
    score = np.asarray(score, float)
    graded = np.asarray(graded, bool)
    n = len(score)
    peaks = []
    i = 1
    while i < n - 1:
        if graded[i] and score[i] >= min_score and score[i] >= score[i - 1] and score[i] >= score[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_separation:
                peaks.append(i)
            elif score[i] > score[peaks[-1]]:
                peaks[-1] = i
        i += 1
    return np.asarray(peaks, np.int32)


def frame_context_row(features, baseline, team_logits, boundary_logits, i):
    """One-row event features at frame i."""
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
    speed = float(np.nan_to_num(features['ball_speed'][i], nan=0.))
    period = features.get('period')
    bx = np.nan_to_num(features['ball_x_m'], nan=0.)
    by = np.nan_to_num(features['ball_y_m'], nan=0.)
    dbx = _delta(bx, period)
    dby = _delta(by, period)
    dspeed = _delta(np.nan_to_num(features['ball_speed'], nan=0.), period)
    ddh = _delta(np.nan_to_num(features['min_dist_home'], nan=99.), period)
    dda = _delta(np.nan_to_num(features['min_dist_away'], nan=99.), period)
    score = contested_score(features)
    # Local peakiness
    a = max(0, i - 5)
    b = min(len(score), i + 6)
    local_max = float(np.max(score[a:b]))
    return np.asarray([
        present, base_sign, logits, abs(logits), blogits, abs(blogits),
        1. if nearest_lab == base else 0.,
        1. if majority_lab == base else 0.,
        nearest, majority,
        float(np.nan_to_num(features['nearest_dist'][i], nan=99.)),
        dh, da, abs(dh - da), min(dh, da),
        speed, float(dspeed[i]), float(dbx[i]), float(dby[i]),
        float(ddh[i]), float(dda[i]),
        float(score[i]), local_max,
        float(bx[i]), float(by[i]),
    ], dtype=np.float64)


def event_feature_matrix(features, baseline, team_logits, boundary_logits, indices,
                         event_kind):
    """Stack per-event rows; event_kind 0=pred_switch, 1=insert_candidate."""
    if len(indices) == 0:
        return np.zeros((0, 26), np.float64)
    # Prepend a few control_design scalars at the event frame for context.
    base = control_design(features)
    rows = []
    kind = float(event_kind)
    for i in indices:
        ctx = frame_context_row(features, baseline, team_logits, boundary_logits, int(i))
        # Truncate control_design to first 16 dims to keep event features compact.
        cd = base[int(i), :16]
        rows.append(np.concatenate([cd, ctx, [kind]]))
    return np.asarray(rows, dtype=np.float64)


def collect_candidate_events(features, graded, baseline, team_logits, boundary_logits,
                             retarget_radius=25, peak_score=0.75):
    """Return dict of pred-switch and insert-candidate indices + feature matrices."""
    pred_sw = ha_switches(baseline, graded)
    score = contested_score(features)
    peaks = contested_peaks(score, graded, min_score=peak_score,
                            min_separation=max(12, retarget_radius // 2))
    # Insert candidates: peaks far from any pred switch.
    insert = []
    for p in peaks:
        if len(pred_sw) == 0 or np.min(np.abs(pred_sw - p)) > retarget_radius:
            insert.append(int(p))
    insert = np.asarray(insert, np.int32)
    x_pred = event_feature_matrix(
        features, baseline, team_logits, boundary_logits, pred_sw, 0)
    x_ins = event_feature_matrix(
        features, baseline, team_logits, boundary_logits, insert, 1)
    return dict(
        pred_switches=pred_sw,
        insert_candidates=insert,
        x_pred=x_pred,
        x_insert=x_ins,
        contested=score,
    )
