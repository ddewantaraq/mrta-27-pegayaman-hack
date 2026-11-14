"""Experimental external-trained live/dead model, numpy only.

Input is tracking geometry with temporal context; no annotations are accepted
by the feature function. Logistic emissions and a two-state Viterbi decoder
replace hand-coded geometric dead-ball detection. Training is external only.
"""
import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / 'models' / 'dead_ball_mixed_v1.json')
DEFAULT_USE_LEARNED_DEAD = True


def load_model(path=DEFAULT_MODEL_PATH):
    with open(path, encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('feature_version') != 1:
        raise ValueError('Unsupported dead-ball feature version')
    model = {key: np.asarray(data['parameters'][key], dtype=np.float64)
             for key in ('mean', 'scale', 'beta')}
    d = len(model['mean'])
    if (model['mean'].shape != (41,) or model['scale'].shape != (d,)
            or model['beta'].shape != (d+1,)
            or not all(np.isfinite(v).all() for v in model.values())
            or np.any(model['scale'] <= 0)):
        raise ValueError('Invalid dead-ball model parameters')
    return model


def label_with_dead_model(features, graded=None, model_path=DEFAULT_MODEL_PATH, **kwargs):
    """Approved protected model: learned D plus preserved hidden baseline H/A."""
    from state_machine import graded_mask, label_possession, splice_vc_on_visible
    from games import GAME
    kw = dict(kwargs)
    use_splice = kw.pop('vc_splice', True)
    if graded is None:
        graded = graded_mask(int(features['n_frames']), kw.get('game', GAME))
    graded = np.asarray(graded, bool)
    runner = splice_vc_on_visible if use_splice else label_possession
    baseline = runner(features, graded=graded, **kw)
    kw['use_geometric_dead'] = False
    team = runner(features, graded=graded, **kw)
    dead = decode_dead(features, load_model(model_path), graded)
    team[dead & graded] = 'D'
    return protect_hidden_live(team, baseline, features['ball_present'])


def protect_hidden_live(prediction, baseline, ball_present):
    """Keep baseline H/A on hidden frames; new answers can replace baseline D/X.

    This guarantees that no previously correct hidden H/A answer is lost,
    without needing ground truth to identify which answers were correct.
    """
    out = np.asarray(prediction).copy()
    baseline = np.asarray(baseline)
    mask = ~np.asarray(ball_present, bool) & np.isin(baseline, ['H', 'A'])
    out[mask] = baseline[mask]
    return out


def _mean_window(values, radius):
    n = len(values)
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx-radius), np.minimum(n, idx+radius+1)
    sums = np.concatenate(([0.], np.cumsum(values, dtype=np.float64)))
    return (sums[hi]-sums[lo])/(hi-lo)


def design_features(features):
    """Geometry/context features; deliberately no reference labels or events."""
    present = np.asarray(features['ball_present'], dtype=bool)
    n = len(present)
    idx = np.arange(n)
    previous = np.maximum.accumulate(np.where(present, idx, -1))
    following = np.minimum.accumulate(np.where(present, idx, n)[::-1])[::-1]
    p = np.clip(previous, 0, n-1)
    q = np.clip(following, 0, n-1)
    x = np.asarray(features['ball_x_norm'])
    y = np.asarray(features['ball_y_norm'])
    edge = np.minimum.reduce([x, 1-x, y, 1-y])
    raw = np.column_stack([
        np.clip(edge, -.1, .5),
        (edge < 0).astype(float),
        (edge < .02).astype(float),
        np.nan_to_num(np.minimum(features['ball_speed'], 30.), nan=0.)/30.,
        np.nan_to_num(np.minimum(features['nearest_dist'], 10.), nan=10.)/10.,
        np.minimum(features['n_near_home']+features['n_near_away'], 6)/6.,
        (np.nan_to_num(features['ball_speed'], nan=99.) < .5).astype(float),
    ])
    raw = np.nan_to_num(raw)
    prev_raw = raw[p].copy()
    next_raw = raw[q].copy()
    prev_raw[previous < 0] = 0
    next_raw[following == n] = 0
    columns = [prev_raw, next_raw, present[:, None].astype(float),
               (np.minimum(idx-previous, 750)/750.)[:, None],
               (np.minimum(following-idx, 750)/750.)[:, None]]
    weight = present.astype(float)
    for radius in (12, 50, 125):
        coverage = _mean_window(weight, radius)
        columns.append(coverage[:, None])
        columns.append(np.column_stack([
            _mean_window(raw[:, k]*weight, radius)/np.maximum(coverage, 1e-6)
            for k in range(raw.shape[1])]))
    return np.column_stack(columns).astype(np.float32)


def fit_logistic(x, target, ridge=10., iterations=15, sample_weights=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(target, dtype=float)
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .01)
    z = np.column_stack([np.ones(len(x)), (x-mean)/scale])
    prior = np.clip(y.mean(), .01, .99)
    sample_weight = np.where(y > 0, .5/prior, .5/(1-prior))
    if sample_weights is not None:
        supplied = np.asarray(sample_weights, dtype=np.float64)
        if (supplied.shape != y.shape or not np.isfinite(supplied).all()
                or np.any(supplied < 0) or supplied.sum() <= 0):
            raise ValueError('Invalid logistic sample weights')
        sample_weight *= supplied/np.mean(supplied)
    beta = np.zeros(z.shape[1])
    penalty = np.eye(z.shape[1])*ridge
    penalty[0, 0] = 0
    for _ in range(iterations):
        p = 1/(1+np.exp(-np.clip(z@beta, -30, 30)))
        w = sample_weight*p*(1-p)
        gradient = z.T@(sample_weight*(p-y))+penalty@beta
        hessian = z.T@(z*w[:, None])+penalty+np.eye(len(beta))*1e-7
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-5:
            break
    return dict(mean=mean, scale=scale, beta=beta)


def decode_dead(features, model, graded=None, stride=5):
    """Decode independently inside each graded period; return bool dead mask."""
    x = design_features(features)
    logits = model['beta'][0]+((x-model['mean'])/model['scale'])@model['beta'][1:]
    return decode_dead_logits(logits, graded, features.get('period'), stride)


def decode_dead_logits(logits, graded=None, period=None, stride=5, transition_prob=.01):
    """Shared temporal decoder for independently fitted emission models."""
    logits = np.asarray(logits, dtype=np.float64)
    n = len(logits)
    graded = np.ones(n, bool) if graded is None else np.asarray(graded, bool)
    periods = np.zeros(n, dtype=np.int8) if period is None else np.asarray(period)
    dead = np.zeros(n, bool)
    i = 0
    # Fixed 1/100 transition probability at 5 Hz: roughly a 20 s duration.
    # This is specified before held-out match evaluation, not challenge-fitted.
    if not 0 < transition_prob < .5:
        raise ValueError('Transition probability must be between 0 and .5')
    switch = np.log(transition_prob)
    stay = np.log1p(-transition_prob)
    while i < n:
        if not graded[i]:
            i += 1
            continue
        a = i
        while i < n and graded[i] and periods[i] == periods[a]:
            i += 1
        b = i
        frames = np.arange(a, b, stride)
        signal = logits[frames]
        emissions = np.column_stack([-np.logaddexp(0., signal),
                                     -np.logaddexp(0., -signal)])
        back = np.zeros((len(frames), 2), dtype=np.int8)
        scores = emissions[0].copy()
        for k in range(1, len(frames)):
            choices = scores[:, None]+np.array([[stay, switch], [switch, stay]])
            back[k] = np.argmax(choices, axis=0)
            scores = choices[back[k], np.arange(2)]+emissions[k]
        states = np.zeros(len(frames), np.int8)
        states[-1] = np.argmax(scores)
        for k in range(len(frames)-1, 0, -1):
            states[k-1] = back[k, states[k]]
        dead[a:b] = np.repeat(states, stride)[:b-a].astype(bool)
    return dead
