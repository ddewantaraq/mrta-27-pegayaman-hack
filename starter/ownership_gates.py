"""Geometry-only gates for conservative ownership replacement experiments."""
import numpy as np


def endpoint_relation(features, reference, graded):
    n = len(reference)
    period = np.asarray(features.get('period', np.zeros(n, np.int8)))
    zone = np.asarray(features['pz_team'])
    anchor = features['ball_present'] & (zone >= 0) & graded & np.isin(reference, ['H', 'A'])
    relation = np.zeros(n, np.int8)  # 0 unknown, 1 same, -1 opposing control endpoints
    a = 0
    while a < n:
        if not graded[a]:
            a += 1
            continue
        b = a+1
        while b < n and graded[b] and period[b] == period[a]:
            b += 1
        idx = np.arange(a, b)
        previous = np.maximum.accumulate(np.where(anchor[a:b], idx, a-1))
        following = np.minimum.accumulate(np.where(anchor[a:b], idx, b)[::-1])[::-1]
        valid = (previous >= a) & (following < b)
        pp, ff = np.clip(previous, a, b-1), np.clip(following, a, b-1)
        relation[a:b] = np.where(valid, np.where(zone[pp] == zone[ff], 1, -1), 0)
        a = b
    return relation


def replacement_mask(features, reference, graded, logits, policy):
    missing = ~features['ball_present']
    live = graded & np.isin(reference, ['H', 'A'])
    if policy in ('point90_min25', 'run90_mean75', 'run90_mean90'):
        proposed = np.where(logits >= 0, 'H', 'A')
        changed = live & missing & (proposed != reference)
        if policy == 'point90_min25':
            changed &= np.abs(logits) >= np.log(9.)
        result = np.zeros(len(reference), bool)
        period = features.get('period', np.zeros(len(reference), np.int8))
        a = 0
        while a < len(reference):
            if not changed[a]:
                a += 1
                continue
            b = a+1
            while (b < len(reference) and changed[b] and proposed[b] == proposed[a]
                   and period[b] == period[a]):
                b += 1
            confidence = np.abs(logits[a:b])
            accept = b-a >= 25
            if policy.startswith('run90_mean'):
                minimum = np.log(9.) if policy == 'run90_mean90' else np.log(3.)
                accept &= confidence.max() >= np.log(9.) and confidence.mean() >= minimum
            if accept:
                result[a:b] = True
            a = b
        return result
    if policy == 'missing_p90':
        return live & missing & (np.abs(logits) >= np.log(9.))
    if policy == 'missing_p95':
        return live & missing & (np.abs(logits) >= np.log(19.))
    relation = endpoint_relation(features, reference, graded)
    if policy == 'opposing':
        return live & missing & (relation == -1)
    if policy == 'opposing_p75':
        return live & missing & (relation == -1) & (np.abs(logits) >= np.log(3.))
    if policy == 'protect_same_p75':
        return live & missing & (relation != 1) & (np.abs(logits) >= np.log(3.))
    if policy == 'uncontrolled_p75':
        return live & (features['pz_team'] < 0) & (relation != 1) & (np.abs(logits) >= np.log(3.))
    raise ValueError('Unknown ownership gate: '+policy)
