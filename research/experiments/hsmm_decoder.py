"""Hidden semi-Markov decoder over states H, A, D with duration priors.

Inputs are per-frame log-odds for home ownership and dead-ball. Duration
histograms are estimated from training-match ground truth inside each fold.
"""
import numpy as np


STATES = ('H', 'A', 'D')


def duration_prior(gt_list, graded_list, max_len=250, alpha=1.0):
    """Return log P(duration | state) tables, length 1..max_len."""
    counts = {s: np.full(max_len + 1, alpha, dtype=np.float64) for s in STATES}
    for gt, graded in zip(gt_list, graded_list):
        gt = np.asarray(gt)
        graded = np.asarray(graded, bool)
        i = 0
        n = len(gt)
        while i < n:
            if not graded[i] or gt[i] not in STATES:
                i += 1
                continue
            s = gt[i]
            j = i + 1
            while j < n and graded[j] and gt[j] == s:
                j += 1
            length = min(j - i, max_len)
            counts[s][length] += 1
            i = j
    logp = {}
    for s in STATES:
        total = counts[s].sum()
        logp[s] = np.log(counts[s] / total)
    return logp


def _emissions(team_logits, dead_logits):
    """Frame emissions as log-probs over H, A, D."""
    team = np.asarray(team_logits, dtype=np.float64)
    dead = np.asarray(dead_logits, dtype=np.float64)
    # P(dead), P(home|live), P(away|live)
    p_dead = 1 / (1 + np.exp(-np.clip(dead, -30, 30)))
    p_home = 1 / (1 + np.exp(-np.clip(team, -30, 30)))
    p_live = 1 - p_dead
    # Small floor
    eps = 1e-6
    e_h = np.log(np.clip(p_live * p_home, eps, 1))
    e_a = np.log(np.clip(p_live * (1 - p_home), eps, 1))
    e_d = np.log(np.clip(p_dead, eps, 1))
    return np.column_stack([e_h, e_a, e_d])


def decode_hsmm(team_logits, dead_logits, graded, period=None, duration_logp=None,
                max_len=100, stride=5):
    """Viterbi HSMM at `stride` Hz, then expand. States indexed H=0,A=1,D=2."""
    n = len(team_logits)
    graded = np.ones(n, bool) if graded is None else np.asarray(graded, bool)
    period = np.zeros(n, np.int8) if period is None else np.asarray(period)
    if duration_logp is None:
        # Weak geometric-like prior: prefer ~2s spells at 5Hz => 10 frames.
        duration_logp = {
            s: np.log(np.maximum(1e-12, np.exp(-np.arange(max_len + 1) / 12.)))
            for s in STATES}
        for s in STATES:
            duration_logp[s] -= np.logaddexp.reduce(duration_logp[s])
    emissions = _emissions(team_logits, dead_logits)
    labels = np.full(n, 'X', dtype='<U1')
    state_ids = {s: i for i, s in enumerate(STATES)}
    i = 0
    while i < n:
        if not graded[i]:
            i += 1
            continue
        a = i
        while i < n and graded[i] and period[i] == period[a]:
            i += 1
        b = i
        frames = np.arange(a, b, stride)
        if len(frames) == 0:
            continue
        em = emissions[frames]  # T x 3
        T = len(frames)
        # DP: best score ending a segment of state s at t
        # We use a compact segment Viterbi with max_len.
        neg = -1e30
        best = np.full((T, 3), neg)
        back_state = np.full((T, 3), -1, np.int16)
        back_len = np.full((T, 3), 1, np.int16)
        for t in range(T):
            for s in range(3):
                name = STATES[s]
                maxlen = min(max_len, t + 1)
                best_score = neg
                best_l = 1
                best_ps = -1
                for length in range(1, maxlen + 1):
                    t0 = t - length + 1
                    seg = em[t0:t + 1, s].sum() + duration_logp[name][length]
                    if t0 == 0:
                        score = seg
                        ps = -1
                    else:
                        # Transition from any other state (encourage switches).
                        prev = best[t0 - 1]
                        # Slight penalty for same-state re-entry (duration handles stickiness)
                        trans = prev.copy()
                        trans[s] -= 2.0
                        ps = int(np.argmax(trans))
                        score = seg + trans[ps]
                    if score > best_score:
                        best_score = score
                        best_l = length
                        best_ps = ps
                best[t, s] = best_score
                back_len[t, s] = best_l
                back_state[t, s] = best_ps
        # Backtrack
        t = T - 1
        s = int(np.argmax(best[t]))
        path = np.zeros(T, np.int8)
        while t >= 0:
            length = int(back_len[t, s])
            path[t - length + 1:t + 1] = s
            ps = int(back_state[t, s])
            t = t - length
            s = ps if t >= 0 else s
        expanded = np.repeat(path, stride)[:b - a]
        labels[a:b] = np.array(STATES)[expanded]
    return labels


def fit_duration_from_games(load_match_fn, games, max_len=100):
    gts, graded = [], []
    for game in games:
        data = load_match_fn(game)
        gts.append(data['gt'])
        graded.append(data['graded'])
    return duration_prior(gts, graded, max_len=max_len)
