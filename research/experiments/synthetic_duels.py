"""Rank 6: synthetic contested-duel snippets with control-outcome labels.

Physics snippets (60f): a carrier dribbles, a challenger tackles at f30.
Outcome label = team that KILLS ball speed and holds it (dwell/control),
not last touch -- deflection outcomes keep the carrier's team. Snippets
are concatenated into one synthetic timeline; touch features come from
touch_design at snippet centres, reference = carrier throughout (the
missed-switch situation). Training-only augmentation for the touch HGB.
"""
import numpy as np


def make_snippets(n=1500, seed=7, length=60, touch_at=30):
    rng = np.random.default_rng(seed)
    total = n * length
    ball_x = np.zeros(total); ball_y = np.zeros(total)
    nearest_team = np.full(total, -1.); nearest_dist = np.full(total, 99.)
    min_h = np.full(total, 99.); min_a = np.full(total, 99.)
    near_h = np.zeros(total); near_a = np.zeros(total)
    ref = np.empty(total, dtype='<U1')
    y = np.zeros(n)
    idx = np.zeros(n, dtype=np.int64)
    for s in range(n):
        a = s * length
        carrier = rng.integers(0, 2)  # 0 home, 1 away
        switch = rng.random() < .5
        winner = 1 - carrier if switch else carrier
        bx, by = rng.uniform(20, 85), rng.uniform(15, 53)
        vx, vy = rng.uniform(-3, 3), rng.uniform(-2, 2)
        # ball path: dribble, deflection kick at touch, then winner control
        xs, ys = [bx], [by]
        cvx, cvy = vx, vy
        for t in range(1, length):
            if t == touch_at:
                ang = rng.uniform(0, 2 * np.pi)
                cvx, cvy = np.cos(ang) * rng.uniform(6, 10), np.sin(ang) * rng.uniform(6, 10)
            elif t > touch_at + 6:
                cvx, cvy = cvx * .8, cvy * .8  # winner kills speed
            xs.append(xs[-1] + cvx / 25.); ys.append(ys[-1] + cvy / 25.)
        xs, ys = np.array(xs), np.array(ys)
        ball_x[a:a + length], ball_y[a:a + length] = xs, ys
        # players: carrier + challenger hug the ball, winner closest after
        for t in range(length):
            off_c = np.array([rng.normal(0, .4), rng.normal(0, .4)])
            off_w = np.array([rng.normal(0, .5), rng.normal(0, .5)])
            pos_c = np.array([xs[t], ys[t]]) + off_c
            pos_w = np.array([xs[t], ys[t]]) + off_w + np.array([1.2, 0.])
            dc = np.hypot(*(pos_c - [xs[t], ys[t]]))
            dw = np.hypot(*(pos_w - [xs[t], ys[t]]))
            dists = {carrier: dc, winner: dw}
            if carrier == winner:
                dists = {carrier: min(dc, dw)}
            dh = dists.get(0, 99.); da = dists.get(1, 99.)
            min_h[a + t], min_a[a + t] = dh, da
            near_h[a + t] = 1. if dh <= 3. else 0.
            near_a[a + t] = 1. if da <= 3. else 0.
            if dh < da:
                nearest_team[a + t], nearest_dist[a + t] = 0., dh
            else:
                nearest_team[a + t], nearest_dist[a + t] = 1., da
        ref[a:a + length] = 'H' if carrier == 0 else 'A'
        y[s] = 1. if winner == 0 else 0.
        idx[s] = a + touch_at
    features = dict(ball_x_m=ball_x, ball_y_m=ball_y,
                    ball_present=np.ones(total, bool),
                    period=np.ones(total), nearest_team=nearest_team,
                    nearest_dist=nearest_dist, min_dist_home=min_h,
                    min_dist_away=min_a, n_near_home=near_h, n_near_away=near_a)
    logits = np.where(ref == 'H', 1., -1.)
    return features, ref, logits, idx, y


def synthetic_design(n=1500, seed=7):
    from event_first_model import touch_candidates, touch_design
    features, ref, logits, idx, y = make_snippets(n=n, seed=seed)
    graded = np.ones(len(ref), bool)
    kin = touch_candidates(features, graded)[1]
    x = touch_design(features, ref, logits, idx, kin)
    return x, y
