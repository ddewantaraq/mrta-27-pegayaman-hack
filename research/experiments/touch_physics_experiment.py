"""Ranks 1+2 vs v2.7 event-first: physics intercept features / dwell target.

Rank 1: Spearman-style intercept race per team (arrival at predicted ball
position at vmax, from 5Hz player nodes) added to the 22 touch features.
Rank 2: touch label = majority GT team over next 25f instead of lead+3.
Same candidates, HGB config and decoder grid as event_first.py; FIT LOMO
selection only. Writes evaluation/reports/touch_physics_crossfit.json.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first import FIT, choose, fit_model, game_cache, score_holdout

VMAX = 7.0
T_CAP = 2.0


def intercept_features(features, nodes, idx):
    """8 physics cols at touch candidates: per-team intercept race."""
    nodes = np.asarray(nodes, np.float32)
    n = len(features['ball_present'])
    frame5 = np.clip(np.asarray(idx) // 5, 0, len(nodes) - 1)
    bx = np.asarray(features['ball_x_m'], np.float64)
    by = np.asarray(features['ball_y_m'], np.float64)
    length, width = (float(v) for v in features.get('pitch', (105.3, 68.)))
    t = np.clip(np.asarray(idx), 2, n - 3)
    vbx = (bx[t + 2] - bx[t - 2]) * 25. / 4.
    vby = (by[t + 2] - by[t - 2]) * 25. / 4.
    cols = []
    team_feats = []
    for team in (0, 1):
        block = nodes[frame5, team]  # (k,11,8)
        sign = np.where(block[:, 0, 7] >= 0, 1., -1.)
        px = block[:, :, 0] * length
        py = block[:, :, 1] * width
        pvx = block[:, :, 2] * 10.
        pvy = block[:, :, 3] * 10.
        exists = block[:, :, 6] > 0
        ball_x = np.where(sign > 0, bx[t], length - bx[t])
        rel_x = px - ball_x[:, None]
        rel_y = py - by[t][:, None]
        dist = np.hypot(rel_x, rel_y)
        dist = np.where(exists, dist, np.inf)
        # intercept: |rel + vb*t| = VMAX*t, smallest positive root
        bvx = vbx * sign
        a = (bvx ** 2 + vby ** 2)[:, None] - VMAX ** 2
        b = 2. * (rel_x * bvx[:, None] + rel_y * vby[:, None])
        c = dist ** 2
        disc = b ** 2 - 4. * a * c
        root = np.where((disc >= 0) & (a < -1e-9),
                        (-b - np.sqrt(np.maximum(disc, 0.))) / (2. * a), np.inf)
        root = np.where(root > 0, root, np.inf)
        t_min = np.minimum(root.min(axis=1), T_CAP)
        near = np.argmin(dist, axis=1)
        rows = np.arange(len(idx))
        t_near = np.minimum(root[rows, near], T_CAP)
        # closing speed of nearest player (positive = closing)
        rvx = pvx[rows, near] - bvx
        rvy = pvy[rows, near] - vby
        denom = np.maximum(dist[rows, near], 1e-6)
        closing = -(rel_x[rows, near] * rvx + rel_y[rows, near] * rvy) / denom
        team_feats.append((t_min, t_near, closing))
    (th_min, th_near, ch), (ta_min, ta_near, ca) = team_feats
    cols = np.column_stack([th_min, ta_min, th_near, ta_near, ch, ca,
                            th_min - ta_min, ch - ca]).astype(np.float64)
    cols[~np.isfinite(cols)] = T_CAP
    return cols


def dwell_labels(gt, idx, window=25):
    gt = np.asarray(gt)
    out = []
    for i in idx:
        seg = gt[int(i) + 1:int(i) + 1 + window]
        live = seg[np.isin(seg, ['H', 'A'])]
        if len(live) == 0:
            out.append('X')
        else:
            out.append('H' if (live == 'H').mean() >= .5 else 'A')
    return np.array(out)


def build(game, mode):
    d = game_cache(game)
    if mode in ('intercept', 'both'):
        nodes = load_match(game)['nodes']
        extra = intercept_features(d['f'], nodes, d['idx'])
        d = dict(d, x=np.hstack([d['x'], extra]))
    if mode in ('dwell', 'both'):
        labels = dwell_labels(d['gt'], d['idx'])
        keep = np.isin(labels, ['H', 'A'])
        d = dict(d, y=(labels == 'H').astype(float), keep=keep)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('intercept', 'dwell', 'both'),
                        required=True)
    parser.add_argument('--with-sc', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    items = [build(g, args.mode) for g in FIT]
    sc = []
    if args.with_sc:
        from event_first import skillcorner_train
        sc = [build(g, args.mode) for g in skillcorner_train()]
    from event_first import lomo
    rows = lomo(items, sc, sc_weight=1.0 if args.with_sc else 0.0)
    best, table = choose(rows)
    print(json.dumps(dict(mode=args.mode, best=best,
                          best_row=table.get(best)), indent=2))
    out = args.out or str(REPORTS / 'touch_physics_{}_crossfit.json'.format(args.mode))
    Path(out).write_text(json.dumps(dict(mode=args.mode, fit=FIT, best=best,
                                         table=table), indent=2) + '\n')


if __name__ == '__main__':
    main()
