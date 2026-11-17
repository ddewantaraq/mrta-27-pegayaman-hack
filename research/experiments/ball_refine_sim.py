"""New-direction step a-sim: can trajectory refinement recover ATD ball?

GT ball (Sportec/Metrica caches) is degraded to an ATD proxy: block
dropout (41%) + Gaussian jitter (contested peaks worse). Refiner =
period-wise linear gap fill + Savitzky-Golay(7,2). Score xy error vs GT
on contested windows (+-25f of GT switches) across a jitter sweep.
Kill (workflow step a/b): contested RMSE improvement <20% relative.
Writes evaluation/reports/ball_refine_sim.json. No challenge data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first import FIT
from sportec_experiment import block_dropout

SIGMAS = (0.15, 0.3, 0.6)


def switches(gt):
    gt = np.asarray(gt)
    live = np.isin(gt, ['H', 'A'])
    return np.flatnonzero(live[:-1] & live[1:] & (gt[:-1] != gt[1:])) + 1


def contested_mask(gt, window=25):
    mask = np.zeros(len(gt), bool)
    for s in switches(gt):
        mask[max(0, s - window):s + window + 1] = True
    return mask


def degrade(f, gt, sigma, seed=41):
    rng = np.random.default_rng(seed)
    n = f['n_frames']
    present = np.asarray(f['ball_present'], bool)
    drop = block_dropout(n, 5, seed=seed) & present
    contested = contested_mask(gt)
    jitter = np.where(contested, sigma, sigma / 3.)
    x = np.asarray(f['ball_x_m'], np.float64).copy()
    y = np.asarray(f['ball_y_m'], np.float64).copy()
    noisy = present & ~drop
    x[noisy] += rng.normal(0, 1, noisy.sum()) * jitter[noisy]
    y[noisy] += rng.normal(0, 1, noisy.sum()) * jitter[noisy]
    x[~noisy] = np.nan
    y[~noisy] = np.nan
    return x, y, noisy, contested


def refine(x, y, period, max_gap=25):
    out_x, out_y = x.copy(), y.copy()
    for p in np.unique(period):
        sel = np.flatnonzero(period == p)
        for src, dst in ((x, out_x), (y, out_y)):
            v = src[sel]
            ok = np.isfinite(v)
            if ok.sum() < 8:
                continue
            filled = np.interp(np.arange(len(sel)), np.flatnonzero(ok), v[ok])
            # only fill missing runs of length <= max_gap
            keep = ok.copy()
            idx_ok = np.flatnonzero(ok)
            for a, b in zip(idx_ok[:-1], idx_ok[1:]):
                if b - a - 1 <= max_gap:
                    keep[a:b + 1] = True
            smooth = savgol_filter(filled, 7, 2)
            filled = np.where(keep, filled, np.nan)
            smooth = np.where(keep, smooth, np.nan)
            dst[sel] = np.where(ok, .5 * v + .5 * smooth, filled)
    return out_x, out_y


def rmse(a, b, mask):
    d = (a - b)[mask]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'ball_refine_sim.json'))
    args = parser.parse_args()
    rows = []
    for game in FIT:
        d = load_match(game)
        f, gt = d['features'], d['gt']
        gx = np.asarray(f['ball_x_m'], np.float64)
        gy = np.asarray(f['ball_y_m'], np.float64)
        period = np.asarray(f.get('period', np.ones(f['n_frames'])), np.int64)
        for sigma in SIGMAS:
            x, y, noisy, contested = degrade(f, gt, sigma)
            rx, ry = refine(x, y, period)
            eval_mask = contested & np.asarray(f['ball_present'], bool)
            raw = np.sqrt(rmse(x, gx, eval_mask) ** 2 + rmse(y, gy, eval_mask) ** 2)
            ref = np.sqrt(rmse(rx, gx, eval_mask) ** 2 + rmse(ry, gy, eval_mask) ** 2)
            rows.append(dict(game=game, sigma=sigma,
                             contested_rmse_raw=raw, contested_rmse_refined=ref,
                             rel_improvement=float(1 - ref / raw) if raw else None,
                             coverage_gain=float(np.isfinite(rx[eval_mask]).mean()
                                                 - noisy[eval_mask].mean())))
            print(game, 'sigma', sigma, 'raw', round(raw, 3), 'refined', round(ref, 3),
                  'rel', round(1 - ref / raw, 3), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')
    for sigma in SIGMAS:
        vals = [r['rel_improvement'] for r in rows if r['sigma'] == sigma]
        print('sigma', sigma, 'mean rel improvement', round(float(np.mean(vals)), 3),
              '(kill threshold 0.20)')


if __name__ == '__main__':
    main()
