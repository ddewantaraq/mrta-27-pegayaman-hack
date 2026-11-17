"""Step b proxy: nearest-team agreement from degraded vs refined ball.

At 5Hz node frames in contested windows, nearest player team (from ball
position) vs GT possession, for raw degraded and refined ball. Kill if
refined does not improve agreement. Writes ball_refine_nearest.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball_refine_sim import contested_mask, degrade, refine
from crossfit_common import REPORTS, load_match
from event_first import FIT
from gnn_probe import LENGTH, WIDTH, match_arrays


def nearest_sign(players, bx_m, by_m):
    px = players[:, :, 0] * LENGTH
    py = players[:, :, 1] * WIDTH
    exists = players[:, :, 8] > 0
    d = np.hypot(px - bx_m[:, None], py - by_m[:, None])
    d = np.where(exists, d, np.inf)
    near_home = d[:, :11].min(axis=1) < d[:, 11:].min(axis=1)
    return np.where(near_home, 1., -1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'ball_refine_nearest.json'))
    args = parser.parse_args()
    rows = []
    for game in FIT:
        d = load_match(game)
        f, gt = d['features'], d['gt']
        arr = match_arrays(d)
        frames = arr['frames']
        period = np.asarray(f.get('period', np.ones(f['n_frames'])), np.int64)
        x, y, noisy, _ = degrade(f, gt, 0.3)
        rx, ry = refine(x, y, period)
        contested25 = contested_mask(gt)[frames]
        mask = contested25 & arr['visible']
        gt_sign = arr['y']
        for name, bx, by in (('degraded', x, y), ('refined', rx, ry)):
            bx_f = np.where(np.isfinite(bx[frames]), bx[frames], LENGTH / 2.)
            by_f = np.where(np.isfinite(by[frames]), by[frames], WIDTH / 2.)
            sign = nearest_sign(arr['players'], bx_f, by_f)
            acc = float(np.mean(sign[mask] == gt_sign[mask]))
            rows.append(dict(game=game, ball=name, nearest_agree=acc))
            print(game, name, round(acc, 4), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')
    for name in ('degraded', 'refined'):
        vals = [r['nearest_agree'] for r in rows if r['ball'] == name]
        print(name, 'mean nearest agreement', round(float(np.mean(vals)), 4))


if __name__ == '__main__':
    main()
