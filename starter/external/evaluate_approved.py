"""Compare frozen approved models on external matches, using NumPy inference.

No challenge reference is accepted. Full-timeline block dropout includes D,
unlike the older live-window-only Metrica dropout suite.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one,
                           write_csv, aggregate_by_candidate, per_game_means)


def version(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='2.3')
    parser.add_argument('--reference', default='2.2')
    parser.add_argument('--matches', nargs='+', default=[
        'J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9', 'Sample_Game_2'])
    parser.add_argument('--out', default='evaluation/reports/approved_full_timeline.csv')
    args = parser.parse_args()
    rows = []
    for game in args.matches:
        f, gt, graded = dataset(game)
        for sec in (0, 2, 5, 10, 30):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec)
            dropped = apply_dropout(f, hide)
            for name, ver in [('candidate', args.candidate), ('reference', args.reference)]:
                pred = run_labeller(dropped, graded, kwargs_for_version(*version(ver)))
                rows.append(dict(game=game, dropout=str(sec), candidate=name,
                                 **score_one(gt, pred, hide | ~f['ball_present'])))
            print(game, sec, flush=True)
    write_csv(args.out, rows)
    print(json.dumps(dict(aggregate=aggregate_by_candidate(rows), per_game=per_game_means(rows)), indent=2))


if __name__ == '__main__':
    main()
