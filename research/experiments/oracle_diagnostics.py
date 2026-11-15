"""EXTERNAL ONLY diagnostic upper bounds; never generates a challenge submission.

Uses external reference labels deliberately to isolate team versus dead-ball
errors. Oracle scores are diagnostics, not model results or deployable scores.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sportec_experiment import cached_sportec, block_dropout
from eval_external import kwargs_for_version, run_labeller, score_one
from dropout import apply_dropout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WN1')
    args = parser.parse_args()
    f, gt, graded = cached_sportec(args.match)
    rows = []
    for sec in (0, 5):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec)
        dropped = apply_dropout(f, hide)
        pred = run_labeller(dropped, graded, kwargs_for_version(1, 9))
        kw = kwargs_for_version(1, 8)
        kw['use_geometric_dead'] = False
        team = run_labeller(dropped, graded, kw)
        perfect_dead = team.copy()
        perfect_dead[gt == 'D'] = 'D'
        perfect_team = pred.copy()
        both_alive = np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
        perfect_team[both_alive] = gt[both_alive]
        for name, labels in [('actual', pred), ('ORACLE_dead_only', perfect_dead),
                             ('ORACLE_team_only', perfect_team)]:
            rows.append(dict(game=args.match, dropout=str(sec), candidate=name,
                             **score_one(gt, labels, hide)))
        print(args.match, sec, flush=True)
    path = Path('evaluation/reports/oracle_diagnostics_'+args.match+'.json')
    path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
