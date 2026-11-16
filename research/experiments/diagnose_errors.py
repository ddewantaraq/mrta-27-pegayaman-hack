"""Per-match error breakdown for an approved / named kwargs candidate.

Splits visible vs missing ball: wrong team, false D, missed D.
No challenge ground truth. Diagnostics only.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import kwargs_for_version, run_labeller, smooth50_labels
from diagnose_v22 import confusion


DEFAULT_MATCHES = [
    'J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9', 'Sample_Game_2', 'Sample_Game_3']


def resolve_kwargs(spec):
    if spec == 'smooth50':
        return None
    if spec.startswith('{'):
        return ast.literal_eval(spec)
    major, minor = spec.split('.')
    return kwargs_for_version(int(major), int(minor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='2.3')
    parser.add_argument('--matches', nargs='+', default=DEFAULT_MATCHES)
    parser.add_argument('--dropouts', nargs='+', default=['0', '5'],
                        help='block seconds; 0 = natural')
    parser.add_argument('--out', default='evaluation/reports/diagnose_errors.json')
    args = parser.parse_args()
    kw = resolve_kwargs(args.candidate)
    rows = []
    for game in args.matches:
        try:
            f, gt, graded = dataset(game)
        except FileNotFoundError as exc:
            print('SKIP', game, exc, flush=True)
            continue
        for sec in args.dropouts:
            sec_i = int(sec)
            hide = np.zeros(f['n_frames'], bool) if sec_i == 0 else block_dropout(
                f['n_frames'], sec_i, seed=41)
            dropped = apply_dropout(f, hide)
            if kw is None:
                pred = smooth50_labels(dropped)
            else:
                pred = run_labeller(dropped, graded, kw)
            valid = graded & (gt != 'X')
            missing = valid & ~f['ball_present']
            visible = valid & f['ball_present']
            row = dict(
                game=game, dropout=str(sec_i), candidate=args.candidate,
                all_frames=confusion(gt, pred, valid),
                missing_frames=confusion(gt, pred, missing),
                visible_frames=confusion(gt, pred, visible),
            )
            miss = row['missing_frames']
            row['missing_team_vs_dead'] = dict(
                wrong_team=miss['wrong_team'],
                false_dead=miss['false_dead'],
                missed_dead=miss['missed_dead'],
                live_dead_errors=miss['false_dead'] + miss['missed_dead'],
                dead_dominates_team=(
                    miss['false_dead'] + miss['missed_dead'] > miss['wrong_team']),
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    # Aggregate: does D dominate wrong-team on missing frames for most matches?
    natural = [r for r in rows if r['dropout'] == '0']
    if natural:
        dominate = sum(1 for r in natural
                       if r['missing_team_vs_dead']['dead_dominates_team'])
        print('D_DOMINATES_TEAM natural_matches={}/{}'.format(
            dominate, len(natural)), flush=True)


if __name__ == '__main__':
    main()
