"""Separate team-control errors from live/dead errors; direct external training."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sportec_experiment import cached_sportec, block_dropout
from dropout import apply_dropout
from eval_external import kwargs_for_version, run_labeller, score_one, write_csv, aggregate_by_candidate
from dead_ball_model import fit_logistic
from team_control_model import control_design, predict_teams

MODEL = Path('evaluation/reports/team_control_v1.npz')


def train(spell_balanced=False):
    xs, ys, weights = [], [], []
    for game in ('J03WOH', 'J03WN1', 'J03WOY'):
        f, gt, graded = cached_sportec(game)
        for sec in (0, 5, 15):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec, seed=73)
            x = control_design(apply_dropout(f, hide))
            if spell_balanced:
                samples, mass = [], []
                start = 0
                while start < len(gt):
                    end = start+1
                    while end < len(gt) and gt[end] == gt[start] and graded[end] == graded[start]:
                        end += 1
                    if graded[start] and gt[start] in ('H', 'A'):
                        points = np.arange(start, end, 5)
                        samples.extend(points)
                        mass.extend([1./len(points)]*len(points))
                    start = end
                take = np.asarray(samples, dtype=int)
                weights.append(np.asarray(mass))
            else:
                take = np.flatnonzero(graded & np.isin(gt, ['H', 'A']))[::5]
            xs.append(x[take])
            ys.append((gt[take] == 'H').astype(float))
        print('team training', game, flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    # Swapping team names negates every feature and complements ownership.
    weight = np.tile(np.concatenate(weights), 2) if spell_balanced else None
    model = fit_logistic(np.concatenate([x, -x]), np.concatenate([y, 1-y]), sample_weights=weight)
    path = Path(str(MODEL).replace('_v1.npz', '_balanced_v1.npz')) if spell_balanced else MODEL
    np.savez(path, **model)
    print('fit team model', 2*len(y), flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WMX')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--spell-balanced', action='store_true')
    args = parser.parse_args()
    path = Path(str(MODEL).replace('_v1.npz', '_balanced_v1.npz')) if args.spell_balanced else MODEL
    if args.train or not path.exists():
        model = train(args.spell_balanced)
    else:
        with np.load(path) as archive:
            model = dict(archive)
    f, gt, graded = cached_sportec(args.match)
    rows = []
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec)
        dropped = apply_dropout(f, hide)
        latest = run_labeller(dropped, graded, kwargs_for_version(1, 9))
        teams = predict_teams(dropped, model, graded)
        pred = latest.copy()
        live = graded & (pred != 'D')
        pred[live] = teams[live]
        for name, labels in [('latest', latest), ('team_model', pred)]:
            rows.append(dict(game=args.match, dropout=str(sec), candidate=name,
                             **score_one(gt, labels, hide)))
        print(args.match, sec, flush=True)
    suffix = '_balanced' if args.spell_balanced else ''
    write_csv('evaluation/reports/team_control_'+args.match+suffix+'.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
