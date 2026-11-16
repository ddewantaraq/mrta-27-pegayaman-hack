"""Apply prior team learner only where the ball is absent; retain visible control.

Never changes reference D/X. Compare the unbalanced and spell-balanced
learners without repeating the already rejected full-frame replacement.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from frozen_predictions import cases
from team_control_model import predict_teams
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WMX')
    args = parser.parse_args()
    models = {}
    for name in ('team_control_v1', 'team_control_balanced_v1'):
        with np.load('evaluation/reports/'+name+'.npz') as archive:
            models[name] = dict(archive)
    rows = []
    for drop, f, gt, graded, hide, reference in cases(args.match):
        predictions = {'reference': reference}
        for name, model in models.items():
            team = predict_teams(f, model, graded)
            pred = reference.copy()
            mask = graded & ~f['ball_present'] & np.isin(reference, ['H', 'A'])
            pred[mask] = team[mask]
            predictions[name] = _merge_short_spells(pred, graded, 18)
        for name, pred in predictions.items():
            rows.append(dict(game=args.match, dropout=drop, candidate=name,
                             **score_one(gt, pred, hide)))
        print(args.match, drop, flush=True)
    write_csv('evaluation/reports/missing_team_'+args.match+'.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
