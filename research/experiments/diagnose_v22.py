"""EXTERNAL ONLY: isolate remaining team and live/dead possession errors.

Counterfactual reference-assisted scores below are diagnostics, not achieved
model scores, deployable models, or challenge predictions.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from approved_predictions import approved_cases
from dead_ball_model import decode_dead_logits
from eval_external import score_one


def confusion(gt,pred,mask):
    live = mask & np.isin(gt,['H','A'])
    dead = mask & (gt == 'D')
    true_dead = int((dead & (pred == 'D')).sum())
    false_dead = int((live & (pred == 'D')).sum())
    missed_dead = int((dead & np.isin(pred,['H','A'])).sum())
    return dict(live_frames=int(live.sum()),correct_team=int((live & (pred == gt)).sum()),
                wrong_team=int((live & np.isin(pred,['H','A']) & (pred != gt)).sum()),
                false_dead=false_dead,abstain=int((live & (pred == 'X')).sum()),
                dead_frames=int(dead.sum()),missed_dead=missed_dead,
                dead_precision=true_dead/max(1,true_dead+false_dead),
                dead_recall=true_dead/max(1,int(dead.sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='Sample_Game_2')
    args = parser.parse_args()
    allowed = {'Sample_Game_1','Sample_Game_2','Sample_Game_3','J03WOH','J03WN1','J03WOY','J03WMX','J03WPY','J03WQQ','J03WR9'}
    if args.match not in allowed:
        raise ValueError('External reference diagnostics only')
    rows = []
    for sec,f,gt,g,hide,old,pred,logits in approved_cases(args.match):
        if sec not in ('0','5'):
            continue
        valid = g & (gt != 'X')
        perfect_team = pred.copy()
        existing_live = np.isin(gt,['H','A']) & np.isin(pred,['H','A'])
        perfect_team[existing_live] = gt[existing_live]
        home = decode_dead_logits(logits,g,f.get('period'))
        home[f['ball_present']] = logits[f['ball_present']] >= 0
        perfect_dead = pred.copy()
        restore = valid & np.isin(gt,['H','A']) & ~np.isin(pred,['H','A'])
        perfect_dead[restore] = np.where(home[restore],'H','A')
        perfect_dead[valid & (gt == 'D')] = 'D'
        scores = {name:score_one(gt,labels,hide) for name,labels in
                  [('actual',pred),('REFERENCE_assisted_team_only',perfect_team),
                   ('REFERENCE_assisted_live_dead_latent_teams',perfect_dead)]}
        row = dict(game=args.match,dropout=sec,scores=scores,
                   all_frames=confusion(gt,pred,valid),
                   missing_frames=confusion(gt,pred,valid & ~f['ball_present']),
                   previous_missing=confusion(gt,old,valid & ~f['ball_present']))
        rows.append(row); print(json.dumps(row,indent=2),flush=True)
    Path('evaluation/reports/diagnose_v22_'+args.match+'.json').write_text(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
