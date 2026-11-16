"""Test whether suppressing short spells actually improves external spellF1."""
import argparse
import json
from approved_predictions import approved_cases
from state_machine import _merge_short_spells
from short_spell_experiment import suppress_live_islands
from eval_external import score_one,write_csv,aggregate_by_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='J03WMX'); args=parser.parse_args()
    rows=[]
    for sec,f,gt,g,hide,reference,approved,logits in approved_cases(args.match):
        predictions={'approved':approved,'merge25':_merge_short_spells(approved,g,25),
                     'merge50':_merge_short_spells(approved,g,50),
                     'dead_islands25':suppress_live_islands(approved,g,25)}
        for name,pred in predictions.items():
            row=dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec=='0':
                print('natural short ablation',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('short ablation',args.match,sec,flush=True)
    write_csv('evaluation/reports/v22_short_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__=='__main__':
    main()
