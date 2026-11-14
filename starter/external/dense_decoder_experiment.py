"""Test sub-5-Hz-bin ownership timing without changing trained parameters."""
import argparse
import json
import numpy as np
from approved_predictions import approved_cases
from temporal_decode import decode_scaled
from state_machine import _merge_short_spells
from eval_external import score_one,write_csv,aggregate_by_candidate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='J03WMX'); args=parser.parse_args(); rows=[]
    for sec,f,gt,g,hide,reference,approved,logits in approved_cases(args.match):
        live=g & np.isin(reference,['H','A']); predictions={'approved':approved}
        for name,stride,scale in [('dense1',1,.2),('dense2',2,.4),('dense1_unscaled',1,1.)]:
            home=decode_scaled(logits,live,f.get('period'),stride,scale)
            home[f['ball_present']]=logits[f['ball_present']]>=0
            pred=reference.copy(); pred[live]=np.where(home[live],'H','A')
            predictions[name]=_merge_short_spells(pred,g,18)
        for name,pred in predictions.items():
            row=dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec=='0':
                print('natural dense decoder',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('dense decoder',args.match,sec,flush=True)
    write_csv('evaluation/reports/dense_decoder_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__=='__main__':
    main()
