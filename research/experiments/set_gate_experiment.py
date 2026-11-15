"""Compare conservative ownership gates on external development matches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from frozen_predictions import cases
from ownership_gates import replacement_mask
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WMX')
    parser.add_argument('--policies', nargs='+', default=['missing_p90','missing_p95','opposing',
                        'opposing_p75','protect_same_p75','uncontrolled_p75','point90_min25','run90_mean75'])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.match == 'Sample_Game_3':
        raise ValueError('Do not select gates on the reserved match')
    rows = []
    model = None
    signature = hashlib.sha256(Path('evaluation/reports/set_temporal_v1.pt').read_bytes()).hexdigest()
    for sec,f,gt,graded,hide,reference in cases(args.match):
        path = Path('evaluation/reports/prediction_cache')/('set_logits_'+args.match+'_'+sec+'s.npz')
        logits = None
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                if str(data['signature']) == signature:
                    logits = data['logits'].copy()
        if logits is None:
            import torch
            from set_temporal_team_model import load_set_temporal_model, predict_set_logits
            from player_node_data import load_player_nodes
            torch.set_num_threads(2)
            if model is None:
                model = load_set_temporal_model('evaluation/reports/set_temporal_v1.pt')
                nodes = load_player_nodes(args.match)
            logits = predict_set_logits(f, graded, nodes, model)
            np.savez_compressed(path, logits=logits, signature=signature)
        predictions = {'reference': reference}
        for policy in args.policies:
            mask = replacement_mask(f, reference, graded, logits, policy)
            pred = reference.copy()
            pred[mask] = np.where(logits[mask] >= 0, 'H', 'A')
            predictions[policy] = _merge_short_spells(pred, graded, 18)
        for name,pred in predictions.items():
            row = dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide))
            rows.append(row)
            if sec == '0':
                print('natural',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('gate evaluation',args.match,sec,flush=True)
    write_csv(args.out or 'evaluation/reports/set_gate_'+args.match+'.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
