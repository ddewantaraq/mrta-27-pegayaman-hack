"""Research-only ownership smoothing; select on WMX/Metrica2, not reserved M3."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from frozen_predictions import cases
from player_node_data import load_player_nodes
from set_temporal_team_model import load_set_temporal_model, predict_set_logits
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match', default='J03WMX')
    args = parser.parse_args()
    if args.match == 'Sample_Game_3':
        raise ValueError('Reserved match is excluded from decoder selection')
    torch.set_num_threads(2)
    path = Path('evaluation/reports/set_temporal_v1.pt')
    signature = hashlib.sha256(path.read_bytes()).hexdigest()
    model = load_set_temporal_model(path)
    nodes = load_player_nodes(args.match)
    rows = []
    for sec,f,gt,graded,hide,reference in cases(args.match):
        cache = Path('evaluation/reports/prediction_cache')/('set_logits_'+args.match+'_'+sec+'s.npz')
        logits = None
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                if str(data['signature']) == signature:
                    logits = data['logits'].copy()
        if logits is None:
            logits = predict_set_logits(f, graded, nodes, model)
            np.savez_compressed(cache, logits=logits, signature=signature)
        live = graded & np.isin(reference, ['H', 'A'])
        predictions = {'reference': reference}
        for name, prob in [('raw',None),('viterbi_05',.05),('viterbi_01',.01)]:
            home = logits >= 0 if prob is None else decode_dead_logits(
                logits, live, f.get('period'), transition_prob=prob)
            pred = reference.copy()
            pred[live] = np.where(home[live], 'H', 'A')
            predictions[name] = _merge_short_spells(pred, graded, 18)
        for name,pred in predictions.items():
            rows.append(dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)))
        print('decoder experiment',args.match,sec,flush=True)
    write_csv('evaluation/reports/set_decoder_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
