"""Evaluate official pretrained PathCRF emissions without reference boundaries.

This first adapter tests macro/source-marginal emissions, not full CRF event
decoding. No target-match annotations are used. Sportec training matches must
not be reported as independent validation for this pretrained checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

from frozen_predictions import cases
from pathcrf_geometry import prepare_geometry, phases
from eval_external import score_one,write_csv,aggregate_by_candidate
from state_machine import _merge_short_spells

ROOT = Path(__file__).resolve().parents[2]/'data_external'/'pathcrf'


def infer(game):
    sys.path.insert(0,str(ROOT))
    from models.utils import build_model
    args = json.loads((ROOT/'saved/140/args.json').read_text())
    model_path = ROOT/'saved/140/model/state_dict_best.pt'
    signature = hashlib.sha256(model_path.read_bytes()).hexdigest()
    cache = Path('evaluation/reports/prediction_cache')/('pathcrf_emissions_'+game+'.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            if str(data['signature']) == signature:
                return data['macro'].copy(),data['micro'].copy(),data['covered'].copy()
    model = build_model(args,torch.device('cpu'))
    state = torch.load(model_path,map_location='cpu',weights_only=True)
    model.load_state_dict(state); model.eval()
    data = prepare_geometry(game)
    n = int(data['n_frames'])
    macro, micro, covered = np.zeros(n),np.zeros(n),np.zeros(n,bool)
    valid_edges = model.valid_orig_edge_ids
    src = model.comp_src.numpy(); dst = model.comp_dst.numpy()
    for phase,(a,b,inputs,available,home_nodes) in enumerate(phases(data)):
        length = len(inputs)
        starts = list(range(0,max(1,length-49),25))
        if starts[-1] < max(0,length-50):
            starts.append(length-50)
        acc_macro,acc_micro,weights = np.zeros(length),np.zeros(length),np.zeros(length)
        edge_available = available[src] & available[dst]
        hs = torch.from_numpy(home_nodes[src] & edge_available)
        aw = torch.from_numpy((src < 22) & ~home_nodes[src] & edge_available)
        home = torch.from_numpy(home_nodes)
        away = torch.from_numpy((np.arange(26) < 22) & ~home_nodes & available)
        with torch.inference_mode():
            for batch_start in range(0,len(starts),4):
                if batch_start % 100 == 0:
                    print('PathCRF phase/window',phase,batch_start,'/',len(starts),flush=True)
                selected = starts[batch_start:batch_start+4]
                chunks = [inputs[s:min(s+50,length)] for s in selected]
                padded = [np.pad(x,((0,50-len(x)),(0,0),(0,0)),mode='edge') for x in chunks]
                coarse,fine,_ = model(torch.from_numpy(np.stack(padded)))
                coarse = coarse[...,0]
                fine = fine.index_select(2,valid_edges)
                cm = (torch.logsumexp(coarse[:,:,home],dim=-1)-torch.logsumexp(coarse[:,:,away],dim=-1)).numpy()
                fm = (torch.logsumexp(fine[:,:,hs],dim=-1)-torch.logsumexp(fine[:,:,aw],dim=-1)).numpy()
                for j,s in enumerate(selected):
                    take = len(chunks[j])
                    w = np.minimum(np.arange(take)+1,np.arange(take,0,-1)).astype(float)
                    acc_macro[s:s+take] += cm[j,:take]*w
                    acc_micro[s:s+take] += fm[j,:take]*w
                    weights[s:s+take] += w
        ff = data['frames'][a:b]
        lo,hi = int(ff[0]),min(n,int(ff[-1])+5)
        macro[lo:hi] = np.interp(np.arange(lo,hi),ff,acc_macro/np.maximum(weights,1))
        micro[lo:hi] = np.interp(np.arange(lo,hi),ff,acc_micro/np.maximum(weights,1))
        covered[lo:hi] = True
        print('PathCRF geometry phase',phase,'5Hz frames',length,flush=True)
    np.savez_compressed(cache,macro=macro,micro=micro,covered=covered,signature=signature)
    return macro,micro,covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='Sample_Game_2')
    args = parser.parse_args(); torch.set_num_threads(2)
    macro,micro,covered = infer(args.match)
    print('Geometry inference coverage:',int(covered.sum()),'/',len(covered),flush=True)
    rows = []
    for sec,f,gt,graded,hide,reference in cases(args.match):
        predictions = {'reference':reference}
        for name,signal in [('macro',macro),('micro',micro)]:
            pred = reference.copy()
            mask = graded & covered & np.isin(reference,['H','A'])
            pred[mask] = np.where(signal[mask] >= 0,'H','A')
            predictions[name] = _merge_short_spells(pred,graded,18)
        for name,pred in predictions.items():
            row = dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec == '0':
                print('natural PathCRF',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('PathCRF emission evaluation',sec,flush=True)
    write_csv('evaluation/reports/pathcrf_emissions_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
