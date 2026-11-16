"""Official dynamic CRF with geometry windows and optional observed-team hints.

Windows/roster phases come solely from tracking. Soft hints use possession-zone
geometry on observed ball frames, never reference events or episode boundaries.
Fixed before evaluation: hint log-odds log(9), 10s windows, 50% overlap, majority
vote gate abs(vote)>=.5. Reference D/X and uncertain teams are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

from frozen_predictions import cases
from pathcrf_geometry import prepare_geometry,phases
from pathcrf_experiment import ROOT
from eval_external import score_one,write_csv,aggregate_by_candidate
from state_machine import _merge_short_spells


def infer(game,case_features):
    sys.path.insert(0,str(ROOT))
    from models.utils import build_model
    model_path = ROOT/'saved/140/model/state_dict_best.pt'
    signature = hashlib.sha256(model_path.read_bytes()+Path(__file__).read_bytes()).hexdigest()
    cache = Path('evaluation/reports/prediction_cache')/('pathcrf_crf_votes_'+game+'.npz')
    if cache.exists():
        with np.load(cache,allow_pickle=False) as data:
            if str(data['signature']) == signature:
                return data['free'].copy(),data['hinted'].copy(),data['covered'].copy()
    args = json.loads((ROOT/'saved/140/args.json').read_text())
    net = build_model(args,torch.device('cpu'))
    net.load_state_dict(torch.load(model_path,map_location='cpu',weights_only=True)); net.eval()
    data = prepare_geometry(game); n = int(data['n_frames'])
    free,hinted,covered = np.zeros(n),np.zeros((len(case_features),n)),np.zeros(n,bool)
    valid_edges = net.valid_orig_edge_ids
    src,dst = net.comp_src.numpy(),net.comp_dst.numpy()
    for phase,(a,b,inputs,available,home_nodes) in enumerate(phases(data)):
        length = len(inputs); starts = list(range(0,max(1,length-49),25))
        if starts[-1] < max(0,length-50):
            starts.append(length-50)
        accum = np.zeros((len(case_features)+1,length)); weights = np.zeros(length)
        edge_available = available[src] & available[dst]
        home_edge = torch.from_numpy(home_nodes[src] & edge_available)
        away_edge = torch.from_numpy((src < 22) & ~home_nodes[src] & edge_available)
        src_vote = np.where(home_nodes[src],1.,np.where(src < 22,-1.,0.))
        ff = data['frames'][a:b]
        anchors = [np.asarray(f['pz_team'])[ff] for f in case_features]
        with torch.inference_mode():
            for batch_start in range(0,len(starts),4):
                if batch_start % 100 == 0:
                    print('PathCRF sequence phase/window',phase,batch_start,'/',len(starts),flush=True)
                selected = starts[batch_start:batch_start+4]
                chunks = [inputs[s:min(s+50,length)] for s in selected]
                padded = [np.pad(x,((0,50-len(x)),(0,0),(0,0)),mode='edge') for x in chunks]
                _,emissions,embedding = net(torch.from_numpy(np.stack(padded)))
                emissions = emissions.index_select(2,valid_edges)
                embedding = embedding.index_select(2,valid_edges)
                emissions[:,:,~torch.from_numpy(edge_available)] = -1e4
                decoded = [net.crf.decode(emissions,embedding).numpy()]
                for anchor in anchors:
                    aa = np.stack([np.pad(anchor[s:s+len(chunks[j])],(0,50-len(chunks[j])),constant_values=-1)
                                   for j,s in enumerate(selected)])
                    bias_h = torch.from_numpy((aa == 0).astype(np.float32))[:,:,None]
                    bias_a = torch.from_numpy((aa == 1).astype(np.float32))[:,:,None]
                    bias = np.log(9.)*(bias_h*home_edge[None,None]+bias_a*away_edge[None,None])
                    decoded.append(net.crf.decode(emissions+bias,embedding).numpy())
                for j,s in enumerate(selected):
                    take = len(chunks[j]); w = np.minimum(np.arange(take)+1,np.arange(take,0,-1)).astype(float)
                    for c,path in enumerate(decoded):
                        accum[c,s:s+take] += src_vote[path[j,:take]]*w
                    weights[s:s+take] += w
        lo,hi = int(ff[0]),min(n,int(ff[-1])+5)
        for c in range(len(case_features)+1):
            vote = np.interp(np.arange(lo,hi),ff,accum[c]/np.maximum(weights,1))
            if c == 0:
                free[lo:hi] = vote
            else:
                hinted[c-1,lo:hi] = vote
        covered[lo:hi] = True
        print('PathCRF sequence phase completed',phase,length,flush=True)
    np.savez_compressed(cache,free=free,hinted=hinted,covered=covered,signature=signature)
    return free,hinted,covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--match',default='Sample_Game_2')
    args = parser.parse_args(); torch.set_num_threads(2)
    configurations = list(cases(args.match))
    free,hinted,covered = infer(args.match,[c[1] for c in configurations])
    rows = []
    for c,(sec,f,gt,graded,hide,reference) in enumerate(configurations):
        predictions = {'reference':reference}
        for name,vote in [('ball_free_crf',free),('observed_team_crf',hinted[c])]:
            pred = reference.copy()
            mask = graded & covered & np.isin(reference,['H','A']) & (np.abs(vote) >= .5)
            pred[mask] = np.where(vote[mask] >= 0,'H','A')
            predictions[name] = _merge_short_spells(pred,graded,18)
        for name,pred in predictions.items():
            row = dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec == '0':
                print('natural CRF',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('sequence evaluation',sec,flush=True)
    write_csv('evaluation/reports/pathcrf_sequence_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
