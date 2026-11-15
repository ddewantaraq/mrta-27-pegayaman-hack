"""External-only joint ownership and turnover-boundary training."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from frozen_predictions import cases
from approved_predictions import approved_cases
from player_node_data import load_player_nodes
from residual_set_temporal import load_residual_model
from temporal_team_model import temporal_design
from temporal_team_experiment import spell_weights
from ownership_context import ball_context
from boundary_ownership_model import BoundaryOwnershipNet,load_boundary_model,predict_boundary_logits,decode_boundary
from research_checkpoint import save_checkpoint,resume_checkpoint
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells
from eval_external import score_one,write_csv,aggregate_by_candidate

MODEL=Path('evaluation/reports/boundary_ownership_v1.pt')
GAMES=['J03WOH','J03WN1','J03WOY','Sample_Game_1']


def boundary_target(gt,graded,period):
    target=np.zeros(len(gt),np.float32)
    live=np.isin(gt,['H','A']) & graded
    changes=np.flatnonzero(live[1:] & live[:-1] & (gt[1:]!=gt[:-1]) & (period[1:]==period[:-1]))+1
    for frame in changes:
        idx=np.arange(max(0,frame-15),min(len(gt),frame+16))
        target[idx]=np.maximum(target[idx],np.exp(-.5*((idx-frame)/5.)**2))
    return target


def train(resume=False):
    torch.manual_seed(373); rng=np.random.RandomState(373)
    old,mean,scale=load_residual_model('evaluation/reports/residual_set_v1.pt')
    streams,sampling=[],[]
    for game in GAMES:
        nodes=load_player_nodes(game)
        for sec,f,gt,g,hide,reference in cases(game,seed=73):
            x=(temporal_design(f)-mean)/scale; ctx=ball_context(f)
            base=np.where(reference=='H',1.,np.where(reference=='A',-1.,0.)).astype(np.float32)
            y=(gt=='H').astype(np.float32); w=spell_weights(gt)*g*np.isin(reference,['H','A'])
            period=f.get('period',np.zeros(len(gt),np.int8)); edge=boundary_target(gt,g,period)
            for half in np.unique(period):
                idx=np.flatnonzero(g & (period==half))
                if not len(idx):
                    continue
                ff=np.arange(((idx[0]+4)//5)*5,idx[-1]+1,5)
                if len(ff)>=512:
                    streams.append((x[ff],y[ff],w[ff],nodes[ff//5],ctx[ff],base[ff],edge[ff]))
                    sampling.append(3. if sec=='0' else 1.)
        print('prepared ownership/boundary',game,flush=True)
    sampling=np.asarray(sampling); sampling/=sampling.sum()
    net=BoundaryOwnershipNet(); incompatible=net.load_state_dict(old.state_dict(),strict=False)
    assert set(incompatible.missing_keys)=={'boundary.weight','boundary.bias'} and not incompatible.unexpected_keys
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    checkpoint=MODEL.with_suffix('.last.pt')
    first=resume_checkpoint(checkpoint,net,optimizer,rng) if resume and checkpoint.exists() else 0
    metadata=dict(architecture=dict(width=32,layers=6,prior=net.prior),mean=torch.from_numpy(mean),
                  scale=torch.from_numpy(scale),training_games=GAMES,boundary_sigma_frames=5,boundary_positive_weight=10.)
    history_path=MODEL.with_suffix('.history.json')
    history=json.loads(history_path.read_text())[:first] if first and history_path.exists() else []
    for epoch in range(first,12):
        start=time.monotonic(); losses=[]; net.train()
        for _ in range(80):
            batch=[]
            for _ in range(4):
                stream=streams[rng.choice(len(streams),p=sampling)]; a=rng.randint(len(stream[0])-512+1)
                x,y,w,nodes,ctx,base,edge=[s[a:a+512].copy() for s in stream]
                if rng.rand()<.5:
                    x[:,:45]*=-1; y=1-y; nodes=nodes[:,::-1].copy(); base*=-1
                if rng.rand()<.5:
                    seen=nodes[...,6]>0; nodes[...,1]=np.where(seen,1-nodes[...,1],0.)
                    nodes[...,3]*=-1; ctx[:,1],ctx[:,3]=1-ctx[:,1],1-ctx[:,3]
                batch.append((x.T,y,w,nodes,ctx,base,edge))
            xx,y,w,nnodes,ctx,base,edge=[torch.from_numpy(np.stack(items)) for items in zip(*batch)]
            score,boundary=net(xx,nnodes,ctx,base)
            score,boundary=score[:,128:-128],boundary[:,128:-128]
            y,w,edge=y[:,128:-128],w[:,128:-128],edge[:,128:-128]
            loss=(F.binary_cross_entropy_with_logits(score,y,reduction='none')*w).sum()/w.sum().clamp_min(1.)
            loss+=.2*(F.binary_cross_entropy_with_logits(boundary,edge,pos_weight=torch.tensor(10.),reduction='none')*(w>0)).sum()/(w>0).sum().clamp_min(1.)
            pair=(w[:,1:]>0)&(w[:,:-1]>0)&(y[:,1:]==y[:,:-1])
            lp=torch.stack([F.logsigmoid(score),F.logsigmoid(-score)],dim=1)
            smooth=(lp[:,:,1:]-lp.detach()[:,:,:-1]).square().clamp_max(16.).mean(dim=1)
            loss+=.15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.monotonic()-start)
        history.append(row); history_path.write_text(json.dumps(history,indent=2))
        save_checkpoint(checkpoint,net,optimizer,rng,epoch+1,metadata); print(json.dumps(row),flush=True)
    torch.save(dict(metadata,state_dict=net.state_dict()),MODEL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--match',default='J03WMX')
    parser.add_argument('--candidates',nargs='+',default=None,
                        help='report only these frozen policies plus approved')
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.train or args.resume:
        train(args.resume)
    model=load_boundary_model(MODEL); nodes=load_player_nodes(args.match); rows=[]
    signature=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    for sec,f,gt,g,hide,reference,approved,old_logits in approved_cases(args.match):
        cache=Path('evaluation/reports/prediction_cache')/('boundary_logits_'+args.match+'_'+sec+'s.npz')
        logits,boundary=None,None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                if str(data['signature'])==signature:
                    logits,boundary=data['logits'].copy(),data['boundary'].copy()
        if logits is None:
            logits,boundary=predict_boundary_logits(f,g,nodes,reference,model)
            np.savez_compressed(cache,logits=logits,boundary=boundary,signature=signature)
        live=g & np.isin(reference,['H','A']); predictions={'approved':approved}
        for name,adaptive,hybrid in [('tuned_hybrid',False,True),('boundary_all',True,False),('boundary_hybrid',True,True)]:
            home=decode_boundary(logits,boundary,live,f.get('period')) if adaptive else decode_dead_logits(logits,live,f.get('period'))
            if hybrid:
                home[f['ball_present']]=logits[f['ball_present']]>=0
            pred=reference.copy(); pred[live]=np.where(home[live],'H','A')
            predictions[name]=_merge_short_spells(pred,g,18)
        for name,signal,adaptive in [('old_with_boundary',old_logits,True),
                                      ('ensemble_constant',.5*(old_logits+logits),False),
                                      ('ensemble_boundary',.5*(old_logits+logits),True)]:
            home=decode_boundary(signal,boundary,live,f.get('period')) if adaptive else decode_dead_logits(signal,live,f.get('period'))
            home[f['ball_present']]=signal[f['ball_present']]>=0
            pred=reference.copy(); pred[live]=np.where(home[live],'H','A')
            predictions[name]=_merge_short_spells(pred,g,18)
        for name,pred in predictions.items():
            if args.candidates and name!='approved' and name not in args.candidates:
                continue
            row=dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec=='0':
                print('natural boundary candidate',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('boundary evaluation',args.match,sec,flush=True)
    write_csv('evaluation/reports/boundary_ownership_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__=='__main__':
    main()
