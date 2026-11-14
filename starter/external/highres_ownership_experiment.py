"""External-only fine-time ownership correction; frozen v2.2 remains teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from approved_predictions import approved_cases
from player_node_data import load_player_nodes
from temporal_team_model import temporal_design
from temporal_team_experiment import spell_weights
from residual_set_temporal import load_residual_model
from ownership_context import ball_context
from highres_ownership_model import HighResolutionOwnershipNet,load_highres_model,predict_highres
from research_checkpoint import save_checkpoint,resume_checkpoint
from temporal_decode import decode_scaled
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells
from eval_external import score_one,write_csv,aggregate_by_candidate

MODEL=Path('evaluation/reports/highres_ownership_v1.pt')
GAMES=['J03WOH','J03WN1','J03WOY','Sample_Game_1']


def train(resume=False):
    torch.manual_seed(573); rng=np.random.RandomState(573)
    old,mean,scale=load_residual_model('evaluation/reports/residual_set_v1.pt')
    streams,sampling=[],[]
    for game in GAMES:
        nodes=load_player_nodes(game,stride=2); half_nodes={}
        for sec,f,gt,g,hide,reference,approved,teacher in approved_cases(game,seed=73):
            x=(temporal_design(f)-mean)/scale; context=ball_context(f)
            y=(gt=='H').astype(np.float32); w=spell_weights(gt)*g*np.isin(reference,['H','A'])
            period=f.get('period',np.zeros(len(gt),np.int8))
            for half in np.unique(period):
                idx=np.flatnonzero(g & (period==half))
                if not len(idx):
                    continue
                ff=np.arange(((idx[0]+1)//2)*2,idx[-1]+1,2)
                if len(ff)>=1024:
                    if half not in half_nodes:
                        half_nodes[half]=nodes[ff//2]
                    streams.append((x[ff],y[ff],w[ff],half_nodes[half],context[ff],teacher[ff].astype(np.float32)))
                    sampling.append(3. if sec=='0' else 1.)
            print('prepared fine-time ownership',game,sec,flush=True)
    sampling=np.asarray(sampling); sampling/=sampling.sum()
    net=HighResolutionOwnershipNet()
    missing=net.load_state_dict(old.state_dict(),strict=False)
    assert all(key.startswith('temporal.blocks.6.') for key in missing.missing_keys) and not missing.unexpected_keys
    torch.nn.init.zeros_(net.temporal.out.weight); torch.nn.init.zeros_(net.temporal.out.bias)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    checkpoint=MODEL.with_suffix('.last.pt')
    first=resume_checkpoint(checkpoint,net,optimizer,rng) if resume and checkpoint.exists() else 0
    metadata=dict(architecture=dict(width=32,layers=7),mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),
                  stride=2,training_games=GAMES,teacher='residual_set_v1')
    history_path=MODEL.with_suffix('.history.json')
    history=json.loads(history_path.read_text())[:first] if first and history_path.exists() else []
    for epoch in range(first,12):
        started=time.monotonic(); losses=[]; net.train()
        for _ in range(80):
            batch=[]
            for _ in range(4):
                stream=streams[rng.choice(len(streams),p=sampling)]; a=rng.randint(len(stream[0])-1024+1)
                x,y,w,nodes,ctx,teacher=[s[a:a+1024].copy() for s in stream]
                if rng.rand()<.5:
                    x[:,:45]*=-1; y=1-y; nodes=nodes[:,::-1].copy(); teacher*=-1
                if rng.rand()<.5:
                    seen=nodes[...,6]>0; nodes[...,1]=np.where(seen,1-nodes[...,1],0.)
                    nodes[...,3]*=-1; ctx[:,1],ctx[:,3]=1-ctx[:,1],1-ctx[:,3]
                batch.append((x.T,y,w,nodes,ctx,teacher))
            xx,y,w,nnodes,ctx,teacher=[torch.from_numpy(np.stack(items)) for items in zip(*batch)]
            score=net(xx,nnodes,ctx,teacher)[:,256:-256]; y,w=y[:,256:-256],w[:,256:-256]
            loss=(F.binary_cross_entropy_with_logits(score,y,reduction='none')*w).sum()/w.sum().clamp_min(1.)
            pair=(w[:,1:]>0)&(w[:,:-1]>0)&(y[:,1:]==y[:,:-1])
            lp=torch.stack([F.logsigmoid(score),F.logsigmoid(-score)],dim=1)
            smooth=(lp[:,:,1:]-lp.detach()[:,:,:-1]).square().clamp_max(16.).mean(dim=1)
            loss+=.15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.monotonic()-started)
        history.append(row); history_path.write_text(json.dumps(history,indent=2))
        save_checkpoint(checkpoint,net,optimizer,rng,epoch+1,metadata); print(json.dumps(row),flush=True)
    torch.save(dict(metadata,state_dict=net.state_dict()),MODEL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--match',default='J03WMX'); args=parser.parse_args(); torch.set_num_threads(2)
    if args.train or args.resume:
        train(args.resume)
    model=load_highres_model(MODEL); nodes=load_player_nodes(args.match,stride=2); rows=[]
    signature=hashlib.sha256(MODEL.read_bytes()).hexdigest()
    for sec,f,gt,g,hide,reference,approved,teacher in approved_cases(args.match):
        cache=Path('evaluation/reports/prediction_cache')/('highres_logits_'+args.match+'_'+sec+'s.npz')
        logits=None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                if str(data['signature'])==signature:
                    logits=data['logits'].copy()
        if logits is None:
            logits=predict_highres(f,g,nodes,teacher,model)
            np.savez_compressed(cache,logits=logits,signature=signature)
        live=g & np.isin(reference,['H','A']); predictions={'approved':approved}
        for name,dense in [('teacher_grid',False),('fine_grid',True)]:
            home=decode_scaled(logits,live,f.get('period'),stride=2) if dense else decode_dead_logits(logits,live,f.get('period'))
            home[f['ball_present']]=logits[f['ball_present']]>=0
            pred=reference.copy(); pred[live]=np.where(home[live],'H','A')
            predictions[name]=_merge_short_spells(pred,g,18)
        for name,pred in predictions.items():
            row=dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec=='0':
                print('natural fine-time candidate',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('fine-time evaluation',args.match,sec,flush=True)
    write_csv('evaluation/reports/highres_ownership_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__=='__main__':
    main()
