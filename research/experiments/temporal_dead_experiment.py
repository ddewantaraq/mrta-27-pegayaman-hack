"""External-only training of a temporal live/dead correction network."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from motion_dead_experiment import dataset
from player_node_data import load_player_nodes
from dropout import apply_dropout
from sportec_experiment import block_dropout
from approved_predictions import approved_cases
from research_checkpoint import save_checkpoint,resume_checkpoint
from set_temporal_team_model import load_set_temporal_model,ball_context
from temporal_dead_model import TemporalDeadNet,load_temporal_dead,predict_temporal_dead
from motion_dead_model import design_motion_features
from boosted_dead_model import load_boosted_model,boosted_logits,DEFAULT_BOOSTED_MODEL_PATH
from dead_ball_model import decode_dead_logits
from state_machine import _bridge_short_dead,_merge_short_spells
from eval_external import score_one,write_csv,aggregate_by_candidate

MODEL = Path('evaluation/reports/temporal_dead_v1.pt')
GAMES = ['J03WOH','J03WN1','J03WOY','Sample_Game_1']


def prepare():
    streams,sampling = [],[]
    teacher = load_boosted_model(DEFAULT_BOOSTED_MODEL_PATH)
    for game in GAMES:
        f,gt,g = dataset(game); nodes = load_player_nodes(game)
        period = f.get('period',np.zeros(len(gt),np.int8))
        halves = []
        for half in np.unique(period):
            idx = np.flatnonzero(g & (period==half))
            if len(idx):
                ff = np.arange(((idx[0]+4)//5)*5,idx[-1]+1,5)
                if len(ff)>=512:
                    halves.append((ff,nodes[ff//5]))
        for sec in (0,2,5,10,30):
            hide = np.zeros(len(gt),bool) if sec==0 else block_dropout(len(gt),sec,seed=73)
            dropped = apply_dropout(f,hide); design = design_motion_features(dropped); ctx = ball_context(dropped)
            for ff,nnodes in halves:
                xx = design[ff]
                base = boosted_logits(xx,teacher).astype(np.float32)
                streams.append((xx,(gt[ff]=='D').astype(np.float32),
                                (g[ff] & (gt[ff]!='X')).astype(np.float32),nnodes,ctx[ff],base))
                sampling.append(3. if sec==0 else 1.)
        print('prepared temporal live/dead',game,flush=True)
    all_x = np.concatenate([s[0] for s in streams]); mean = all_x.mean(axis=0)
    scale = np.maximum(all_x.std(axis=0),.01)
    streams = [((x-mean)/scale,y,w,n,c,b) for x,y,w,n,c,b in streams]
    sampling = np.asarray(sampling); sampling /= sampling.sum()
    return streams,sampling,mean,scale


def train(resume=False):
    torch.manual_seed(473); rng = np.random.RandomState(473)
    streams,sampling,mean,scale = prepare(); net = TemporalDeadNet()
    pretrained,_,_ = load_set_temporal_model('evaluation/reports/set_temporal_v1.pt')
    net.player.load_state_dict(pretrained.player.state_dict())
    optimizer = torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.0001)
    checkpoint = MODEL.with_suffix('.last.pt')
    first = resume_checkpoint(checkpoint,net,optimizer,rng) if resume and checkpoint.exists() else 0
    metadata = dict(architecture=dict(width=32,layers=6),mean=torch.from_numpy(mean),
                    scale=torch.from_numpy(scale),training_games=GAMES)
    history_path = MODEL.with_suffix('.history.json')
    history = json.loads(history_path.read_text())[:first] if first and history_path.exists() else []
    for epoch in range(first,12):
        started=time.monotonic(); losses=[]; net.train()
        for _ in range(80):
            batch=[]
            for _ in range(4):
                stream=streams[rng.choice(len(streams),p=sampling)]; a=rng.randint(len(stream[0])-512+1)
                x,y,w,nodes,ctx,base=[s[a:a+512].copy() for s in stream]
                if rng.rand()<.5:
                    nodes=nodes[:,::-1].copy()
                if rng.rand()<.5:
                    seen=nodes[...,6]>0; nodes[...,1]=np.where(seen,1-nodes[...,1],0.)
                    nodes[...,3]*=-1; ctx[:,1],ctx[:,3]=1-ctx[:,1],1-ctx[:,3]
                batch.append((x.T,y,w,nodes,ctx,base))
            xx,y,w,nnodes,ctx,base=[torch.from_numpy(np.stack(items)) for items in zip(*batch)]
            score=net(xx,nnodes,ctx,base)[:,128:-128]; y,w=y[:,128:-128],w[:,128:-128]
            loss=(F.binary_cross_entropy_with_logits(score,y,reduction='none')*w).sum()/w.sum().clamp_min(1.)
            pair=(w[:,1:]>0)&(w[:,:-1]>0)&(y[:,1:]==y[:,:-1])
            lp=torch.stack([F.logsigmoid(score),F.logsigmoid(-score)],dim=1)
            smooth=(lp[:,:,1:]-lp.detach()[:,:,:-1]).square().clamp_max(16.).mean(dim=1)
            loss+=.15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.monotonic()-started)
        history.append(row); history_path.write_text(json.dumps(history,indent=2))
        save_checkpoint(checkpoint,net,optimizer,rng,epoch+1,metadata)
        print(json.dumps(row),flush=True)
    torch.save(dict(metadata,state_dict=net.state_dict()),MODEL)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--match',default='J03WMX')
    args=parser.parse_args(); torch.set_num_threads(2)
    if args.train or args.resume:
        train(args.resume)
    model=load_temporal_dead(MODEL); nodes=load_player_nodes(args.match); rows=[]
    for sec,f,gt,g,hide,reference,approved,owner_logits in approved_cases(args.match):
        dead_logits=predict_temporal_dead(f,g,nodes,model)
        latent=decode_dead_logits(owner_logits,g,f.get('period'))
        latent[f['ball_present']]=owner_logits[f['ball_present']]>=0
        predictions={'approved':approved}
        for name,dead in [('raw_dead',dead_logits>=0),('smooth_dead',decode_dead_logits(dead_logits,g,f.get('period')))]:
            pred=approved.copy(); restore=g&(pred=='D')&~dead
            pred[restore]=np.where(latent[restore],'H','A'); pred[g&dead]='D'
            predictions[name]=_merge_short_spells(_bridge_short_dead(pred,g,25),g,18)
        for name,pred in predictions.items():
            row=dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec=='0':
                print('natural dead candidate',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('temporal dead evaluation',args.match,sec,flush=True)
    write_csv('evaluation/reports/temporal_dead_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__=='__main__':
    main()
