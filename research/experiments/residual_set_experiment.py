"""Fit a baseline-aware correction head using external training matches only."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F

from frozen_predictions import cases
from player_node_data import load_player_nodes
from set_temporal_team_model import load_set_temporal_model, ball_context
from temporal_team_model import temporal_design
from temporal_team_experiment import spell_weights
from residual_set_temporal import ResidualSetTemporalNet, load_residual_model, predict_residual_logits
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate
from dead_ball_model import decode_dead_logits

MODEL = Path('evaluation/reports/residual_set_v1.pt')


def train():
    torch.manual_seed(273)
    rng = np.random.RandomState(273)
    pretrained, mean, scale = load_set_temporal_model('evaluation/reports/set_temporal_v1.pt')
    streams, sampling = [], []
    games = ['J03WOH','J03WN1','J03WOY','Sample_Game_1']
    for game in games:
        nodes = load_player_nodes(game)
        for sec,f,gt,graded,hide,reference in cases(game, seed=73):
            x = (temporal_design(f)-mean)/scale
            ctx = ball_context(f)
            baseline = np.where(reference == 'H',1.,np.where(reference == 'A',-1.,0.)).astype(np.float32)
            y = (gt == 'H').astype(np.float32)
            w = spell_weights(gt)*graded*np.isin(reference,['H','A'])
            period = f.get('period',np.zeros(len(gt),np.int8))
            for half in np.unique(period):
                idx = np.flatnonzero(graded & (period == half))
                if not len(idx):
                    continue
                ff = np.arange(((idx[0]+4)//5)*5,idx[-1]+1,5)
                if len(ff) >= 512:
                    streams.append((x[ff],y[ff],w[ff],nodes[ff//5],ctx[ff],baseline[ff]))
                    sampling.append(3. if sec == '0' else 1.)
            print('prepared baseline-aware training',game,sec,flush=True)
    sampling = np.asarray(sampling); sampling /= sampling.sum()
    net = ResidualSetTemporalNet()
    state = net.state_dict()
    for key,value in pretrained.state_dict().items():
        if key == 'temporal.project.weight':
            state[key].zero_(); state[key][:,:157] = value
        else:
            state[key] = value.clone()
    state['temporal.out.weight'] *= .25
    state['temporal.out.bias'] *= .25
    net.load_state_dict(state)
    optimizer = torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.0001)
    history = []
    for epoch in range(12):
        started = time.monotonic(); losses = []; net.train()
        for _ in range(80):
            batch = []
            for _ in range(4):
                stream = streams[rng.choice(len(streams),p=sampling)]
                a = rng.randint(len(stream[0])-512+1)
                x,y,w,nodes,ctx,base = [s[a:a+512].copy() for s in stream]
                if rng.rand() < .5:
                    x[:,:45] *= -1; y = 1-y; nodes = nodes[:,::-1].copy(); base *= -1
                if rng.rand() < .5:
                    visible = nodes[...,6] > 0
                    nodes[...,1] = np.where(visible,1-nodes[...,1],0.); nodes[...,3] *= -1
                    ctx[:,1],ctx[:,3] = 1-ctx[:,1],1-ctx[:,3]
                batch.append((x.T,y,w,nodes,ctx,base))
            xx,y,w,nnodes,ctx,base = [torch.from_numpy(np.stack(items)) for items in zip(*batch)]
            score = net(xx,nnodes,ctx,base)[:,128:-128]
            y,w = y[:,128:-128],w[:,128:-128]
            loss = (F.binary_cross_entropy_with_logits(score,y,reduction='none')*w).sum()/w.sum().clamp_min(1.)
            pair = (w[:,1:] > 0) & (w[:,:-1] > 0) & (y[:,1:] == y[:,:-1])
            lp = torch.stack([F.logsigmoid(score),F.logsigmoid(-score)],dim=1)
            smooth = (lp[:,:,1:]-lp.detach()[:,:,:-1]).square().clamp_max(16.).mean(dim=1)
            loss += .15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1,loss=float(np.mean(losses)),seconds=time.monotonic()-started)
        history.append(row); print(json.dumps(row),flush=True)
        MODEL.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
    torch.save(dict(architecture=dict(width=32,layers=6,prior=net.prior),state_dict=net.state_dict(),
                    mean=torch.from_numpy(mean),scale=torch.from_numpy(scale),training_games=games),MODEL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true'); parser.add_argument('--match',default='J03WMX')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.train:
        train()
    model = load_residual_model(MODEL); nodes = load_player_nodes(args.match); rows = []
    signature = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    for sec,f,gt,graded,hide,reference in cases(args.match):
        cache = Path('evaluation/reports/prediction_cache')/('residual_logits_'+args.match+'_'+sec+'s.npz')
        logits = None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                if str(data['signature']) == signature:
                    logits = data['logits'].copy()
        if logits is None:
            logits = predict_residual_logits(f,graded,nodes,reference,model)
            np.savez_compressed(cache,logits=logits,signature=signature)
        predictions = {'reference':reference}
        for name,missing in [('full',False),('missing',True)]:
            mask = graded & np.isin(reference,['H','A'])
            if missing:
                mask &= ~f['ball_present']
            pred = reference.copy(); pred[mask] = np.where(logits[mask] >= 0,'H','A')
            predictions[name] = _merge_short_spells(pred,graded,18)
        live = graded & np.isin(reference,['H','A'])
        for name,prob in [('full_v05',.05),('full_v01',.01)]:
            home = decode_dead_logits(logits,live,f.get('period'),transition_prob=prob)
            pred = reference.copy(); pred[live] = np.where(home[live],'H','A')
            predictions[name] = _merge_short_spells(pred,graded,18)
            if prob == .01:
                home[f['ball_present']] = logits[f['ball_present']] >= 0
                pred = reference.copy(); pred[live] = np.where(home[live],'H','A')
                predictions['raw_visible_smooth_missing'] = _merge_short_spells(pred,graded,18)
        for name,pred in predictions.items():
            row = dict(game=args.match,dropout=sec,candidate=name,**score_one(gt,pred,hide)); rows.append(row)
            if sec == '0':
                print('natural residual',name,row['spell_f1_mean'],row['hidden_team_acc'],flush=True)
        print('residual evaluation',args.match,sec,flush=True)
    write_csv('evaluation/reports/residual_set_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
