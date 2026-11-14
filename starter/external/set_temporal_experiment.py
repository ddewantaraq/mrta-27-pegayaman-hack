"""Fit anonymous-player temporal ownership on external training matches only."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from player_node_data import load_player_nodes
from frozen_predictions import cases
from dropout import apply_dropout
from sportec_experiment import block_dropout
from temporal_team_experiment import spell_weights
from temporal_team_model import temporal_design
from set_temporal_team_model import (SetTemporalTeamNet, ball_context,
                                     load_set_temporal_model, predict_set_logits)
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate

MODEL = Path('evaluation/reports/set_temporal_v1.pt')


def prepare():
    streams = []
    for game in ('J03WOH', 'J03WN1', 'J03WOY', 'Sample_Game_1'):
        f, gt, graded = dataset(game)
        nodes = load_player_nodes(game)
        period = f.get('period', np.zeros(len(gt), np.int8))
        frames_by_half = []
        for half in np.unique(period):
            idx = np.flatnonzero(graded & (period == half))
            if len(idx):
                ff = np.arange(((idx[0]+4)//5)*5, idx[-1]+1, 5)
                if len(ff) >= 512:
                    frames_by_half.append((ff, nodes[ff//5]))
        for sec in (0, 2, 5, 15, 30):
            hide = np.zeros(len(gt), bool) if sec == 0 else block_dropout(len(gt), sec, seed=73)
            dropped = apply_dropout(f, hide)
            x, ctx = temporal_design(dropped), ball_context(dropped)
            y, w = (gt == 'H').astype(np.float32), spell_weights(gt)
            for ff, nnodes in frames_by_half:
                streams.append((x[ff], y[ff], w[ff]*graded[ff], nnodes, ctx[ff]))
        print('prepared player-set training', game, flush=True)
    all_x = np.concatenate([s[0] for s in streams])
    mean = all_x.mean(axis=0)
    mean[:45] = 0.
    scale = np.maximum(np.sqrt(np.mean((all_x-mean)**2, axis=0)), .01)
    return [((x-mean)/scale, y, w, nodes, ctx) for x,y,w,nodes,ctx in streams], mean, scale


def train(epochs=16, batches=80):
    torch.manual_seed(173)
    torch.set_num_threads(2)
    rng = np.random.RandomState(173)
    streams, mean, scale = prepare()
    net = SetTemporalTeamNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.0001)
    history = []
    for epoch in range(epochs):
        start = time.monotonic()
        losses = []
        net.train()
        for _ in range(batches):
            xs, ys, ws, ns, cs = [], [], [], [], []
            for _ in range(4):
                x,y,w,nodes,ctx = streams[rng.randint(len(streams))]
                a = rng.randint(len(x)-512+1)
                xx, yy = x[a:a+512].copy(), y[a:a+512].copy()
                nnodes, cc = nodes[a:a+512].copy(), ctx[a:a+512].copy()
                if rng.rand() < .5:
                    xx[:, :45] *= -1
                    yy = 1-yy
                    nnodes = nnodes[:, ::-1].copy()
                if rng.rand() < .5:
                    visible = nnodes[..., 6] > 0
                    nnodes[..., 1] = np.where(visible, 1-nnodes[..., 1], 0.)
                    nnodes[..., 3] *= -1
                    cc[:, 1], cc[:, 3] = 1-cc[:, 1], 1-cc[:, 3]
                xs.append(xx.T); ys.append(yy); ws.append(w[a:a+512]); ns.append(nnodes); cs.append(cc)
            xx = torch.from_numpy(np.stack(xs)); target = torch.from_numpy(np.stack(ys))
            weight = torch.from_numpy(np.stack(ws)); nnodes = torch.from_numpy(np.stack(ns))
            ctx = torch.from_numpy(np.stack(cs))
            score = net(xx, nnodes, ctx)[:, 128:-128]
            target, weight = target[:, 128:-128], weight[:, 128:-128]
            loss = (F.binary_cross_entropy_with_logits(score, target, reduction='none')*weight).sum()/weight.sum().clamp_min(1.)
            pair = (weight[:, 1:] > 0) & (weight[:, :-1] > 0) & (target[:, 1:] == target[:, :-1])
            lp = torch.stack([F.logsigmoid(score), F.logsigmoid(-score)], dim=1)
            smooth = (lp[:, :, 1:]-lp.detach()[:, :, :-1]).square().clamp_max(16.).mean(dim=1)
            loss += .15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step(); losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), seconds=time.monotonic()-start)
        history.append(row); print(json.dumps(row), flush=True)
    torch.save(dict(architecture=dict(width=32,layers=6), state_dict=net.state_dict(),
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale),
                    training_games=['J03WOH','J03WN1','J03WOY','Sample_Game_1']), MODEL)
    MODEL.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batches', type=int, default=80)
    parser.add_argument('--match', default='J03WMX')
    args = parser.parse_args(); torch.set_num_threads(2)
    if args.train:
        train(args.epochs, args.batches)
    loaded = load_set_temporal_model(MODEL)
    nodes = load_player_nodes(args.match)
    rows = []
    for drop,f,gt,graded,hide,reference in cases(args.match):
        logits = predict_set_logits(f, graded, nodes, loaded)
        teams = np.where(logits >= 0, 'H', 'A')
        predictions = {'reference': reference}
        for name, missing, threshold in [('full',False,0.),('missing',True,0.),('missing_confident75',True,np.log(3.))]:
            pred = reference.copy()
            mask = graded & np.isin(reference,['H','A']) & (np.abs(logits) >= threshold)
            if missing:
                mask &= ~f['ball_present']
            pred[mask] = teams[mask]
            predictions[name] = _merge_short_spells(pred, graded, 18)
        for name,pred in predictions.items():
            rows.append(dict(game=args.match,dropout=drop,candidate=name,**score_one(gt,pred,hide)))
        print('player-set evaluation',args.match,drop,flush=True)
    write_csv('evaluation/reports/set_temporal_'+args.match+'.csv',rows)
    print(json.dumps(aggregate_by_candidate(rows),indent=2))


if __name__ == '__main__':
    main()
