"""Locked external-only training of a temporal ownership model.

Fit: Sportec WOH/WN1/WOY + Metrica 1. Development: WMX and Metrica 2.
Sample Game 3 is reserved until a final candidate/decision rule is frozen.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from frozen_predictions import cases
from sportec_experiment import block_dropout
from dropout import apply_dropout
from temporal_team_model import (TemporalTeamNet, temporal_design, load_temporal_model,
                                 predict_temporal_logits)
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells
from eval_external import score_one, write_csv, aggregate_by_candidate

MODEL = Path('evaluation/reports/temporal_team_v1.pt')


def spell_weights(gt):
    weights = np.zeros(len(gt), np.float32)
    i = 0
    while i < len(gt):
        j = i+1
        while j < len(gt) and gt[j] == gt[i]:
            j += 1
        if gt[i] in ('H', 'A'):
            weights[i:j] = np.clip(np.sqrt(250./(j-i)), .25, 4.)
        i = j
    return weights


def prepare_training():
    streams = []
    for game in ('J03WOH', 'J03WN1', 'J03WOY', 'Sample_Game_1'):
        f, gt, graded = dataset(game)
        period = f.get('period', np.zeros(len(gt), np.int8))
        for sec in (0, 2, 5, 15, 30):
            hide = np.zeros(len(gt), bool) if sec == 0 else block_dropout(len(gt), sec, seed=73)
            x = temporal_design(apply_dropout(f, hide))
            w = spell_weights(gt)
            y = (gt == 'H').astype(np.float32)
            for p in np.unique(period):
                idx = np.flatnonzero(graded & (period == p))
                if len(idx) == 0:
                    continue
                # Tracking period is contiguous; ungraded gaps are excluded.
                frames = np.arange(idx[0], idx[-1]+1, 5)
                if len(frames) >= 512:
                    streams.append((x[frames], y[frames], w[frames]*graded[frames]))
        print('prepared temporal training', game, flush=True)
    all_x = np.concatenate([s[0] for s in streams])
    mean = all_x.mean(axis=0)
    mean[:45] = 0.  # signed ownership features must remain team-symmetric
    scale = np.maximum(np.sqrt(np.mean((all_x-mean)**2, axis=0)), .01)
    streams = [((x-mean)/scale, y, w) for x, y, w in streams]
    return streams, mean, scale


def train(epochs=12, batches=80):
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)
    torch.set_num_threads(2)
    streams, mean, scale = prepare_training()
    net = TemporalTeamNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.0001)
    rng = np.random.RandomState(73)
    history = []
    for epoch in range(epochs):
        started = time.monotonic()
        net.train()
        losses = []
        for _ in range(batches):
            xs, ys, ws = [], [], []
            for _ in range(4):
                x, y, w = streams[rng.randint(len(streams))]
                a = rng.randint(len(x)-512+1)
                xx, yy = x[a:a+512].copy(), y[a:a+512].copy()
                if rng.rand() < .5:
                    xx[:, :45] *= -1
                    yy = 1-yy
                xs.append(xx.T)
                ys.append(yy)
                ws.append(w[a:a+512])
            tensor = torch.from_numpy(np.stack(xs))
            target = torch.from_numpy(np.stack(ys))
            weight = torch.from_numpy(np.stack(ws))
            score = net(tensor)
            # Full receptive-field context on both sides of supervised frames.
            score, target, weight = score[:, 128:-128], target[:, 128:-128], weight[:, 128:-128]
            loss = (F.binary_cross_entropy_with_logits(score, target, reduction='none')*weight).sum()/weight.sum().clamp_min(1.)
            pair = (weight[:, 1:] > 0) & (weight[:, :-1] > 0) & (target[:, 1:] == target[:, :-1])
            logp = torch.stack([F.logsigmoid(score), F.logsigmoid(-score)], dim=1)
            smooth = (logp[:, :, 1:]-logp.detach()[:, :, :-1]).square().clamp_max(16.).mean(dim=1)
            loss += .15*(smooth*pair).sum()/pair.sum().clamp_min(1.)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), seconds=time.monotonic()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
    torch.save(dict(architecture=dict(input_dim=57, width=32, layers=6),
                    state_dict=net.state_dict(), mean=torch.from_numpy(mean),
                    scale=torch.from_numpy(scale), epochs=epochs,
                    training_games=['J03WOH','J03WN1','J03WOY','Sample_Game_1']), MODEL)
    MODEL.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batches', type=int, default=80)
    parser.add_argument('--match', default='J03WMX')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.train:
        train(args.epochs, args.batches)
    loaded = load_temporal_model(MODEL)
    rows = []
    for drop, f, gt, graded, hide, reference in cases(args.match):
        logits = predict_temporal_logits(f, graded, loaded)
        raw_team = np.where(logits >= 0, 'H', 'A')
        viterbi = np.where(decode_dead_logits(logits, graded, f.get('period')), 'H', 'A')
        predictions = {'reference': reference}
        for name, team, missing_only in [('full_raw', raw_team, False),
                                         ('full_viterbi', viterbi, False),
                                         ('missing_raw', raw_team, True),
                                         ('missing_viterbi', viterbi, True)]:
            pred = reference.copy()
            mask = graded & np.isin(reference, ['H', 'A'])
            if missing_only:
                mask &= ~f['ball_present']
            pred[mask] = team[mask]
            predictions[name] = _merge_short_spells(pred, graded, 18)
        confident = reference.copy()
        mask = (graded & ~f['ball_present'] & np.isin(reference, ['H', 'A'])
                & (np.abs(logits) >= np.log(3.)))
        confident[mask] = raw_team[mask]
        predictions['missing_confident75'] = _merge_short_spells(confident, graded, 18)
        for name, pred in predictions.items():
            rows.append(dict(game=args.match, dropout=drop, candidate=name,
                             **score_one(gt, pred, hide)))
        print('temporal evaluation', args.match, drop, flush=True)
    write_csv('evaluation/reports/temporal_team_'+args.match+'.csv', rows)
    print(json.dumps(aggregate_by_candidate(rows), indent=2))


if __name__ == '__main__':
    main()
