"""Rank 8 probe: SSL pretrain (masked ball forecasting) + frozen probe.

Pretrain the GNN encoder label-free: randomly mask ball inputs, regress
next-step ball displacement from frame embeddings. Freeze, train ONLY a
linear probe on contested labels, LOMO over FIT. Kill gate: frozen probe
must beat v2.3 logit contested accuracy (from gnn_probe protocol).
Writes evaluation/reports/ssl_probe.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first import FIT
from gnn_probe import Net, match_arrays


def pretrain(train, epochs=12, seed=0):
    torch.manual_seed(seed)
    model = Net(message=True)
    fore = nn.Linear(64, 2)
    opt = torch.optim.Adam(list(model.parameters()) + list(fore.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for d in train:
            x = torch.from_numpy(d['players'])[None]
            b = torch.from_numpy(d['ball'])[None].clone()
            target = torch.zeros_like(b[..., :2])
            target[:, :-1] = b[:, 1:, :2] - b[:, :-1, :2]
            mask = (torch.rand(b.shape[1], generator=gen) < .25)
            b[:, mask, 2] = 0.
            b[:, mask, :2] = .5
            opt.zero_grad()
            emb = model.embed(x, b)
            pred = fore(emb)
            w = torch.zeros(b.shape[1])
            w[mask] = 1.
            loss = ((pred - target) ** 2).mean(-1)[0]
            loss = (loss * w).sum() / w.sum().clamp(min=1.)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def embed(model, d):
    model.eval()
    return model.embed(torch.from_numpy(d['players'])[None],
                       torch.from_numpy(d['ball'])[None]).numpy()[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--out', default=str(REPORTS / 'ssl_probe.json'))
    args = parser.parse_args()
    data = {g: match_arrays(load_match(g)) for g in FIT}
    rows = []
    for hold in FIT:
        train = [data[g] for g in FIT if g != hold]
        test = data[hold]
        model = pretrain(train, epochs=args.epochs)
        # frozen linear probe (logistic regression via torch, head only)
        torch.manual_seed(0)
        probe = nn.Linear(64, 1)
        opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
        xs = [torch.from_numpy(embed(model, d)) for d in train]
        ys = [torch.from_numpy(d['y']) for d in train]
        ws = [torch.from_numpy(
            (np.where(d['live'], 1., 0.) * np.where(d['contested'], 3., 1.)).astype(np.float32))
            for d in train]
        for ep in range(30):
            for x, y, w in zip(xs, ys, ws):
                opt.zero_grad()
                logit = probe(x).squeeze(-1)
                loss = (nn.functional.binary_cross_entropy_with_logits(
                    logit, (y > 0).float(), reduction='none') * w).sum() / w.sum().clamp(min=1.)
                loss.backward()
                opt.step()
        with torch.inference_mode():
            logit = probe(torch.from_numpy(embed(model, test))).numpy()
        mask = test['contested'] & test['visible']
        acc = float(np.mean(np.sign(logit[mask]) == test['y'][mask]))
        rows.append(dict(game=hold, ssl_probe_acc=acc, n=int(mask.sum())))
        print(hold, 'ssl frozen probe contested', round(acc, 4), flush=True)
    result = dict(rows=rows, mean=float(np.mean([r['ssl_probe_acc'] for r in rows])),
                  v23_reference=0.772)
    print(json.dumps(result, indent=2))
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
