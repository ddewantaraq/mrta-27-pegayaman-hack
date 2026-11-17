"""Rank 5 probe: trajectory GNN vs pooled baseline on contested windows.

Per 5Hz frame: 22 anonymous players (common pitch frame, unflipped from
team-oriented nodes) + ball-relative geometry; 2 rounds of distance-
weighted message passing, team-diff pooling, temporal conv, team logit.
Ablation `--no-message` removes message passing (pooled MLP control).
LOMO over FIT matches; metric = accuracy on visible contested frames
(within 25f of a GT switch) vs v2.3 ensemble logit sign on same frames.
Kill gate: GNN must beat both pooled control and v2.3 on contested.
Probe only -- no NumPy export, no ship claim.
Writes evaluation/reports/gnn_probe.json.
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
from error_corrector_model import ensemble_team_logits
from event_first import FIT

LENGTH, WIDTH = 105.3, 68.


def match_arrays(data):
    """players (T,22,9), ball (T,4), y (T,), contested (T,), visible (T,)."""
    nodes = np.asarray(data['nodes'], np.float32)  # (T,2,11,8)
    f, gt = data['features'], np.asarray(data['gt'])
    T = len(nodes)
    frames = np.arange(T) * 5
    frames_c = np.clip(frames, 0, len(gt) - 1)
    sign = np.where(nodes[:, :, 0, 7] >= 0, 1., -1.)  # (T,2)
    px = np.where(sign[:, :, None] > 0, nodes[:, :, :, 0],
                  1. - nodes[:, :, :, 0]) * LENGTH
    py = nodes[:, :, :, 1] * WIDTH
    pvx = nodes[:, :, :, 2] * 10. * sign[:, :, None]
    pvy = nodes[:, :, :, 3] * 10.
    exists = nodes[:, :, :, 6] > 0
    bx = np.asarray(f['ball_x_m'], np.float64)[frames_c]
    by = np.asarray(f['ball_y_m'], np.float64)[frames_c]
    present = np.asarray(f['ball_present'], bool)[frames_c]
    bx = np.where(present, bx, LENGTH / 2.)
    by = np.where(present, by, WIDTH / 2.)
    rel_x = px - bx[:, None, None]
    rel_y = py - by[:, None, None]
    dist = np.hypot(rel_x, rel_y)
    players = np.stack([px / LENGTH, py / WIDTH, pvx / 10., pvy / 10.,
                        nodes[:, :, :, 4], dist / 20.,
                        rel_x / 20., rel_y / 20.,
                        exists.astype(np.float32)], axis=-1)
    players = players.reshape(T, 22, 9)
    is_home = np.zeros((T, 22, 1), np.float32)
    is_home[:, :11] = 1.
    players = np.concatenate([players, is_home], axis=-1).astype(np.float32)
    ball = np.stack([np.nan_to_num(bx / LENGTH), np.nan_to_num(by / WIDTH),
                     present.astype(np.float32),
                     np.asarray(f['ball_speed'], np.float64)[frames_c] / 10.], axis=-1)
    ball = np.nan_to_num(ball).astype(np.float32)
    y = np.where(gt[frames_c] == 'H', 1., np.where(gt[frames_c] == 'A', -1., 0.))
    live = np.isin(gt[frames_c], ['H', 'A'])
    switches = np.flatnonzero(live[:-1] & live[1:] & (gt[frames_c][:-1] != gt[frames_c][1:])) + 1
    contested = np.zeros(T, bool)
    for s in switches:  # within 5 node steps = 25f
        contested[max(0, s - 5):s + 6] = True
    return dict(players=players, ball=ball, y=y.astype(np.float32),
                live=live, contested=contested, visible=present & live,
                frames=frames)


class Net(nn.Module):
    def __init__(self, message=True, ball_dim=4):
        super().__init__()
        self.message = message
        self.ball_dim = ball_dim
        self.enc = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.msg = nn.Linear(32, 32)
        self.temporal = nn.Sequential(
            nn.Conv1d(96, 64, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=4, dilation=2), nn.ReLU())
        self.head = nn.Linear(64, 1)

    def embed(self, players, ball):
        return self._frame_embed(players, ball)

    def _frame_embed(self, players, ball):
        # players (B,T,22,10), ball (B,T,4)
        B, T = players.shape[:2]
        h = self.enc(players)  # (B,T,22,32)
        exists = players[..., 8:9]
        if self.message:
            pos = players[..., :2] * torch.tensor([LENGTH, WIDTH], device=players.device)
            d = torch.cdist(pos, pos)  # (B,T,22,22)
            w = torch.exp(-d / 10.) * exists.transpose(-2, -1) if False else torch.exp(-d / 10.)
            w = w * players[..., 8].unsqueeze(-2)
            w = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
            agg = torch.matmul(w, self.msg(h))
            h = h + agg
        home = (h[:, :, :11] * exists[:, :, :11]).sum(2) / exists[:, :, :11].sum(2).clamp(min=1.)
        away = (h[:, :, 11:] * exists[:, :, 11:]).sum(2) / exists[:, :, 11:].sum(2).clamp(min=1.)
        both = (h * exists).sum(2) / exists.sum(2).clamp(min=1.)
        ball_e = ball  # (B,T,ball_dim)
        ball_pad = torch.zeros(B, T, 32 - self.ball_dim, device=players.device)
        frame = torch.cat([home - away, both, ball, ball_pad], dim=-1)  # 96? home32+both32+ball4+28
        x = self.temporal(frame.transpose(1, 2)).transpose(1, 2)
        return x

    def forward(self, players, ball):
        return self.head(self._frame_embed(players, ball)).squeeze(-1)


def train_one(train, seed=0, epochs=6, message=True):
    torch.manual_seed(seed)
    model = Net(message=message, ball_dim=train[0]['ball'].shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xs = [torch.from_numpy(d['players'])[None] for d in train]
    bs = [torch.from_numpy(d['ball'])[None] for d in train]
    ys = [torch.from_numpy(d['y'])[None] for d in train]
    ws = []
    for d in train:
        w = np.where(d['live'], 1., 0.) * np.where(d['contested'], 3., 1.)
        ws.append(torch.from_numpy(w.astype(np.float32))[None])
    for ep in range(epochs):
        model.train()
        for x, b, y, w in zip(xs, bs, ys, ws):
            opt.zero_grad()
            logit = model(x, b)
            loss = (nn.functional.binary_cross_entropy_with_logits(
                logit, (y > 0).float(), reduction='none') * w).sum() / w.sum().clamp(min=1.)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def predict(model, d):
    model.eval()
    logit = model(torch.from_numpy(d['players'])[None],
                  torch.from_numpy(d['ball'])[None]).numpy()[0]
    return logit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--with-v23', action='store_true',
                        help='append v2.3 logit + nearest features to ball input')
    parser.add_argument('--out', default=str(REPORTS / 'gnn_probe.json'))
    args = parser.parse_args()
    data = {}
    for g in FIT:
        d = load_match(g)
        arr = match_arrays(d)
        # v2.3 logit sign on node frames
        ref = np.asarray(d['features'].get('game', ''))  # placeholder
        from crossfit_common import predict_version
        ref_labels = predict_version(d['features'], d['graded'], (2, 3))
        logits = ensemble_team_logits(d['features'], d['graded'], d['nodes'], ref_labels)
        arr['v23_sign'] = np.sign(np.nan_to_num(logits))[arr['frames'].clip(0, len(logits) - 1)]
        if args.with_v23:
            fc = arr['frames'].clip(0, len(logits) - 1)
            f = d['features']
            extra = np.stack([
                np.clip(np.nan_to_num(logits)[fc] / 3., -2., 2.),
                np.nan_to_num(np.asarray(f['nearest_team'], np.float64))[fc],
                np.clip((np.nan_to_num(np.asarray(f['min_dist_home'], np.float64), nan=20.)
                         - np.nan_to_num(np.asarray(f['min_dist_away'], np.float64), nan=20.))[fc] / 5., -2., 2.),
                (np.asarray(f['n_near_home'], np.float64) - np.asarray(f['n_near_away'], np.float64))[fc] / 3.,
            ], axis=-1).astype(np.float32)
            arr['ball'] = np.concatenate([arr['ball'], extra], axis=-1)
        data[g] = arr
        print('loaded', g, 'contested visible', int((arr['contested'] & arr['visible']).sum()), flush=True)
    rows = []
    for hold in FIT:
        train = [data[g] for g in FIT if g != hold]
        test = data[hold]
        mask = test['contested'] & test['visible']
        y = test['y'][mask]
        for name, message in (('gnn', True), ('pooled', False)):
            model = train_one(train, message=message, epochs=args.epochs)
            pred = np.sign(predict(model, test))[mask]
            acc = float(np.mean(pred == y))
            rows.append(dict(game=hold, model=name, contested_acc=acc, n=int(mask.sum())))
            print(hold, name, round(acc, 4), 'n', int(mask.sum()), flush=True)
        v23 = float(np.mean(test['v23_sign'][mask] == y))
        rows.append(dict(game=hold, model='v23_logit', contested_acc=v23, n=int(mask.sum())))
        print(hold, 'v23_logit', round(v23, 4), flush=True)
    summary = {}
    for name in ('gnn', 'pooled', 'v23_logit'):
        vals = [r['contested_acc'] for r in rows if r['model'] == name]
        summary[name] = float(np.mean(vals))
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps(dict(rows=rows, summary=summary), indent=2) + '\n')


if __name__ == '__main__':
    main()
