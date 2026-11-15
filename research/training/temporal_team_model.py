"""Research-only temporal ownership network; no challenge annotations accepted.

Dilated residual temporal convolutions and log-probability smoothing are
motivated by MS-TCN (Abu Farha & Gall, CVPR 2019). This is an adaptation,
not a reproduction of their multi-stage architecture or reported scores.
"""
import numpy as np
import torch
from torch import nn

from team_control_model import control_design
from dead_ball_model import decode_dead_logits


def temporal_design(features):
    return np.column_stack([control_design(features), features['player_motion']]).astype(np.float32)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.1),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(), nn.Dropout(.1))

    def forward(self, x):
        return torch.relu(x+self.net(x))


class TemporalTeamNet(nn.Module):
    def __init__(self, input_dim=57, width=32, layers=6):
        super().__init__()
        self.project = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*[ResidualTemporalBlock(width, 2**i) for i in range(layers)])
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        return self.out(self.blocks(torch.relu(self.project(x)))).squeeze(1)


def load_temporal_model(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    net = TemporalTeamNet(**payload['architecture'])
    net.load_state_dict(payload['state_dict'])
    net.eval()
    return net, payload['mean'].numpy(), payload['scale'].numpy()


def predict_temporal_logits(features, graded, loaded):
    net, mean, scale = loaded
    x = (temporal_design(features)-mean)/scale
    n = len(x)
    periods = np.asarray(features.get('period', np.zeros(n, dtype=np.int8)))
    logits = np.zeros(n, dtype=np.float64)
    i = 0
    with torch.inference_mode():
        while i < n:
            if not graded[i]:
                i += 1
                continue
            a = i
            while i < n and graded[i] and periods[i] == periods[a]:
                i += 1
            b = i
            frames = np.arange(a, b, 5)
            tensor = torch.from_numpy(x[frames].T.copy()).unsqueeze(0)
            swapped = tensor.clone()
            swapped[:, :45] *= -1
            # Exact home/away equivariance, independent of training imbalance.
            score = .5*(net(tensor)-net(swapped))
            logits[a:b] = np.interp(np.arange(a, b), frames, score[0].numpy())
    return logits


def temporal_teams(features, graded, loaded, smooth=True):
    logits = predict_temporal_logits(features, graded, loaded)
    home = decode_dead_logits(logits, graded, features.get('period')) if smooth else logits >= 0
    return np.where(home, 'H', 'A')
