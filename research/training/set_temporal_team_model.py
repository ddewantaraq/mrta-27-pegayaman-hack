"""Permutation-invariant player-set encoder with temporal ownership context."""
import numpy as np
import torch
from torch import nn

from temporal_team_model import TemporalTeamNet, temporal_design
from ownership_context import ball_context


class SetTemporalTeamNet(nn.Module):
    def __init__(self, width=32, layers=6):
        super().__init__()
        self.player = nn.Sequential(nn.Linear(10, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU())
        self.temporal = TemporalTeamNet(input_dim=157, width=width, layers=layers)

    def forward(self, x, nodes, context):
        return self.temporal(self.forward_features(x, nodes, context))

    def forward_features(self, x, nodes, context):
        # nodes: B,T,team,player,8. Context is B,T,8 in source orientation.
        available = nodes[..., 6:7]
        sign = nodes[..., 7:8]
        previous = context[:, :, None, None, :2].expand(*nodes.shape[:-1], 2).clone()
        following = context[:, :, None, None, 2:4].expand(*nodes.shape[:-1], 2).clone()
        previous[..., :1] = torch.where(sign > 0, previous[..., :1], 1-previous[..., :1])
        following[..., :1] = torch.where(sign > 0, following[..., :1], 1-following[..., :1])
        position = nodes[..., :2]
        values = torch.cat([position-.5, nodes[..., 2:6], position-previous,
                            position-following], dim=-1)*available
        embedding = self.player(values)
        mean = (embedding*available).sum(dim=3)/available.sum(dim=3).clamp_min(1.)
        maximum = embedding.masked_fill(available == 0, -1e4).max(dim=3).values
        maximum = torch.where(available.sum(dim=3) > 0, maximum, torch.zeros_like(maximum))
        pooled = torch.cat([mean.flatten(2), maximum.flatten(2)], dim=-1).transpose(1, 2)
        extra = context[:, :, 4:].transpose(1, 2)
        return torch.cat([x, extra, pooled], dim=1)


def load_set_temporal_model(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    net = SetTemporalTeamNet(**data['architecture'])
    net.load_state_dict(data['state_dict'])
    net.eval()
    return net, data['mean'].numpy(), data['scale'].numpy()


def predict_set_logits(features, graded, player_nodes, loaded):
    net, mean, scale = loaded
    x = (temporal_design(features)-mean)/scale
    context = ball_context(features)
    n = len(x)
    period = np.asarray(features.get('period', np.zeros(n, np.int8)))
    result = np.zeros(n, np.float64)
    i = 0
    with torch.inference_mode():
        while i < n:
            if not graded[i]:
                i += 1
                continue
            a = i
            while i < n and graded[i] and period[i] == period[a]:
                i += 1
            b = i
            frames = np.arange(((a+4)//5)*5, b, 5)
            if len(frames) == 0:
                continue
            xx = torch.from_numpy(x[frames].T.copy()).unsqueeze(0)
            nnodes = torch.from_numpy(player_nodes[frames//5].copy()).unsqueeze(0)
            ctx = torch.from_numpy(context[frames].copy()).unsqueeze(0)
            swapped = xx.clone()
            swapped[:, :45] *= -1
            score = .5*(net(xx, nnodes, ctx)-net(swapped, nnodes[:, :, [1, 0]], ctx))
            result[a:b] = np.interp(np.arange(a, b), frames, score[0].numpy())
    return result
