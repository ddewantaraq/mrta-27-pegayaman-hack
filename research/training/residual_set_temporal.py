"""Research correction network that explicitly conditions on the baseline."""
import numpy as np
import torch
from set_temporal_team_model import SetTemporalTeamNet, ball_context
from temporal_team_model import TemporalTeamNet, temporal_design


class ResidualSetTemporalNet(SetTemporalTeamNet):
    def __init__(self, width=32, layers=6, prior=2.1972245773362196):
        super().__init__(width, layers)
        self.temporal = TemporalTeamNet(input_dim=158, width=width, layers=layers)
        self.prior = prior

    def forward(self, x, nodes, context, baseline):
        features = self.forward_features(x, nodes, context)
        score = self.temporal(torch.cat([features, baseline[:, None]], dim=1))
        return score + self.prior*baseline


def load_residual_model(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    model = ResidualSetTemporalNet(**data['architecture'])
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, data['mean'].numpy(), data['scale'].numpy()


def predict_residual_logits(features, graded, nodes, reference, loaded):
    net, mean, scale = loaded
    x = (temporal_design(features)-mean)/scale
    context = ball_context(features)
    baseline = np.where(reference == 'H', 1., np.where(reference == 'A', -1., 0.)).astype(np.float32)
    n = len(x)
    period = features.get('period', np.zeros(n, np.int8))
    logits = np.zeros(n)
    a = 0
    with torch.inference_mode():
        while a < n:
            if not graded[a]:
                a += 1
                continue
            b = a+1
            while b < n and graded[b] and period[b] == period[a]:
                b += 1
            ff = np.arange(((a+4)//5)*5, b, 5)
            if len(ff):
                xx = torch.from_numpy(x[ff].T.copy())[None]
                nnodes = torch.from_numpy(nodes[ff//5].copy())[None]
                ctx = torch.from_numpy(context[ff].copy())[None]
                base = torch.from_numpy(baseline[ff])[None]
                swapped = xx.clone(); swapped[:, :45] *= -1
                score = .5*(net(xx,nnodes,ctx,base)-net(swapped,nnodes[:,:,[1,0]],ctx,-base))
                logits[a:b] = np.interp(np.arange(a,b), ff, score[0].numpy())
            a = b
    return logits
