"""Research-only temporal correction to boosted live/dead emissions."""
import numpy as np
import torch
from temporal_team_model import TemporalTeamNet
from set_temporal_team_model import SetTemporalTeamNet,ball_context
from motion_dead_model import design_motion_features
from boosted_dead_model import load_boosted_model,boosted_logits,DEFAULT_BOOSTED_MODEL_PATH


class TemporalDeadNet(SetTemporalTeamNet):
    def __init__(self,width=32,layers=6):
        super().__init__(width,layers)
        self.temporal = TemporalTeamNet(input_dim=274,width=width,layers=layers)
        torch.nn.init.zeros_(self.temporal.out.weight)
        torch.nn.init.zeros_(self.temporal.out.bias)

    def forward(self,x,nodes,context,baseline):
        encoded = self.forward_features(x,nodes,context)
        encoded = torch.cat([encoded,(baseline.clamp(-12,12)/4.)[:,None]],dim=1)
        return baseline+self.temporal(encoded)


def load_temporal_dead(path):
    data = torch.load(path,map_location='cpu',weights_only=True)
    net = TemporalDeadNet(**data['architecture']); net.load_state_dict(data['state_dict']); net.eval()
    return net,data['mean'].numpy(),data['scale'].numpy()


def predict_temporal_dead(features,graded,nodes,loaded):
    net,mean,scale = loaded
    design = design_motion_features(features)
    baseline = boosted_logits(design,load_boosted_model(DEFAULT_BOOSTED_MODEL_PATH)).astype(np.float32)
    x = (design-mean)/scale; context = ball_context(features)
    n = len(x); period = features.get('period',np.zeros(n,np.int8)); result = np.zeros(n)
    a = 0
    with torch.inference_mode():
        while a < n:
            if not graded[a]:
                a += 1; continue
            b = a+1
            while b<n and graded[b] and period[b]==period[a]:
                b += 1
            ff = np.arange(((a+4)//5)*5,b,5)
            if len(ff):
                xx = torch.from_numpy(x[ff].T.copy())[None]
                nnodes = torch.from_numpy(nodes[ff//5].copy())[None]
                ctx = torch.from_numpy(context[ff].copy())[None]
                base = torch.from_numpy(baseline[ff])[None]
                score = .5*(net(xx,nnodes,ctx,base)+net(xx,nnodes[:,:,[1,0]],ctx,base))
                result[a:b] = np.interp(np.arange(a,b),ff,score[0].numpy())
            a = b
    return result
