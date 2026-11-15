"""Research 12.5 Hz residual correction to the frozen 5 Hz ownership teacher."""
import numpy as np
import torch
from residual_set_temporal import ResidualSetTemporalNet
from temporal_team_model import temporal_design
from ownership_context import ball_context


class HighResolutionOwnershipNet(ResidualSetTemporalNet):
    def __init__(self,width=32,layers=7):
        super().__init__(width,layers,prior=1.)
        torch.nn.init.zeros_(self.temporal.out.weight)
        torch.nn.init.zeros_(self.temporal.out.bias)
        torch.nn.init.zeros_(self.temporal.blocks[-1].net[3].weight)
        torch.nn.init.zeros_(self.temporal.blocks[-1].net[3].bias)

    def forward(self,x,nodes,context,teacher):
        features=self.forward_features(x,nodes,context)
        inputs=torch.cat([features,(teacher.clamp(-12,12)/4.)[:,None]],dim=1)
        return teacher+self.temporal(inputs)


def load_highres_model(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    net=HighResolutionOwnershipNet(**data['architecture']); net.load_state_dict(data['state_dict']); net.eval()
    return net,data['mean'].numpy(),data['scale'].numpy()


def predict_highres(features,graded,nodes,teacher_logits,loaded,stride=2):
    net,mean,scale=loaded; x=(temporal_design(features)-mean)/scale; ctx=ball_context(features)
    n=len(x); period=features.get('period',np.zeros(n,np.int8)); result=np.zeros(n); a=0
    with torch.inference_mode():
        while a<n:
            if not graded[a]:
                a+=1; continue
            b=a+1
            while b<n and graded[b] and period[b]==period[a]:
                b+=1
            ff=np.arange(((a+stride-1)//stride)*stride,b,stride)
            if len(ff):
                xx=torch.from_numpy(x[ff].T.copy())[None]
                nnodes=torch.from_numpy(nodes[ff//stride].copy())[None]
                cc=torch.from_numpy(ctx[ff].copy())[None]
                base=torch.from_numpy(np.asarray(teacher_logits[ff],np.float32))[None]
                swapped=xx.clone(); swapped[:,:45]*=-1
                score=.5*(net(xx,nnodes,cc,base)-net(swapped,nnodes[:,:,[1,0]],cc,-base))
                result[a:b]=np.interp(np.arange(a,b),ff,score[0].numpy())
            a=b
    return result
