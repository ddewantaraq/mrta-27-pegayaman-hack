"""Research ownership/boundary heads inspired by ASRF (WACV 2021)."""
import numpy as np
import torch
from residual_set_temporal import ResidualSetTemporalNet
from temporal_team_model import temporal_design
from ownership_context import ball_context


class BoundaryOwnershipNet(ResidualSetTemporalNet):
    def __init__(self,width=32,layers=6,prior=2.1972245773362196):
        super().__init__(width,layers,prior)
        self.boundary=torch.nn.Conv1d(width,1,1)
        torch.nn.init.zeros_(self.boundary.weight)
        torch.nn.init.constant_(self.boundary.bias,-2.)

    def forward(self,x,nodes,context,baseline):
        features=self.forward_features(x,nodes,context)
        features=torch.cat([features,baseline[:,None]],dim=1)
        hidden=self.temporal.blocks(torch.relu(self.temporal.project(features)))
        team=self.temporal.out(hidden).squeeze(1)+self.prior*baseline
        return team,self.boundary(hidden).squeeze(1)


def load_boundary_model(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    net=BoundaryOwnershipNet(**data['architecture']); net.load_state_dict(data['state_dict']); net.eval()
    return net,data['mean'].numpy(),data['scale'].numpy()


def predict_boundary_logits(features,graded,nodes,reference,loaded):
    net,mean,scale=loaded; x=(temporal_design(features)-mean)/scale
    context=ball_context(features)
    baseline=np.where(reference=='H',1.,np.where(reference=='A',-1.,0.)).astype(np.float32)
    n=len(x); period=features.get('period',np.zeros(n,np.int8)); team=np.zeros(n); boundary=np.zeros(n)
    a=0
    with torch.inference_mode():
        while a<n:
            if not graded[a]:
                a+=1; continue
            b=a+1
            while b<n and graded[b] and period[b]==period[a]:
                b+=1
            ff=np.arange(((a+4)//5)*5,b,5)
            if len(ff):
                xx=torch.from_numpy(x[ff].T.copy())[None]; nnodes=torch.from_numpy(nodes[ff//5].copy())[None]
                ctx=torch.from_numpy(context[ff].copy())[None]; base=torch.from_numpy(baseline[ff])[None]
                swapped=xx.clone(); swapped[:,:45]*=-1
                normal,edge=net(xx,nnodes,ctx,base)
                opposite,swapped_edge=net(swapped,nnodes[:,:,[1,0]],ctx,-base)
                team[a:b]=np.interp(np.arange(a,b),ff,(.5*(normal-opposite))[0].numpy())
                boundary[a:b]=np.interp(np.arange(a,b),ff,(.5*(edge+swapped_edge))[0].numpy())
            a=b
    return team,boundary


def decode_boundary(logits,boundary_logits,live,period=None,stride=5):
    """Variable transition costs from externally supervised boundary evidence."""
    n=len(logits); period=np.zeros(n,np.int8) if period is None else period
    result=np.zeros(n,bool); a=0
    while a<n:
        if not live[a]:
            a+=1; continue
        b=a+1
        while b<n and live[b] and period[b]==period[a]:
            b+=1
        ff=np.arange(a,b,stride); signal=logits[ff]
        probability=1/(1+np.exp(-np.clip(boundary_logits[ff]-np.log(10.),-30,30)))
        probability=np.clip(probability,.001,.3)
        emissions=np.column_stack([-np.logaddexp(0.,signal),-np.logaddexp(0.,-signal)])
        scores=emissions[0].copy(); back=np.zeros((len(ff),2),np.int8)
        for k in range(1,len(ff)):
            stay,switch=np.log1p(-probability[k]),np.log(probability[k])
            choices=scores[:,None]+np.array([[stay,switch],[switch,stay]])
            back[k]=choices.argmax(axis=0); scores=choices[back[k],np.arange(2)]+emissions[k]
        states=np.zeros(len(ff),np.int8); states[-1]=scores.argmax()
        for k in range(len(ff)-1,0,-1):
            states[k-1]=back[k,states[k]]
        result[a:b]=np.repeat(states,stride)[:b-a]
        a=b
    return result
