"""NumPy-only joint ownership/boundary inference using frozen numeric kernels."""
import numpy as np
from numpy_set_temporal import forward
from team_control_model import control_design
from ownership_context import ball_context


def forward_dual(x,nodes,context,p):
    boundary=dict(p)
    boundary['temporal.out.weight']=p['boundary.weight']
    boundary['temporal.out.bias']=p['boundary.bias']
    return forward(x,nodes,context,p),forward(x,nodes,context,boundary)


def predict_dual(features,graded,nodes,reference,p):
    x=np.column_stack([control_design(features),features['player_motion']]).astype(np.float32)
    x=(x-p['mean'])/p['scale']; context=ball_context(features)
    base=np.where(reference=='H',1.,np.where(reference=='A',-1.,0.)).astype(np.float32)
    x=np.column_stack([x,base]); n=len(x)
    period=features.get('period',np.zeros(n,np.int8)); team=np.zeros(n); edge=np.zeros(n); a=0
    while a<n:
        if not graded[a]:
            a+=1; continue
        b=a+1
        while b<n and graded[b] and period[b]==period[a]:
            b+=1
        ff=np.arange(((a+4)//5)*5,b,5)
        if len(ff):
            xx,nnodes,ctx=x[ff],nodes[ff//5],context[ff]
            swapped=xx.copy(); swapped[:,:45]*=-1; swapped[:,57]*=-1
            normal,boundary=forward_dual(xx,nnodes,ctx,p)
            opposite,swapped_boundary=forward_dual(swapped,nnodes[:,[1,0]],ctx,p)
            score=.5*(normal-opposite)+float(p['baseline_prior'])*base[ff]
            team[a:b]=np.interp(np.arange(a,b),ff,score)
            edge[a:b]=np.interp(np.arange(a,b),ff,.5*(boundary+swapped_boundary))
        a=b
    return team,edge


def decode_boundary(logits,boundary_logits,live,period=None,stride=5):
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
        result[a:b]=np.repeat(states,stride)[:b-a]; a=b
    return result
