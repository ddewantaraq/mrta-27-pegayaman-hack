"""Two-state decoding with explicit evidence density across frame rates."""
import numpy as np


def decode_scaled(logits,live,period=None,stride=1,emission_scale=None,transition_prob=.01):
    n=len(logits)
    if stride<1 or not 0<transition_prob<.5:
        raise ValueError('Invalid decoder parameters')
    scale=stride/5. if emission_scale is None else float(emission_scale)
    if scale<=0 or not np.isfinite(scale):
        raise ValueError('Invalid emission scale')
    period=np.zeros(n,np.int8) if period is None else period
    result=np.zeros(n,bool); a=0
    transition=np.array([[np.log1p(-transition_prob),np.log(transition_prob)],
                         [np.log(transition_prob),np.log1p(-transition_prob)]])
    while a<n:
        if not live[a]:
            a+=1; continue
        b=a+1
        while b<n and live[b] and period[b]==period[a]:
            b+=1
        ff=np.arange(a,b,stride); signal=np.asarray(logits)[ff]
        emissions=scale*np.column_stack([-np.logaddexp(0.,signal),-np.logaddexp(0.,-signal)])
        scores=emissions[0].copy(); back=np.zeros((len(ff),2),np.int8)
        for k in range(1,len(ff)):
            choices=scores[:,None]+transition
            back[k]=choices.argmax(axis=0); scores=choices[back[k],np.arange(2)]+emissions[k]
        states=np.zeros(len(ff),np.int8); states[-1]=scores.argmax()
        for k in range(len(ff)-1,0,-1):
            states[k-1]=back[k,states[k]]
        result[a:b]=np.repeat(states,stride)[:b-a]
        a=b
    return result
