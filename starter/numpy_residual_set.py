"""NumPy inference for the baseline-aware correction network."""
import numpy as np
from numpy_set_temporal import forward, load_parameters
from team_control_model import control_design
from ownership_context import ball_context


def predict_logits(features, graded, nodes, reference, p):
    x = np.column_stack([control_design(features),features['player_motion']]).astype(np.float32)
    x = (x-p['mean'])/p['scale']
    baseline = np.where(reference == 'H',1.,np.where(reference == 'A',-1.,0.)).astype(np.float32)
    # Export reorders project columns so the added signed scalar follows x57.
    x = np.column_stack([x,baseline])
    context = ball_context(features)
    n = len(x); period = features.get('period',np.zeros(n,np.int8)); result = np.zeros(n)
    a = 0
    while a < n:
        if not graded[a]:
            a += 1
            continue
        b = a+1
        while b < n and graded[b] and period[b] == period[a]:
            b += 1
        ff = np.arange(((a+4)//5)*5,b,5)
        if len(ff):
            xx,nnodes,ctx = x[ff],nodes[ff//5],context[ff]
            swapped = xx.copy(); swapped[:,:45] *= -1; swapped[:,57] *= -1
            score = .5*(forward(xx,nnodes,ctx,p)-forward(swapped,nnodes[:,[1,0]],ctx,p))
            score += float(p['baseline_prior'])*baseline[ff]
            result[a:b] = np.interp(np.arange(a,b),ff,score)
        a = b
    return result
