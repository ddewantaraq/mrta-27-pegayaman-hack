"""Export both boundary-model heads and verify NumPy parity on varied lengths."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boundary_ownership_model import load_boundary_model
from numpy_set_temporal import load_parameters
from numpy_boundary_ownership import forward_dual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='evaluation/reports/boundary_ownership_v1.pt')
    parser.add_argument('--out',default='evaluation/reports/boundary_ownership_v1.npz')
    args=parser.parse_args(); torch.set_num_threads(2)
    net,mean,scale=load_boundary_model(args.source)
    p={key:value.numpy() for key,value in net.state_dict().items()}
    p['temporal.project.weight']=p['temporal.project.weight'][:,np.r_[np.arange(57),157,np.arange(57,157)]]
    p.update(mean=mean,scale=scale,layers=np.array(len(net.temporal.blocks)),baseline_prior=np.array(net.prior))
    np.savez_compressed(args.out,**p); p=load_parameters(args.out)
    rng=np.random.RandomState(39); errors=[0.,0.]
    for n in (1,17,257,1400):
        x=rng.randn(n,57).astype(np.float32); nodes=rng.rand(n,2,11,8).astype(np.float32)
        nodes[...,6]=nodes[...,6]>.3; context=rng.rand(n,8).astype(np.float32)
        base=rng.choice([-1.,0.,1.],n).astype(np.float32)
        with torch.inference_mode():
            expected=net(torch.from_numpy(x.T.copy())[None],torch.from_numpy(nodes)[None],
                         torch.from_numpy(context)[None],torch.from_numpy(base)[None])
        team,boundary=forward_dual(np.column_stack([x,base]),nodes,context,p)
        actual=(team+net.prior*base,boundary)
        for k in (0,1):
            target=expected[k][0].numpy()
            np.testing.assert_allclose(actual[k],target,rtol=2e-5,atol=2e-5)
            errors[k]=max(errors[k],float(np.max(np.abs(actual[k]-target))))
    print('Export parity max errors, team/boundary:',errors)


if __name__=='__main__':
    main()
