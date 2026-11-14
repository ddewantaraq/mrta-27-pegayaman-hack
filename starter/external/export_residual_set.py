"""Export numeric residual weights and check reordered-input parity."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from residual_set_temporal import load_residual_model
from numpy_set_temporal import load_parameters, forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='evaluation/reports/residual_set_v1.pt')
    parser.add_argument('--out',default='evaluation/reports/residual_set_v1.npz')
    args = parser.parse_args(); torch.set_num_threads(2)
    net,mean,scale = load_residual_model(args.source)
    p = {key:value.numpy() for key,value in net.state_dict().items()}
    order = np.r_[np.arange(57),157,np.arange(57,157)]
    p['temporal.project.weight'] = p['temporal.project.weight'][:,order]
    p.update(mean=mean,scale=scale,layers=np.array(len(net.temporal.blocks)),baseline_prior=np.array(net.prior))
    np.savez_compressed(args.out,**p); p = load_parameters(args.out)
    rng = np.random.RandomState(29); error = 0.
    for n in (1,17,257,1400):
        x = rng.randn(n,57).astype(np.float32)
        nodes = rng.rand(n,2,11,8).astype(np.float32); nodes[...,6] = nodes[...,6] > .3
        ctx = rng.rand(n,8).astype(np.float32); base = rng.choice([-1.,0.,1.],n).astype(np.float32)
        with torch.inference_mode():
            expected = net(torch.from_numpy(x.T.copy())[None],torch.from_numpy(nodes)[None],
                           torch.from_numpy(ctx)[None],torch.from_numpy(base)[None])[0].numpy()
        actual = forward(np.column_stack([x,base]),nodes,ctx,p)+net.prior*base
        np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-5)
        error = max(error,float(np.max(np.abs(actual-expected))))
    print('Residual export verified; maximum absolute logit difference:',error)


if __name__ == '__main__':
    main()
