"""Export numeric parameters and verify PyTorch/NumPy forward parity."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from set_temporal_team_model import load_set_temporal_model
from numpy_set_temporal import load_parameters, forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='evaluation/reports/set_temporal_v1.pt')
    parser.add_argument('--out', default='evaluation/reports/set_temporal_v1.npz')
    args = parser.parse_args()
    torch.set_num_threads(2)
    net, mean, scale = load_set_temporal_model(args.source)
    parameters = {key: value.numpy() for key, value in net.state_dict().items()}
    parameters.update(mean=mean, scale=scale, layers=np.array(len(net.temporal.blocks)))
    np.savez_compressed(args.out, **parameters)
    p = load_parameters(args.out)
    rng = np.random.RandomState(19)
    maximum_error = 0.
    for n in (1, 17, 257, 1400):
        x = rng.randn(n, 57).astype(np.float32)
        nodes = rng.randn(n, 2, 11, 8).astype(np.float32)
        nodes[..., 6] = rng.rand(n, 2, 11) > .2
        nodes[..., 7] = np.where(rng.rand(n, 2, 11) > .5, 1., -1.)
        nodes[n//2:, :, 5:, 6] = 0
        context = rng.rand(n, 8).astype(np.float32)
        with torch.inference_mode():
            expected = net(torch.from_numpy(x.T.copy())[None], torch.from_numpy(nodes)[None],
                           torch.from_numpy(context)[None])[0].numpy()
        actual = forward(x, nodes, context, p)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual-expected))))
    print('Numeric export verified; maximum absolute logit error:', maximum_error)


if __name__ == '__main__':
    main()
