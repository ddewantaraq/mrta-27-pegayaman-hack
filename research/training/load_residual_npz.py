"""Rebuild a ResidualSetTemporalNet from the packaged NumPy residual export."""
import numpy as np
import torch
from residual_set_temporal import ResidualSetTemporalNet


def residual_state_dict_from_npz(path):
    """Invert the column reorder applied by export_residual_set.py."""
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    order = np.r_[np.arange(57), 157, np.arange(57, 157)]
    inverse = np.empty(158, dtype=int)
    inverse[order] = np.arange(158)
    weight = arrays['temporal.project.weight'][:, inverse]
    state = {}
    for key, value in arrays.items():
        if key in ('mean', 'scale', 'layers', 'baseline_prior'):
            continue
        if key == 'temporal.project.weight':
            value = weight
        state[key] = torch.from_numpy(np.asarray(value))
    mean = arrays['mean']
    scale = arrays['scale']
    prior = float(arrays['baseline_prior'])
    layers = int(arrays['layers'])
    return state, mean, scale, prior, layers


def load_residual_from_npz(path):
    state, mean, scale, prior, layers = residual_state_dict_from_npz(path)
    net = ResidualSetTemporalNet(width=32, layers=layers, prior=prior)
    missing, unexpected = net.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    net.eval()
    return net, mean, scale
