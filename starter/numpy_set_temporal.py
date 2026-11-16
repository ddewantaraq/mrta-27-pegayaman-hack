"""Portable NumPy inference for the experimental player-set temporal network."""
import numpy as np


def load_parameters(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _linear(x, parameters, name):
    return x @ parameters[name+'.weight'].T + parameters[name+'.bias']


def _conv(x, parameters, name, dilation=1):
    weights = parameters[name+'.weight']
    kernel = weights.shape[2]
    padding = dilation*(kernel//2)
    padded = np.pad(x, ((padding, padding), (0, 0)))
    result = np.broadcast_to(parameters[name+'.bias'], (len(x), weights.shape[0])).copy()
    for k in range(kernel):
        result += padded[k*dilation:k*dilation+len(x)] @ weights[:, :, k].T
    return result


def forward(x, nodes, context, p):
    """One sequence, T x features; eval mode, no stochastic operations."""
    available = nodes[..., 6:7]
    sign = nodes[..., 7:8]
    previous = np.broadcast_to(context[:, None, None, :2], nodes.shape[:-1]+(2,)).copy()
    following = np.broadcast_to(context[:, None, None, 2:4], nodes.shape[:-1]+(2,)).copy()
    previous[..., :1] = np.where(sign > 0, previous[..., :1], 1-previous[..., :1])
    following[..., :1] = np.where(sign > 0, following[..., :1], 1-following[..., :1])
    position = nodes[..., :2]
    values = np.concatenate([position-.5, nodes[..., 2:6], position-previous,
                             position-following], axis=-1)*available
    embed = np.maximum(_linear(values, p, 'player.0'), 0)
    embed = np.maximum(_linear(embed, p, 'player.2'), 0)
    count = available.sum(axis=2)
    mean = (embed*available).sum(axis=2)/np.maximum(count, 1)
    maximum = np.where(available > 0, embed, -1e4).max(axis=2)
    maximum = np.where(count > 0, maximum, 0)
    pooled = np.concatenate([mean.reshape(len(x), -1), maximum.reshape(len(x), -1)], axis=1)
    values = np.concatenate([x, context[:, 4:], pooled], axis=1)
    values = np.maximum(_conv(values, p, 'temporal.project'), 0)
    for layer in range(int(p['layers'])):
        prefix = 'temporal.blocks.'+str(layer)+'.net.'
        residual = np.maximum(_conv(values, p, prefix+'0', 2**layer), 0)
        residual = np.maximum(_conv(residual, p, prefix+'3', 2**layer), 0)
        values = np.maximum(values+residual, 0)
    return _conv(values, p, 'temporal.out')[:, 0]


def predict_logits(features, graded, player_nodes, parameters):
    from team_control_model import control_design
    # Importing this module must not require the optional training framework.
    from ownership_context import ball_context
    x = np.column_stack([control_design(features), features['player_motion']]).astype(np.float32)
    x = (x-parameters['mean'])/parameters['scale']
    context = ball_context(features)
    n = len(x)
    period = np.asarray(features.get('period', np.zeros(n, np.int8)))
    result = np.zeros(n, np.float64)
    i = 0
    while i < n:
        if not graded[i]:
            i += 1
            continue
        a = i
        while i < n and graded[i] and period[i] == period[a]:
            i += 1
        frames = np.arange(((a+4)//5)*5, i, 5)
        if not len(frames):
            continue
        xx, nodes, ctx = x[frames], player_nodes[frames//5], context[frames]
        swapped = xx.copy()
        swapped[:, :45] *= -1
        logits = .5*(forward(xx, nodes, ctx, parameters)-forward(swapped, nodes[:, [1, 0]], ctx, parameters))
        result[a:i] = np.interp(np.arange(a, i), frames, logits)
    return result
