"""Research candidate adding ball-free player movement to live/dead emissions.

Adaptation inspired by Lang et al. (2022), doi:10.1038/s41598-022-19948-1.
Aggregates replace fixed player identities to tolerate fragmented tracking.
"""
import numpy as np

from dead_ball_model import design_features, _mean_window, decode_dead_logits


def design_motion_features(features, include_ball=True):
    motion = np.asarray(features['player_motion'], dtype=np.float32)
    n = len(motion)
    idx = np.arange(n)
    parts = [design_features(features)] if include_ball else []
    for offset in (-100, -50, -20, -10, 0, 10, 20, 50, 100):
        parts.append(motion[np.clip(idx+offset, 0, n-1)])
    for radius in (50, 125):
        parts.append(np.column_stack([_mean_window(motion[:, k], radius)
                                     for k in range(motion.shape[1])]))
    return np.column_stack(parts).astype(np.float32)


def predict_motion_dead(features, model, graded, include_ball=True):
    x = design_motion_features(features, include_ball)
    logits = model['beta'][0]+((x-model['mean'])/model['scale'])@model['beta'][1:]
    return decode_dead_logits(logits, graded, features.get('period'))
