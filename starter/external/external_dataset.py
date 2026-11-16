"""Load external match features + labels for assess Track C (and similar).

Sportec and Metrica sample games only. No challenge data.
"""
from __future__ import print_function

import numpy as np

from motion_features import load_external_motion
from sportec_experiment import cached_sportec
from metrica_gt import build_metrica_gt
from fetch_metrica import DEFAULT_OUT
from metrica_features import extract_metrica_features


def cached_metrica_features(game, root=None):
    """Metrica features with a small on-disk cache under evaluation/reports."""
    import os
    root = root or DEFAULT_OUT
    cache = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "evaluation", "reports", "feature_cache", "metrica_{}.npz".format(game))
    if os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    feats = extract_metrica_features(game, root=root, pz_radius_m=1.0, dz_radius_m=1.0)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **{k: feats[k] for k in feats if k != "game"})
    return feats


def dataset(game):
    """Return (features, gt_labels, graded_mask) for an external match id."""
    if game.startswith("Sample_Game_"):
        feats = cached_metrica_features(game)
        truth = build_metrica_gt(game, feats["n_frames"], root=DEFAULT_OUT)
        graded = np.ones(feats["n_frames"], dtype=bool)
    else:
        feats, truth, graded = cached_sportec(game)
    if "player_motion" not in feats:
        feats["player_motion"] = load_external_motion(game)
    feats["game"] = game
    return feats, truth, graded
