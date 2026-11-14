"""Portable NumPy inference for numeric boosted live/dead trees.

Training uses scikit-learn 1.7.2; exported inference needs no sklearn/joblib.
"""
import json
from pathlib import Path

import numpy as np

from motion_dead_model import design_motion_features
from dead_ball_model import decode_dead_logits

DEFAULT_BOOSTED_MODEL_PATH = str(Path(__file__).resolve().parent/'models'/'boosted_dead_v1.json')
DEFAULT_USE_BOOSTED_DEAD = True


def label_with_boosted_model(features, graded=None, model_path=DEFAULT_BOOSTED_MODEL_PATH,
                             postprocess=True, **kwargs):
    from state_machine import (graded_mask, label_possession, splice_vc_on_visible,
                               _bridge_short_dead, _merge_short_spells,
                               DEFAULT_USE_DEAD_BRIDGE, DEFAULT_MAX_DEAD_BRIDGE_FRAMES,
                               DEFAULT_MIN_SPELL_FRAMES)
    from games import GAME
    if 'player_motion' not in features:
        raise ValueError('Boosted dead-ball inference requires ball-independent player_motion')
    kw = dict(kwargs)
    use_splice = kw.pop('vc_splice', True)
    if graded is None:
        graded = graded_mask(int(features['n_frames']), kw.get('game', GAME))
    graded = np.asarray(graded, bool)
    kw['use_geometric_dead'] = False
    runner = splice_vc_on_visible if use_splice else label_possession
    labels = runner(features, graded=graded, **kw)
    dead = predict_boosted_dead(features, load_boosted_model(model_path), graded)
    labels[dead & graded] = 'D'
    if postprocess:
        if kw.get('use_dead_bridge', DEFAULT_USE_DEAD_BRIDGE):
            labels = _bridge_short_dead(labels, graded, kw.get(
                'max_dead_bridge_frames', DEFAULT_MAX_DEAD_BRIDGE_FRAMES))
        minimum = kw.get('min_spell_frames', DEFAULT_MIN_SPELL_FRAMES)
        if minimum > 1:
            labels = _merge_short_spells(labels, graded, minimum)
    return labels


def load_boosted_model(path):
    with open(path, encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('feature_version') != 'motion_v1_173':
        raise ValueError('Unsupported boosted motion feature schema')
    if not np.isfinite(float(data['baseline'])):
        raise ValueError('Invalid boosted baseline score')
    arrays = []
    for tree in data['trees']:
        converted = {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
                     for k, v in tree.items()}
        n = len(converted['value'])
        if n == 0 or any(len(v) != n for v in converted.values()):
            raise ValueError('Mismatched tree node arrays')
        internal = converted['leaf'] == 0
        indices = np.arange(n)
        if (np.any(converted['feature'][internal] >= 173)
                or np.any(converted['feature'][internal] < 0)
                or np.any(converted['left'][internal] >= n)
                or np.any(converted['right'][internal] >= n)
                or np.any(converted['left'][internal] <= indices[internal])
                or np.any(converted['right'][internal] <= indices[internal])
                or not np.isfinite(converted['threshold']).all()
                or not np.isfinite(converted['value']).all()):
            raise ValueError('Invalid tree child or feature index')
        arrays.append(converted)
    return dict(baseline=float(data['baseline']), trees=arrays)


def boosted_logits(x, model):
    x = np.asarray(x)
    out = np.full(len(x), model['baseline'], dtype=np.float64)
    rows = np.arange(len(x))
    for tree in model['trees']:
        nodes = np.zeros(len(x), dtype=np.int32)
        for _ in range(len(tree['value'])):
            active = tree['leaf'][nodes] == 0
            if not np.any(active):
                break
            r = rows[active]
            node = nodes[active]
            values = x[r, tree['feature'][node]]
            left = np.where(np.isnan(values), tree['missing_left'][node] != 0,
                            values <= tree['threshold'][node])
            nodes[active] = np.where(left, tree['left'][node], tree['right'][node])
        else:
            raise ValueError('Tree traversal did not reach a leaf')
        out += tree['value'][nodes]
    return out


def predict_boosted_dead(features, model, graded):
    logits = boosted_logits(design_motion_features(features), model)
    return decode_dead_logits(logits, graded, features.get('period'))
