"""Apply a geometry-trained ownership head to existing live H/A frames."""
import numpy as np
from pathlib import Path

from numpy_set_temporal import load_parameters, predict_logits
from state_machine import _merge_short_spells

DEFAULT_SET_OWNERSHIP_PATH = str(Path(__file__).resolve().parent/'models'/'set_ownership_v1.npz')
DEFAULT_USE_SET_OWNERSHIP = False
DEFAULT_SET_OWNERSHIP_POLICY = 'run90_mean90'


def apply_set_ownership(features, labels, graded, nodes, model_path, minimum=18, policy='full'):
    labels = np.asarray(labels).copy()
    logits = predict_logits(features, graded, nodes, load_parameters(model_path))
    mask = np.asarray(graded, bool) & np.isin(labels, ['H', 'A'])
    if policy != 'full':
        from ownership_gates import replacement_mask
        mask = replacement_mask(features, labels, np.asarray(graded, bool), logits, policy)
    labels[mask] = np.where(logits[mask] >= 0, 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)
