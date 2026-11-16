"""Portable equal-logit ownership ensemble with a learned boundary decoder."""
from pathlib import Path
import numpy as np
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual, decode_boundary
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells

DEFAULT_BOUNDARY_MODEL_PATH = str(
    Path(__file__).resolve().parent / 'models' / 'boundary_ownership_v1.npz')
DEFAULT_USE_BOUNDARY_ENSEMBLE = True
DEFAULT_BOUNDARY_BLEND = 0.5
DEFAULT_BOUNDARY_SCALE = 1.0
DEFAULT_BOUNDARY_GATE = 'none'
DEFAULT_BOUNDARY_CONFIDENCE = 1.0


def apply_boundary_ensemble(features, reference, graded, nodes,
                            model_path=DEFAULT_BOUNDARY_MODEL_PATH, minimum=18,
                            blend=DEFAULT_BOUNDARY_BLEND,
                            boundary_scale=DEFAULT_BOUNDARY_SCALE,
                            gate=DEFAULT_BOUNDARY_GATE,
                            confidence_threshold=DEFAULT_BOUNDARY_CONFIDENCE):
    """Equal-logit ensemble; defaults reproduce sm_v2_3 exactly.

    gate:
      none — boundary Viterbi on live spans (v2.3)
      missing_only — boundary Viterbi only on missing-ball frames; v2.2
        constant-transition decode elsewhere
      confidence — boundary Viterbi only where |team signal| < threshold
    """
    if gate not in ('none', 'missing_only', 'confidence'):
        raise ValueError('Unknown boundary gate: {!r}'.format(gate))
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    old = predict_logits(features, graded, nodes, labels,
                         load_parameters(DEFAULT_RESIDUAL_MODEL_PATH))
    newer, boundary = predict_dual(features, graded, nodes, labels,
                                   load_parameters(model_path))
    blend = float(blend)
    signal = (1.0 - blend) * old + blend * newer
    live = graded & np.isin(labels, ['H', 'A'])
    present = np.asarray(features['ball_present'], bool)
    if gate == 'none':
        home = decode_boundary(
            signal, boundary, live, features.get('period'),
            boundary_scale=float(boundary_scale))
    elif gate == 'missing_only':
        # Viterbi only inside missing-ball runs (not full live spans).
        home = decode_dead_logits(signal, live, features.get('period'))
        missing_live = live & ~present
        home_b = decode_boundary(
            signal, boundary, missing_live, features.get('period'),
            boundary_scale=float(boundary_scale))
        home = home.copy()
        home[missing_live] = home_b[missing_live]
    else:
        home = decode_dead_logits(signal, live, features.get('period'))
        uncertain = live & (np.abs(signal) < float(confidence_threshold))
        home_b = decode_boundary(
            signal, boundary, uncertain, features.get('period'),
            boundary_scale=float(boundary_scale))
        home = home.copy()
        home[uncertain] = home_b[uncertain]
    home = home.copy()
    home[present] = signal[present] >= 0
    labels[live] = np.where(home[live], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)
