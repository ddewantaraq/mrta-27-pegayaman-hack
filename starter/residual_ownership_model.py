"""Baseline-aware ownership corrections with a hybrid temporal decoder."""
from pathlib import Path
import numpy as np
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells

DEFAULT_RESIDUAL_MODEL_PATH = str(Path(__file__).resolve().parent/'models'/'residual_ownership_v1.npz')
DEFAULT_USE_RESIDUAL_OWNERSHIP = True


def apply_residual_ownership(features,reference,graded,nodes,model_path,minimum=18):
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded,bool)
    logits = predict_logits(features,graded,nodes,labels,load_parameters(model_path))
    live = graded & np.isin(labels,['H','A'])
    home = decode_dead_logits(logits,live,features.get('period'))
    visible = features['ball_present']
    home[visible] = logits[visible] >= 0
    labels[live] = np.where(home[live],'H','A')
    return _merge_short_spells(labels,graded,minimum)
