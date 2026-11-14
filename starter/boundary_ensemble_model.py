"""Portable equal-logit ownership ensemble with a learned boundary decoder."""
from pathlib import Path
import numpy as np
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual,decode_boundary
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from state_machine import _merge_short_spells

DEFAULT_BOUNDARY_MODEL_PATH=str(Path(__file__).resolve().parent/'models'/'boundary_ownership_v1.npz')
DEFAULT_USE_BOUNDARY_ENSEMBLE=True


def apply_boundary_ensemble(features,reference,graded,nodes,model_path=DEFAULT_BOUNDARY_MODEL_PATH,minimum=18):
    labels=np.asarray(reference).copy(); graded=np.asarray(graded,bool)
    old=predict_logits(features,graded,nodes,labels,load_parameters(DEFAULT_RESIDUAL_MODEL_PATH))
    newer,boundary=predict_dual(features,graded,nodes,labels,load_parameters(model_path))
    signal=.5*(old+newer); live=graded & np.isin(labels,['H','A'])
    home=decode_boundary(signal,boundary,live,features.get('period'))
    home[features['ball_present']]=signal[features['ball_present']]>=0
    labels[live]=np.where(home[live],'H','A')
    return _merge_short_spells(labels,graded,minimum)
