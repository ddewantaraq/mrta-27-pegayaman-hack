"""Ignored external-only caches of the frozen v2.2 labels and ownership logits."""
import hashlib
from pathlib import Path
import numpy as np

from frozen_predictions import cases
from player_node_data import load_player_nodes
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells


def decode_ownership(features,reference,graded,logits):
    labels = reference.copy()
    live = graded & np.isin(reference,['H','A'])
    home = decode_dead_logits(logits,live,features.get('period'))
    visible = features['ball_present']
    home[visible] = logits[visible] >= 0
    labels[live] = np.where(home[live],'H','A')
    return _merge_short_spells(labels,graded,18)


def approved_cases(game,seed=41):
    model_path = Path(DEFAULT_RESIDUAL_MODEL_PATH)
    signature = hashlib.sha256(model_path.read_bytes()+b'v2.2-hybrid-18').hexdigest()
    model = load_parameters(model_path)
    nodes = load_player_nodes(game)
    for sec,f,gt,graded,hide,reference in cases(game,seed=seed):
        cache = Path('evaluation/reports/prediction_cache')/('approved_v2_2_'+game+'_seed'+str(seed)+'_'+sec+'s.npz')
        labels,logits = None,None
        if cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                if str(data['signature']) == signature:
                    labels,logits = data['labels'].copy(),data['logits'].copy()
        if labels is None:
            logits = predict_logits(f,graded,nodes,reference,model)
            labels = np.asarray(decode_ownership(f,reference,graded,logits),dtype='<U1')
            np.savez_compressed(cache,labels=labels,logits=logits,signature=signature)
        yield sec,f,gt,graded,hide,reference,labels,logits
