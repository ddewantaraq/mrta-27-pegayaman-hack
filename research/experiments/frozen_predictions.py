"""Cache immutable reference-model labels for repeated external experiments.

All caches are ignored and contain external-match data only. Model parameter
hashes invalidate a cache when a packaged reference changes.
"""
import hashlib
import os
from pathlib import Path

import numpy as np

from eval_external import kwargs_for_version, run_labeller
from dropout import apply_dropout
from sportec_experiment import block_dropout
from motion_dead_experiment import dataset


def cases(game, version=(1, 10), seed=41):
    f, gt, graded = dataset(game)
    kw = kwargs_for_version(*version)
    model = Path(kw['boosted_dead_model']) if 'boosted_dead_model' in kw else Path(kw['learned_dead_model'])
    signature = hashlib.sha256(model.read_bytes()).hexdigest()
    directory = Path('evaluation/reports/prediction_cache')
    directory.mkdir(parents=True, exist_ok=True)
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(f['n_frames'], sec, seed=seed)
        dropped = apply_dropout(f, hide)
        path = directory/('{}_v{}_{}_seed{}_{}s.npz'.format(game, *version, seed, sec))
        pred = None
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if str(data['signature']) == signature:
                        pred = data['labels'].astype('<U1')
            except (ValueError, KeyError, OSError):
                # Old object-dtype caches are regenerated without unpickling.
                pred = None
        if pred is None:
            pred = np.asarray(run_labeller(dropped, graded, kw), dtype='<U1')
            temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, labels=pred, signature=signature)
            temporary.replace(path)
        yield str(sec), dropped, gt, graded, hide | ~f['ball_present'], pred
