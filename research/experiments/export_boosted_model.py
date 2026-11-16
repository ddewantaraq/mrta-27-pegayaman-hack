"""Export a locally trained numeric HGB model and verify NumPy parity."""
import json
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boosted_motion_experiment import sportec_training_data, training_data, REPORTS
from boosted_dead_model import load_boosted_model, boosted_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed-training', action='store_true')
    parser.add_argument('--calibrate-prior', action='store_true')
    args = parser.parse_args()
    estimator = joblib.load(REPORTS/('boosted_motion_v1.joblib' if args.mixed_training
                                    else 'boosted_motion_sportec_v1.joblib'))
    x, y = training_data() if args.mixed_training else sportec_training_data()
    correction = float(np.log(y.mean()/(1-y.mean()))) if args.calibrate_prior else 0.
    trees = []
    for iteration in estimator._predictors:
        nodes = iteration[0].nodes
        if np.any(nodes['is_categorical']):
            raise ValueError('Only numeric trees can be exported')
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf', missing_left='missing_go_to_left')
        trees.append({key: nodes[source].tolist() for key, source in fields.items()})
    payload = dict(feature_version='motion_v1_173', baseline=float(estimator._baseline_prediction[0, 0])+correction,
                   prior_log_odds_correction=correction,
                   training_matches=['J03WOH', 'Sample_Game_1'] if args.mixed_training else
                                    ['J03WOH', 'J03WN1', 'J03WOY'],
                   source='Sportec Open DFL, Bassek et al. 2025, CC BY 4.0',
                   source_doi='10.1038/s41597-025-04505-y',
                   metrica_source='https://github.com/metrica-sports/sample-data' if args.mixed_training else None,
                   metrica_reference_version='historical_v1' if args.mixed_training else None,
                   sklearn_version='1.7.2', trees=trees)
    name = 'boosted_motion_mixed_v1' if args.mixed_training else 'boosted_motion_sportec_v1'
    name += '_prior' if args.calibrate_prior else ''
    path = REPORTS/(name+'.json')
    path.write_text(json.dumps(payload, separators=(',', ':'))+'\n', encoding='utf-8')
    x = x[::10]
    # Include missing-feature routing, although production features are finite.
    probe = x.copy()
    probe[::17, 0] = np.nan
    with threadpool_limits(limits=4):
        reference = estimator.decision_function(probe)+correction
    actual = boosted_logits(probe, load_boosted_model(path))
    np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)
    print('Exported and verified', len(probe), 'samples;', path, flush=True)


if __name__ == '__main__':
    main()
