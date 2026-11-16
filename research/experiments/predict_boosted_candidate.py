"""Generate experimental boosted candidate from ATD only, with packaged JSON parity."""
import json
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import extract_features
from state_machine import graded_mask
from vc_possession import live_structure
from predict_v1 import write_submission
from boosted_dead_model import label_with_boosted_model
from motion_features import challenge_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-file', default='evaluation/reports/boosted_motion_sportec_v1.json')
    parser.add_argument('--out', default='evaluation/results/boosted_motion_candidate.csv')
    parser.add_argument('--no-post', action='store_true')
    args = parser.parse_args()
    features = extract_features()
    motion_cache = Path('evaluation/reports/feature_cache/challenge_motion_v1.npz')
    if motion_cache.exists():
        with np.load(motion_cache) as data:
            features['player_motion'] = data['motion'].copy()
    else:
        features['player_motion'] = challenge_motion()
    graded = graded_mask(features['n_frames'])
    labels = label_with_boosted_model(features, graded,
                                     model_path=args.model_file, postprocess=not args.no_post)
    path = args.out
    write_submission(path, labels)
    print(json.dumps(live_structure(labels, graded), indent=2))
    print(path)


if __name__ == '__main__':
    main()
