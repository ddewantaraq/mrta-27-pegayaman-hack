"""T21 wire-up: features -> state machine -> submission CSV.

    .venv/bin/python starter/predict_v1.py
    .venv/bin/python evaluation/validate_submission.py evaluation/results/sm_v1.csv
    .venv/bin/python evaluation/run_evaluation.py
"""

from __future__ import print_function

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME  # noqa: E402
from features import extract_features, summarize  # noqa: E402
from state_machine import (  # noqa: E402
    DEFAULT_CONTROL_DIST_M,
    DEFAULT_CONTROL_FRAMES,
    DEFAULT_MIN_SPELL_FRAMES,
    DEFAULT_OUT_MARGIN,
    label_possession,
    params_dict,
)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "sm_v1.csv"))
    parser.add_argument("--control-frames", type=int, default=DEFAULT_CONTROL_FRAMES)
    parser.add_argument("--control-dist-m", type=float, default=DEFAULT_CONTROL_DIST_M)
    parser.add_argument("--out-margin", type=float, default=DEFAULT_OUT_MARGIN)
    parser.add_argument("--min-spell-frames", type=int, default=DEFAULT_MIN_SPELL_FRAMES)
    args = parser.parse_args()

    print("T20 extract_features ...")
    features = extract_features(game=args.game)
    summarize(features)

    params = params_dict(
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
    )
    print("T21 state_machine params: {}".format(params))

    labels = label_possession(
        features,
        game=args.game,
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
    )
    path = write_submission(args.out, labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D", "X")}
    print("label counts: {}".format(counts))
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("validate:  python evaluation/validate_submission.py {}".format(
        os.path.relpath(path, ROOT)))
    print("score:     python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
