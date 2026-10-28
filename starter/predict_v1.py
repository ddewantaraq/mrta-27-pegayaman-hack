"""T22 predict step: features -> state machine v1.1 -> submission CSV.

The one command that also validates and scores is predict_v1.bat (Command Prompt)
or predict_v1.sh (Mac Terminal, or Git Bash on Windows). This file is the predict step.

Writes evaluation/results/sm_v1_1.csv (199319 rows). Needs the match bundle in ./data
or MTAH_DATA_DIR. Defaults are the v1.1 thresholds logged in team/RESULTS.md.
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
    DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    DEFAULT_OUT_MARGIN,
    DEFAULT_STILL_NEAR_EDGE_FRAMES,
    DEFAULT_STILL_SPEED_M,
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
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "sm_v1_1.csv"))
    parser.add_argument("--control-frames", type=int, default=DEFAULT_CONTROL_FRAMES)
    parser.add_argument("--control-dist-m", type=float, default=DEFAULT_CONTROL_DIST_M)
    parser.add_argument("--out-margin", type=float, default=DEFAULT_OUT_MARGIN)
    parser.add_argument("--min-spell-frames", type=int, default=DEFAULT_MIN_SPELL_FRAMES)
    parser.add_argument("--missing-near-edge-frames", type=int,
                        default=DEFAULT_MISSING_NEAR_EDGE_FRAMES)
    parser.add_argument("--still-near-edge-frames", type=int,
                        default=DEFAULT_STILL_NEAR_EDGE_FRAMES)
    parser.add_argument("--still-speed-m", type=float, default=DEFAULT_STILL_SPEED_M)
    args = parser.parse_args()

    print("T20 extract_features ...")
    features = extract_features(game=args.game)
    summarize(features)

    params = params_dict(
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
        missing_near_edge_frames=args.missing_near_edge_frames,
        still_near_edge_frames=args.still_near_edge_frames,
        still_speed_m=args.still_speed_m,
    )
    print("T21 state_machine v1.1 params: {}".format(params))

    labels = label_possession(
        features,
        game=args.game,
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
        missing_near_edge_frames=args.missing_near_edge_frames,
        still_near_edge_frames=args.still_near_edge_frames,
        still_speed_m=args.still_speed_m,
    )
    path = write_submission(args.out, labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D", "X")}
    # SUBMISSION is the path the .bat and .sh validate. It is args.out, so changing
    # the --out default (or passing --out) changes the file those scripts check.
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    print("label counts: {}".format(counts))
    print("SUBMISSION {}".format(rel))
    print("validate:  python evaluation/validate_submission.py {}".format(rel))
    print("score:     python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
