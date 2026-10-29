"""Predict step: features -> state machine -> submission CSV.

The one command that also validates and scores is predict_v1.bat (Command Prompt)
or predict_v1.sh (Mac Terminal, or Git Bash on Windows). This file is the predict step.

By default writes the next auto-incremented file under evaluation/results/:
  sm_v1.csv (1.0), sm_v1_1.csv … sm_v1_10.csv, then sm_v2_0.csv, …
Pass --out PATH to force a path and skip auto-versioning.

Needs the match bundle in ./data or MTAH_DATA_DIR. Defaults match kept sm_v2 path:
control=25, min_spell=50 (sandwich merge), set-piece cap=50, min_dead_frames=50,
restart_window=0, restart_reset_dead_frames=0. Smart Tagging soft priors on by default;
pass --no-smart-tagging to disable. --full-set-piece-dead restores T23 full-span D.
"""

from __future__ import print_function

import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from games import GAME  # noqa: E402
from features import extract_features, summarize  # noqa: E402
from smart_priors import (  # noqa: E402
    DEFAULT_SET_PIECE_DEAD_CAP_FRAMES,
    build_smart_priors,
    summarize_priors,
)
from state_machine import (  # noqa: E402
    DEFAULT_CONTROL_DIST_M,
    DEFAULT_CONTROL_FRAMES,
    DEFAULT_MIN_DEAD_FRAMES,
    DEFAULT_MIN_SPELL_FRAMES,
    DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    DEFAULT_OUT_MARGIN,
    DEFAULT_RESTART_CONTROL_FRAMES,
    DEFAULT_RESTART_ESTABLISH_FRAMES,
    DEFAULT_RESTART_RESET_DEAD_FRAMES,
    DEFAULT_RESTART_WINDOW_FRAMES,
    DEFAULT_STILL_NEAR_EDGE_FRAMES,
    DEFAULT_STILL_SPEED_M,
    label_possession,
    params_dict,
)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
MAX_MINOR = 10

# sm_v1.csv -> 1.0; sm_v1_1.csv -> 1.1; sm_v2_0.csv -> 2.0
_RE_MAJOR_ONLY = re.compile(r"^sm_v(\d+)\.csv$", re.IGNORECASE)
_RE_MAJOR_MINOR = re.compile(r"^sm_v(\d+)_(\d+)\.csv$", re.IGNORECASE)


def parse_sm_version(filename):
    """Return (major, minor) or None. minor is 0..10 inclusive when present."""
    name = os.path.basename(filename)
    m = _RE_MAJOR_MINOR.match(name)
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        if 0 <= minor <= MAX_MINOR:
            return major, minor
        return None
    m = _RE_MAJOR_ONLY.match(name)
    if m:
        return int(m.group(1)), 0
    return None


def format_sm_filename(major, minor):
    """1.0 -> sm_v1.csv; 1.2 -> sm_v1_2.csv; 2.0 -> sm_v2.csv."""
    if minor == 0:
        return "sm_v{}.csv".format(major)
    return "sm_v{}_{}.csv".format(major, minor)


def latest_sm_version(results_dir):
    """Max (major, minor) among sm_v*.csv in results_dir, or None."""
    if not os.path.isdir(results_dir):
        return None
    best = None
    for name in os.listdir(results_dir):
        parsed = parse_sm_version(name)
        if parsed is None:
            continue
        if best is None or parsed > best:
            best = parsed
    return best


def next_sm_version(major, minor):
    """Bump minor until 10, then major+1 with minor 0."""
    if minor < MAX_MINOR:
        return major, minor + 1
    return major + 1, 0


def resolve_auto_out_path(results_dir=RESULTS_DIR):
    """Next submission path after the latest sm_v* file (or sm_v1.csv if none)."""
    latest = latest_sm_version(results_dir)
    if latest is None:
        major, minor = 1, 0
    else:
        major, minor = next_sm_version(*latest)
    return os.path.join(results_dir, format_sm_filename(major, minor)), major, minor


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
    parser.add_argument(
        "--out", default=None,
        help="submission CSV path; default = next auto sm_v* under evaluation/results/",
    )
    parser.add_argument("--control-frames", type=int, default=DEFAULT_CONTROL_FRAMES)
    parser.add_argument("--control-dist-m", type=float, default=DEFAULT_CONTROL_DIST_M)
    parser.add_argument("--out-margin", type=float, default=DEFAULT_OUT_MARGIN)
    parser.add_argument("--min-spell-frames", type=int, default=DEFAULT_MIN_SPELL_FRAMES)
    parser.add_argument("--missing-near-edge-frames", type=int,
                        default=DEFAULT_MISSING_NEAR_EDGE_FRAMES)
    parser.add_argument("--still-near-edge-frames", type=int,
                        default=DEFAULT_STILL_NEAR_EDGE_FRAMES)
    parser.add_argument("--still-speed-m", type=float, default=DEFAULT_STILL_SPEED_M)
    parser.add_argument(
        "--restart-control-frames", type=int, default=DEFAULT_RESTART_CONTROL_FRAMES,
        help="opposing-switch frames required soon after leaving D (default {})".format(
            DEFAULT_RESTART_CONTROL_FRAMES
        ),
    )
    parser.add_argument(
        "--restart-window-frames", type=int, default=DEFAULT_RESTART_WINDOW_FRAMES,
        help="live frames after D where restart_control applies (default {}; 0=off)".format(
            DEFAULT_RESTART_WINDOW_FRAMES
        ),
    )
    parser.add_argument(
        "--min-dead-frames", type=int, default=DEFAULT_MIN_DEAD_FRAMES,
        help="fill D islands shorter than N when same team on both sides (default {}; 0=off)".format(
            DEFAULT_MIN_DEAD_FRAMES
        ),
    )
    parser.add_argument(
        "--restart-reset-dead-frames", type=int, default=DEFAULT_RESTART_RESET_DEAD_FRAMES,
        help="after D stretch >= N, clear last_live and re-establish (default {}; 0=off)".format(
            DEFAULT_RESTART_RESET_DEAD_FRAMES
        ),
    )
    parser.add_argument(
        "--restart-establish-frames", type=int, default=DEFAULT_RESTART_ESTABLISH_FRAMES,
        help="frames to establish possession after a long-D reset (default {})".format(
            DEFAULT_RESTART_ESTABLISH_FRAMES
        ),
    )
    parser.add_argument(
        "--no-smart-tagging", action="store_true",
        help="disable Smart Tagging soft priors",
    )
    parser.add_argument(
        "--set-piece-cap-frames", type=int, default=DEFAULT_SET_PIECE_DEAD_CAP_FRAMES,
        help="mark only first N frames of each SET PIECES tag as D (default {})".format(
            DEFAULT_SET_PIECE_DEAD_CAP_FRAMES
        ),
    )
    parser.add_argument(
        "--full-set-piece-dead", action="store_true",
        help="legacy T23: force D for the full SET PIECES tagged span",
    )
    parser.add_argument(
        "--no-phase-votes", action="store_true",
        help="disable weak phase-team votes (set-piece prior still applies)",
    )
    args = parser.parse_args()

    version = None
    if args.out is None:
        out_path, major, minor = resolve_auto_out_path(RESULTS_DIR)
        version = (major, minor)
    else:
        out_path = args.out
        version = parse_sm_version(out_path)

    print("T20 extract_features ...")
    features = extract_features(game=args.game)
    summarize(features)

    smart_priors = None
    if not args.no_smart_tagging:
        cap = None if args.full_set_piece_dead else args.set_piece_cap_frames
        print("smart_priors ...")
        smart_priors = build_smart_priors(
            n_frames=features["n_frames"],
            game=args.game,
            set_piece_dead_cap_frames=cap,
            use_phase_votes=not args.no_phase_votes,
        )
        summarize_priors(smart_priors)
    else:
        print("smart_priors: disabled (--no-smart-tagging)")

    params = params_dict(
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
        missing_near_edge_frames=args.missing_near_edge_frames,
        still_near_edge_frames=args.still_near_edge_frames,
        still_speed_m=args.still_speed_m,
        restart_control_frames=args.restart_control_frames,
        restart_window_frames=args.restart_window_frames,
        min_dead_frames=args.min_dead_frames,
        restart_reset_dead_frames=args.restart_reset_dead_frames,
        restart_establish_frames=args.restart_establish_frames,
    )
    params["smart_tagging"] = smart_priors is not None
    if smart_priors is not None:
        params["set_piece_dead_cap_frames"] = smart_priors.get("set_piece_dead_cap_frames")
        params["use_phase_votes"] = smart_priors.get("use_phase_votes")
    print("state_machine params: {}".format(params))

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
        restart_control_frames=args.restart_control_frames,
        restart_window_frames=args.restart_window_frames,
        min_dead_frames=args.min_dead_frames,
        restart_reset_dead_frames=args.restart_reset_dead_frames,
        restart_establish_frames=args.restart_establish_frames,
        smart_priors=smart_priors,
    )
    path = write_submission(out_path, labels)
    counts = {lab: int(np.sum(labels == lab)) for lab in ("H", "A", "D", "X")}
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    print("label counts: {}".format(counts))
    if version is not None:
        print("VERSION {}.{}".format(version[0], version[1]))
    print("SUBMISSION {}".format(rel))
    print("validate:  python evaluation/validate_submission.py {}".format(rel))
    print("score:     python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
