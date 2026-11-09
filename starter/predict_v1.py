"""Predict step: features -> state machine -> submission CSV.

The one command that also validates and scores is predict_v1.bat (Command Prompt)
or predict_v1.sh (Mac Terminal, or Git Bash on Windows). This file is the predict step.

By default writes the next auto-incremented file under evaluation/results/:
  sm_v1.csv (1.0), sm_v1_1.csv … sm_v1_10.csv, then sm_v2.csv, …
Pass --out PATH to force a path and skip auto-versioning.

Needs the match bundle in ./data or MTAH_DATA_DIR.
Defaults = compliant sm_v1_6 (sm_v1_5 + half-time clear, D before 2nd
kickoff, seed opposite kickoff team). sm_v1_5: backdate + gap-bridge mid;
Smart Tagging off. Archive sm_v1_4: --smart-tagging. Rollback kickoff:
--no-half-kickoff-reset.
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
from features import (  # noqa: E402
    DEFAULT_CLUSTER_RADIUS_M,
    extract_features,
    summarize,
)
from vc_possession import (  # noqa: E402
    DEFAULT_DZ_RADIUS_M,
    DEFAULT_PZ_RADIUS_M,
)
from smart_priors import (  # noqa: E402
    DEFAULT_SET_PIECE_DEAD_CAP_FRAMES,
    build_smart_priors,
    summarize_priors,
)
from state_machine import (  # noqa: E402
    DEFAULT_BACKDATE_SWITCH,
    DEFAULT_CONTEST_DIST_M,
    DEFAULT_CONTROL_DIST_M,
    DEFAULT_CONTROL_FRAMES,
    DEFAULT_GAP_BRIDGE,
    DEFAULT_HALF_KICKOFF_RESET,
    DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
    DEFAULT_LONG_GAP_DEAD_FRAMES,
    DEFAULT_MIN_SPELL_FRAMES,
    DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    DEFAULT_OUT_MARGIN,
    DEFAULT_STILL_NEAR_EDGE_FRAMES,
    DEFAULT_STILL_SPEED_M,
    DEFAULT_USE_MAJORITY_NEAR,
    DEFAULT_USE_SMART_TAGGING,
    DEFAULT_USE_VC_POSSESSION,
    DEFAULT_VC_BA_THRESHOLD_DEG,
    DEFAULT_VC_BV_THRESHOLD,
    label_possession,
    params_dict,
)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
MAX_MINOR = 10

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
        "--keep-last-live-on-dead",
        dest="keep_last_live_on_dead",
        action="store_true",
        default=DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
        help="freeze last_live through D (sm_v1_1 rule); default off for sm_v1",
    )
    parser.add_argument(
        "--clear-last-live-on-dead",
        dest="keep_last_live_on_dead",
        action="store_false",
        help="clear last_live on D (sm_v1 behaviour)",
    )
    parser.add_argument(
        "--cluster-radius-m", type=float, default=DEFAULT_CLUSTER_RADIUS_M,
        help="metres: players inside this radius count toward majority (default 3.0)",
    )
    parser.add_argument(
        "--contest-dist-m", type=float, default=DEFAULT_CONTEST_DIST_M,
        help="metres: |d_home-d_away| <= this allows majority to override nearest",
    )
    parser.add_argument(
        "--majority-near",
        dest="use_majority_near",
        action="store_true",
        default=DEFAULT_USE_MAJORITY_NEAR,
        help="contested majority-near-ball override (sm_v1_3 default on)",
    )
    parser.add_argument(
        "--no-majority-near",
        dest="use_majority_near",
        action="store_false",
        help="disable majority override; pure nearest (sm_v1_2 cand)",
    )
    parser.add_argument(
        "--smart-tagging",
        dest="use_smart_tagging",
        action="store_true",
        default=DEFAULT_USE_SMART_TAGGING,
        help="soft SET PIECES -> D (archived sm_v1_4; default off in sm_v1_5)",
    )
    parser.add_argument(
        "--no-smart-tagging",
        dest="use_smart_tagging",
        action="store_false",
        help="disable Smart Tagging priors (sm_v1_5 / sm_v1_3 path)",
    )
    parser.add_argument(
        "--set-piece-dead-cap-frames", type=int,
        default=DEFAULT_SET_PIECE_DEAD_CAP_FRAMES,
        help="first N frames of each SET PIECES event -> D when ball visible",
    )
    parser.add_argument(
        "--full-set-piece-dead",
        action="store_true",
        help="archive/debug: force D on full SET PIECES span (not compliant default)",
    )
    parser.add_argument(
        "--backdate-switch",
        dest="backdate_switch",
        action="store_true",
        default=DEFAULT_BACKDATE_SWITCH,
        help="relabel pending frames to new team once switch confirms",
    )
    parser.add_argument(
        "--no-backdate-switch",
        dest="backdate_switch",
        action="store_false",
        help="keep lagging last_live through the pending window",
    )
    parser.add_argument(
        "--gap-bridge",
        choices=("off", "start", "mid", "end"),
        default=DEFAULT_GAP_BRIDGE,
        help="place H/A switch inside a live missing gap when teams disagree",
    )
    parser.add_argument(
        "--long-gap-dead-frames", type=int,
        default=DEFAULT_LONG_GAP_DEAD_FRAMES,
        help="missing streak >= N frames -> D (0 = off)",
    )
    parser.add_argument(
        "--half-kickoff-reset",
        dest="half_kickoff_reset",
        action="store_true",
        default=DEFAULT_HALF_KICKOFF_RESET,
        help="clear last_live at HT; D before 2nd KO; seed opposite team (sm_v1_6)",
    )
    parser.add_argument(
        "--no-half-kickoff-reset",
        dest="half_kickoff_reset",
        action="store_false",
        help="disable half-kickoff reset (sm_v1_5 path)",
    )
    parser.add_argument(
        "--vc-possession",
        dest="use_vc_possession",
        action="store_true",
        default=DEFAULT_USE_VC_POSSESSION,
        help="Vidal-Codina gain as the control candidate (off unless it beats v1.5)",
    )
    parser.add_argument(
        "--no-vc-possession",
        dest="use_vc_possession",
        action="store_false",
        help="nearest / majority candidate (sm_v1_6 default)",
    )
    parser.add_argument("--pz-radius-m", type=float, default=DEFAULT_PZ_RADIUS_M,
                        help="Vidal-Codina possession-zone radius in metres")
    parser.add_argument("--dz-radius-m", type=float, default=DEFAULT_DZ_RADIUS_M,
                        help="Vidal-Codina duel-zone radius in metres")
    parser.add_argument("--vc-bv-threshold", type=float, default=DEFAULT_VC_BV_THRESHOLD,
                        help="m/s ball-speed change that counts as a gain")
    parser.add_argument("--vc-ba-threshold-deg", type=float,
                        default=DEFAULT_VC_BA_THRESHOLD_DEG,
                        help="degrees of ball-direction change that counts as a gain")
    args = parser.parse_args()

    version = None
    if args.out is None:
        out_path, major, minor = resolve_auto_out_path(RESULTS_DIR)
        version = (major, minor)
    else:
        out_path = args.out
        version = parse_sm_version(out_path)

    print("T20 extract_features ...")
    features = extract_features(
        game=args.game,
        cluster_radius_m=args.cluster_radius_m,
        pz_radius_m=args.pz_radius_m,
        dz_radius_m=args.dz_radius_m,
    )
    summarize(features)

    set_piece_dead = None
    cap = None if args.full_set_piece_dead else args.set_piece_dead_cap_frames
    if args.use_smart_tagging:
        priors = build_smart_priors(
            n_frames=features["n_frames"],
            game=args.game,
            set_piece_dead_cap_frames=cap,
            use_phase_votes=False,
            ball_present=None if args.full_set_piece_dead else features["ball_present"],
        )
        summarize_priors(priors)
        set_piece_dead = priors["set_piece_dead"]

    params = params_dict(
        control_frames=args.control_frames,
        control_dist_m=args.control_dist_m,
        out_margin=args.out_margin,
        min_spell_frames=args.min_spell_frames,
        missing_near_edge_frames=args.missing_near_edge_frames,
        still_near_edge_frames=args.still_near_edge_frames,
        still_speed_m=args.still_speed_m,
        keep_last_live_on_dead=args.keep_last_live_on_dead,
        use_majority_near=args.use_majority_near,
        contest_dist_m=args.contest_dist_m,
        use_smart_tagging=args.use_smart_tagging,
        set_piece_dead_cap_frames=cap,
        backdate_switch=args.backdate_switch,
        gap_bridge=args.gap_bridge,
        long_gap_dead_frames=args.long_gap_dead_frames,
        half_kickoff_reset=args.half_kickoff_reset,
        use_vc_possession=args.use_vc_possession,
        vc_bv_threshold=args.vc_bv_threshold,
        vc_ba_threshold_deg=args.vc_ba_threshold_deg,
    )
    print("state_machine params: {}".format(params))
    print("pz_radius_m: {}  dz_radius_m: {}".format(
        args.pz_radius_m, args.dz_radius_m))
    print("cluster_radius_m: {}".format(args.cluster_radius_m))

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
        keep_last_live_on_dead=args.keep_last_live_on_dead,
        use_majority_near=args.use_majority_near,
        contest_dist_m=args.contest_dist_m,
        set_piece_dead=set_piece_dead,
        backdate_switch=args.backdate_switch,
        gap_bridge=args.gap_bridge,
        long_gap_dead_frames=args.long_gap_dead_frames,
        half_kickoff_reset=args.half_kickoff_reset,
        use_vc_possession=args.use_vc_possession,
        vc_bv_threshold=args.vc_bv_threshold,
        vc_ba_threshold_deg=args.vc_ba_threshold_deg,
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
