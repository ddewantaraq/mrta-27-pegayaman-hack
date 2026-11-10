"""Predict step: features -> state machine -> submission CSV.

The one command that also validates and scores is predict_v1.bat (Command Prompt)
or predict_v1.sh (Mac Terminal, or Git Bash on Windows). This file is the predict step.

By default writes the next file after evaluation/results/LATEST (approved best):
  bump minor until 10, then major+1; prefer bare sm_v*.csv; add _compliant
  only when that bare name already exists. Pass --out PATH to force a path.

Needs the match bundle in ./data or MTAH_DATA_DIR.
Defaults = compliant sm_v1_7 (sm_v1_6 + Vidal-Codina gain where the ball is
visible; sm_v1_6 labels where it is missing). Rollback splice:
--no-vc-splice. Rollback kickoff: --no-half-kickoff-reset.
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
from features import (  # noqa: E402
    DEFAULT_CLUSTER_RADIUS_M,
    extract_features,
    summarize,
)
from model_latest import (  # noqa: E402
    RESULTS_DIR,
    next_out_path,
    parse_sm_version,
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
    DEFAULT_VC_GAP,
    DEFAULT_VC_SPLICE,
    label_possession,
    params_dict,
    splice_vc_on_visible,
)


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
        help="submission CSV path; default = next after LATEST (bare, or _compliant if bare exists)",
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
    parser.add_argument(
        "--vc-gap",
        choices=("off", "reappear", "edge"),
        default=DEFAULT_VC_GAP,
        help="after a live gap: off, nearby-team reappear, or reappear plus direction change",
    )
    parser.add_argument(
        "--vc-splice",
        dest="vc_splice",
        action="store_true",
        default=DEFAULT_VC_SPLICE,
        help="VC label on visible frames, base model on missing frames (sm_v1_7)",
    )
    parser.add_argument(
        "--no-vc-splice",
        dest="vc_splice",
        action="store_false",
        help="one state machine only (sm_v1_6 when kickoff reset is on)",
    )
    args = parser.parse_args()

    version = None
    if args.out is None:
        out_path, major, minor = next_out_path(RESULTS_DIR)
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
        vc_gap=args.vc_gap,
    )
    params["vc_splice"] = args.vc_splice
    print("state_machine params: {}".format(params))
    print("pz_radius_m: {}  dz_radius_m: {}".format(
        args.pz_radius_m, args.dz_radius_m))
    print("cluster_radius_m: {}".format(args.cluster_radius_m))

    label_kwargs = dict(
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
        vc_gap=args.vc_gap,
    )
    if args.vc_splice:
        labels = splice_vc_on_visible(features, **label_kwargs)
    else:
        labels = label_possession(features, **label_kwargs)
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
