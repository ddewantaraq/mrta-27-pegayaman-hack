"""Predict step: features -> state machine -> submission CSV.

The one command that also validates and scores is predict_v1.bat (Command Prompt)
or predict_v1.sh (Mac Terminal, or Git Bash on Windows). This file is the predict step.

By default writes the next file after evaluation/results/LATEST (approved best):
  bump minor until 10, then major+1; prefer bare sm_v*.csv; add _compliant
  only when that bare name already exists. Pass --out PATH to force a path.

Needs the match bundle in ./data or MTAH_DATA_DIR.
Defaults = compliant sm_v2_3 (boosted dead-ball + boundary-aware ownership ensemble).
Rollback: --no-boundary-model -> sm_v2_2; --no-team-model -> sm_v1_10;
--dead-model logistic -> sm_v1_9;
--no-learned-dead -> sm_v1_8.
Rollback bridge: --no-dead-bridge. T27 --control-confirm stays off.
Rollback splice: --no-vc-splice. Rollback kickoff: --no-half-kickoff-reset.
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

from dead_ball_model import (  # noqa: E402
    DEFAULT_MODEL_PATH, DEFAULT_USE_LEARNED_DEAD, label_with_dead_model,
)
from boosted_dead_model import (  # noqa: E402
    DEFAULT_BOOSTED_MODEL_PATH, DEFAULT_USE_BOOSTED_DEAD, label_with_boosted_model,
)
from set_ownership_model import (DEFAULT_SET_OWNERSHIP_PATH, DEFAULT_USE_SET_OWNERSHIP,
                                 DEFAULT_SET_OWNERSHIP_POLICY)
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH, DEFAULT_USE_RESIDUAL_OWNERSHIP
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH,DEFAULT_USE_BOUNDARY_ENSEMBLE

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
    DEFAULT_USE_CONTROL_CONFIRM,
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_CONFIRM_PASSES,
    DEFAULT_USE_DEAD_BRIDGE,
    DEFAULT_MAX_DEAD_BRIDGE_FRAMES,
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
    parser.add_argument('--no-learned-dead', dest='learned_dead', action='store_false',
                        default=DEFAULT_USE_LEARNED_DEAD,
                        help='roll back protected learned dead-ball model to sm_v1_8')
    parser.add_argument('--dead-model-file', default=DEFAULT_MODEL_PATH,
                        help='JSON parameters trained on external matches only')
    parser.add_argument('--dead-model', choices=('legacy', 'logistic', 'boosted'),
                        default=None, help='explicit live/dead pipeline; preserves legacy rollback')
    parser.add_argument('--boosted-model-file', default=DEFAULT_BOOSTED_MODEL_PATH,
                        help='portable boosted numeric-tree parameters')
    parser.add_argument('--team-model-file', default=None,
                        help='override NumPy player-set ownership model parameters')
    parser.add_argument('--no-team-model', action='store_true',
                        help='disable ownership head; boosted pipeline rolls back to sm_v1_10')
    parser.add_argument('--residual-team-model-file', default=None,
                        help='baseline-aware NumPy ownership correction parameters')
    parser.add_argument('--boundary-team-model-file',default=None,
                        help='override NumPy boundary-aware ensemble parameters')
    parser.add_argument('--no-boundary-model',action='store_true',
                        help='disable the boundary ensemble and use the v2.2 ownership head')
    parser.add_argument('--team-policy', choices=('full','run90_mean90'),
                        default=DEFAULT_SET_OWNERSHIP_POLICY,
                        help='ownership replacement policy; full is the ungated research candidate')
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
        help="one state machine only; add --no-learned-dead for legacy sm_v1_6",
    )
    parser.add_argument(
        "--control-confirm",
        dest="use_control_confirm",
        action="store_true",
        default=DEFAULT_USE_CONTROL_CONFIRM,
        help="revert sandwich spells lacking duration/pass evidence (off by default)",
    )
    parser.add_argument(
        "--no-control-confirm",
        dest="use_control_confirm",
        action="store_false",
        help="disable control-confirm (default)",
    )
    parser.add_argument(
        "--confirm-frames", type=int, default=DEFAULT_CONFIRM_FRAMES,
        help="min sandwich duration frames to keep a turnover (default 75 = 3 s)",
    )
    parser.add_argument(
        "--confirm-passes", type=int, default=DEFAULT_CONFIRM_PASSES,
        help="min same-team hops inside sandwich to keep a turnover (default 3)",
    )
    parser.add_argument(
        "--dead-bridge",
        dest="use_dead_bridge",
        action="store_true",
        default=DEFAULT_USE_DEAD_BRIDGE,
        help="fill brief D between same H/A (T32 sm_v1_8 candidate)",
    )
    parser.add_argument(
        "--no-dead-bridge",
        dest="use_dead_bridge",
        action="store_false",
        help="disable short D-sandwich bridge",
    )
    parser.add_argument(
        "--max-dead-bridge-frames", type=int,
        default=DEFAULT_MAX_DEAD_BRIDGE_FRAMES,
        help="max D length to bridge between same team (default 25 = 1 s)",
    )
    args = parser.parse_args()
    if not args.learned_dead and args.dead_model not in (None, 'legacy'):
        parser.error('--no-learned-dead conflicts with --dead-model logistic/boosted')
    method = args.dead_model
    if method is None:
        method = ('boosted' if DEFAULT_USE_BOOSTED_DEAD else 'logistic') if args.learned_dead else 'legacy'
    boundary_model_path=args.boundary_team_model_file
    if args.no_boundary_model and boundary_model_path:
        parser.error('--no-boundary-model conflicts with --boundary-team-model-file')
    if (boundary_model_path is None and not args.no_team_model and method=='boosted'
            and DEFAULT_USE_BOUNDARY_ENSEMBLE and not args.no_boundary_model
            and not args.team_model_file and not args.residual_team_model_file):
        boundary_model_path=DEFAULT_BOUNDARY_MODEL_PATH
    team_model_path = args.team_model_file
    if args.no_team_model and team_model_path:
        parser.error('--no-team-model conflicts with --team-model-file')
    if team_model_path is None and boundary_model_path is None and not args.no_team_model and method == 'boosted' and DEFAULT_USE_SET_OWNERSHIP:
        team_model_path = DEFAULT_SET_OWNERSHIP_PATH
    residual_model_path = args.residual_team_model_file
    if (residual_model_path is None and team_model_path is None and boundary_model_path is None and not args.no_team_model
            and method == 'boosted' and DEFAULT_USE_RESIDUAL_OWNERSHIP):
        residual_model_path = DEFAULT_RESIDUAL_MODEL_PATH
    if residual_model_path and (team_model_path or args.no_team_model):
        parser.error('Choose one ownership model; --no-team-model disables both')
    if boundary_model_path and (team_model_path or residual_model_path or args.no_team_model):
        parser.error('Choose one ownership model; --no-team-model disables ownership corrections')

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
    player_nodes = None
    if team_model_path or residual_model_path or boundary_model_path:
        from external.motion_features import challenge_motion
        print('Extract geometry-only player sets and motion ...')
        features['player_motion'], player_nodes, features['period'] = challenge_motion(
            game=args.game, include_nodes=True)
    elif method == 'boosted':
        from external.motion_features import challenge_motion
        print('Extract ball-independent player motion ...')
        features['player_motion'] = challenge_motion(game=args.game)

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
        use_control_confirm=args.use_control_confirm,
        confirm_frames=args.confirm_frames,
        confirm_passes=args.confirm_passes,
        use_dead_bridge=args.use_dead_bridge,
        max_dead_bridge_frames=args.max_dead_bridge_frames,
    )
    params["vc_splice"] = args.vc_splice
    params['dead_model_kind'] = method
    params['learned_dead_model'] = args.dead_model_file if method == 'logistic' else None
    params['boosted_dead_model'] = args.boosted_model_file if method == 'boosted' else None
    params['set_ownership_model'] = team_model_path
    params['set_ownership_policy'] = args.team_policy if team_model_path else None
    params['residual_ownership_model'] = residual_model_path
    params['boundary_ensemble_model']=boundary_model_path
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
        use_control_confirm=args.use_control_confirm,
        confirm_frames=args.confirm_frames,
        confirm_passes=args.confirm_passes,
        use_dead_bridge=args.use_dead_bridge,
        max_dead_bridge_frames=args.max_dead_bridge_frames,
    )
    if method == 'boosted':
        labels = label_with_boosted_model(
            features, model_path=args.boosted_model_file,
            vc_splice=args.vc_splice, **label_kwargs)
    elif method == 'logistic':
        labels = label_with_dead_model(
            features, model_path=args.dead_model_file,
            vc_splice=args.vc_splice, **label_kwargs)
    elif args.vc_splice:
        labels = splice_vc_on_visible(features, **label_kwargs)
    else:
        labels = label_possession(features, **label_kwargs)
    if team_model_path:
        from set_ownership_model import apply_set_ownership
        from state_machine import graded_mask
        labels = apply_set_ownership(features, labels,
                                     graded_mask(features['n_frames'], args.game),
                                     player_nodes, team_model_path,
                                     minimum=args.min_spell_frames, policy=args.team_policy)
    if residual_model_path:
        from residual_ownership_model import apply_residual_ownership
        from state_machine import graded_mask
        labels = apply_residual_ownership(features,labels,
                                          graded_mask(features['n_frames'],args.game),
                                          player_nodes,residual_model_path,
                                          minimum=args.min_spell_frames)
    if boundary_model_path:
        from boundary_ensemble_model import apply_boundary_ensemble
        from state_machine import graded_mask
        labels=apply_boundary_ensemble(features,labels,graded_mask(features['n_frames'],args.game),
                                       player_nodes,boundary_model_path,minimum=args.min_spell_frames)
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
