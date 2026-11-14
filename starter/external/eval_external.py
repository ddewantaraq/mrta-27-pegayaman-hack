"""Score sm_v1_3 and missing-ball candidates on Metrica sample games.

Uses event-derived labels (metrica_gt) and synthetic / natural ball dropout.
Selection: beat sm_v1_3 on mean spellF1; hidden-frame team accuracy >= sm_v1_3;
leave-one-game-out (same pick beats sm_v1_3 on each game).

    .venv/bin/python starter/external/eval_external.py
    .venv/bin/python starter/external/eval_external.py --sweep
"""

from __future__ import print_function

import argparse
import csv
import os
import sys
from itertools import product

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STARTER = os.path.dirname(HERE)
ROOT = os.path.dirname(STARTER)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, STARTER)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import possession_metrics as pm  # noqa: E402
from fetch_metrica import DEFAULT_OUT, GAMES, fetch_all  # noqa: E402
from metrica_features import extract_metrica_features  # noqa: E402
from metrica_gt import build_metrica_gt  # noqa: E402
from dropout import apply_dropout, dropout_configs  # noqa: E402
from state_machine import (  # noqa: E402
    DEFAULT_CONTEST_DIST_M,
    DEFAULT_CONTROL_DIST_M,
    DEFAULT_CONTROL_FRAMES,
    DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
    DEFAULT_MIN_SPELL_FRAMES,
    DEFAULT_MISSING_NEAR_EDGE_FRAMES,
    DEFAULT_OUT_MARGIN,
    DEFAULT_STILL_NEAR_EDGE_FRAMES,
    DEFAULT_STILL_SPEED_M,
    DEFAULT_USE_MAJORITY_NEAR,
    DEFAULT_USE_CONTROL_CONFIRM,
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_CONFIRM_PASSES,
    DEFAULT_USE_DEAD_BRIDGE,
    DEFAULT_MAX_DEAD_BRIDGE_FRAMES,
    DEFAULT_VC_SPLICE,
    label_possession,
    splice_vc_on_visible,
)
from baseline import smooth  # noqa: E402
from features import HOME, AWAY, NOBODY  # noqa: E402
from dead_ball_model import (  # noqa: E402
    DEFAULT_MODEL_PATH, DEFAULT_USE_LEARNED_DEAD, label_with_dead_model,
)
from boosted_dead_model import (  # noqa: E402
    DEFAULT_BOOSTED_MODEL_PATH, DEFAULT_USE_BOOSTED_DEAD, label_with_boosted_model,
)
from set_ownership_model import (DEFAULT_SET_OWNERSHIP_PATH, DEFAULT_USE_SET_OWNERSHIP,
                                 DEFAULT_SET_OWNERSHIP_POLICY,
                                 apply_set_ownership)
from residual_ownership_model import (DEFAULT_RESIDUAL_MODEL_PATH, DEFAULT_USE_RESIDUAL_OWNERSHIP,
                                      apply_residual_ownership)
from boundary_ensemble_model import (DEFAULT_BOUNDARY_MODEL_PATH,DEFAULT_USE_BOUNDARY_ENSEMBLE,
                                     apply_boundary_ensemble)

GAME_NAMES = sorted(GAMES.keys())


def _codes_from_labels(labels):
    return pm.label_codes(np.asarray(labels, dtype=object))


def hidden_team_accuracy(gt_labels, pred_labels, hide_mask):
    """Fraction of hidden frames with GT H/A that we also label as that team."""
    hide = np.asarray(hide_mask, dtype=bool)
    gt = np.asarray(gt_labels)
    pred = np.asarray(pred_labels)
    alive_hide = hide & ((gt == "H") | (gt == "A"))
    n = int(alive_hide.sum())
    if n == 0:
        return float("nan"), 0
    correct = int(np.sum(alive_hide & (pred == gt)))
    return float(correct) / float(n), n


def smooth50_labels(features):
    """Nearest-team + 50-frame majority, matching starter/baseline smooth50."""
    codes = np.asarray(features["nearest_team"], dtype=np.int8).copy()
    # Where ball missing, nearest is NOBODY already
    return smooth(codes, 50)


# Published radii only. Not searched on BAR-ATH.
# Paper provider A: pz 0.5 m, dz 1.0 m. Providers B/C: both 1.0 m.
# databallpy's implementation of the same paper uses pz 1.5 m; dz must be >= pz.
VC_RADII = (
    ("vc_pz0.5_dz1.0", 0.5, 1.0),
    ("vc_pz1.0_dz1.0", 1.0, 1.0),
    ("vc_pz1.5_dz1.5", 1.5, 1.5),
)


def sm_v1_5_kwargs():
    """sm_v1_5: backdate + gap-bridge mid, on the sm_v1_3 base. No BAR-ATH kickoff."""
    kw = sm_v1_3_kwargs()
    kw["backdate_switch"] = True
    kw["gap_bridge"] = "mid"
    kw["use_vc_possession"] = False
    return kw


def vc_on_v15_kwargs(vc_gap="off"):
    """Same rules as sm_v1_5, with Vidal-Codina gain as the candidate.

    vc_gap off: path-change on every visible frame.
    vc_gap reappear: nearby team may open a switch in the control window after a gap.
    vc_gap edge: that window also requires a direction change across the gap.
    Radius stays the paper's 1.0 m. It is not searched here.
    """
    kw = sm_v1_5_kwargs()
    kw["use_vc_possession"] = True
    kw["vc_gap"] = vc_gap
    return kw


def sm_v1_3_kwargs():
    """sm_v1_3 reference parameters (Smart Tagging off; no BAR-ATH kickoff reset)."""
    return dict(
        control_frames=DEFAULT_CONTROL_FRAMES,
        control_dist_m=DEFAULT_CONTROL_DIST_M,
        out_margin=DEFAULT_OUT_MARGIN,
        min_spell_frames=DEFAULT_MIN_SPELL_FRAMES,
        missing_near_edge_frames=DEFAULT_MISSING_NEAR_EDGE_FRAMES,
        still_near_edge_frames=DEFAULT_STILL_NEAR_EDGE_FRAMES,
        still_speed_m=DEFAULT_STILL_SPEED_M,
        keep_last_live_on_dead=DEFAULT_KEEP_LAST_LIVE_ON_DEAD,
        use_majority_near=DEFAULT_USE_MAJORITY_NEAR,
        contest_dist_m=DEFAULT_CONTEST_DIST_M,
        set_piece_dead=None,
        backdate_switch=False,
        gap_bridge="off",
        long_gap_dead_frames=0,
        # BAR-ATH published kickoff frames must not apply on Metrica timelines
        half_kickoff_reset=False,
        use_control_confirm=False,
        use_dead_bridge=False,
    )


def run_labeller(features, graded, kwargs):
    kw = dict(kwargs)
    boundary_model=kw.pop('boundary_ensemble_model',None)
    if boundary_model is not None:
        from player_node_data import load_player_nodes
        nodes=load_player_nodes(str(features['game']))
        reference=run_labeller(features,graded,kw)
        return apply_boundary_ensemble(features,reference,graded,nodes,boundary_model,
                                        minimum=kw.get('min_spell_frames',DEFAULT_MIN_SPELL_FRAMES))
    residual_model = kw.pop('residual_ownership_model',None)
    if residual_model is not None:
        from player_node_data import load_player_nodes
        nodes = load_player_nodes(str(features['game']))
        reference = run_labeller(features,graded,kw)
        return apply_residual_ownership(features,reference,graded,nodes,residual_model,
                                        minimum=kw.get('min_spell_frames',DEFAULT_MIN_SPELL_FRAMES))
    team_model = kw.pop('set_ownership_model', None)
    team_policy = kw.pop('set_ownership_policy', 'full')
    if team_model is not None:
        from player_node_data import load_player_nodes
        nodes = load_player_nodes(str(features['game']))
        baseline = run_labeller(features, graded, kw)
        return apply_set_ownership(features, baseline, graded, nodes, team_model,
                                   minimum=kw.get('min_spell_frames', DEFAULT_MIN_SPELL_FRAMES),
                                   policy=team_policy)
    boosted_model = kw.pop('boosted_dead_model', None)
    if boosted_model is not None:
        return label_with_boosted_model(features, graded=graded, model_path=boosted_model, **kw)
    learned_model = kw.pop('learned_dead_model', None)
    if learned_model is not None:
        return label_with_dead_model(features, graded=graded, model_path=learned_model, **kw)
    splice = kw.pop("vc_splice", False)
    if splice:
        return splice_vc_on_visible(features, graded=graded, merge_after=True, **kw)
    return label_possession(
        features,
        game="BAR-ATH",  # unused when graded is passed
        graded=graded,
        **kw
    )


def score_one(gt_labels, pred_labels, hide_mask):
    gt_c = _codes_from_labels(gt_labels)
    pred_c = _codes_from_labels(pred_labels)
    result = pm.evaluate(gt_c, pred_c, bootstrap_n=0)
    hid_acc, hid_n = hidden_team_accuracy(gt_labels, pred_labels, hide_mask)
    return {
        "spell_f1_mean": result["spell_f1_mean"],
        "macro_f1": result["macro_f1"],
        "hidden_team_acc": hid_acc,
        "hidden_n": hid_n,
        "n_pred_spells": result["n_pred_spells"],
        "mean_spell_duration_s": result["mean_spell_duration_s"],
    }


def candidate_grid(include_long_gap=False):
    """Yield (name, kwargs) for A x B (x C). Always includes sm_v1_3 and smooth50."""
    yield "sm_v1_3", sm_v1_3_kwargs()
    yield "smooth50", None  # special

    bridges = ("off", "start", "mid", "end")
    backdates = (False, True)
    long_gaps = (0,)
    if include_long_gap:
        # 10 s and 20 s at 25 fps — literature-scale "can't interact" windows
        long_gaps = (0, 250, 500)

    for bridge, back, lg in product(bridges, backdates, long_gaps):
        if bridge == "off" and not back and lg == 0:
            continue  # already covered by sm_v1_3
        name = "bridge_{}_back_{}_lg_{}".format(bridge, int(back), lg)
        kw = sm_v1_3_kwargs()
        kw["gap_bridge"] = bridge
        kw["backdate_switch"] = back
        kw["long_gap_dead_frames"] = lg
        yield name, kw


def evaluate_game(game_name, root, include_long_gap=False, configs=None):
    """Return list of row dicts for one game across dropout configs and candidates."""
    features = extract_metrica_features(game_name, root=root)
    n = features["n_frames"]
    gt = build_metrica_gt(game_name, n, root=root)
    graded = np.ones(n, dtype=bool)
    # Leave pre-kickoff / halftime as graded=True; evaluate() ignores GT X.

    rows = []
    drop_list = list(dropout_configs(gt, features)) if configs is None else configs
    for drop_name, hide_mask in drop_list:
        feats_d = apply_dropout(features, hide_mask)
        # Natural missing: hide_mask equals already-missing; apply_dropout is no-op
        # on those frames. For scoring hidden-team-acc, use frames that are missing
        # in feats_d AND were H/A in GT — for natural, that's natural missing ∩ H/A.
        # For synthetic, hide_mask itself.
        score_hide = hide_mask | (~np.asarray(features["ball_present"], dtype=bool))

        for cand_name, kw in candidate_grid(include_long_gap=include_long_gap):
            if cand_name == "smooth50":
                # Smooth50 on the dropped features: nearest is NOBODY on hidden,
                # so those frames become D after smooth — pessimistic vs hold.
                # Use original nearest for a fair "baseline with missing=D" comparison:
                # rebuild codes from dropped features.
                pred = smooth50_labels(feats_d)
            else:
                pred = run_labeller(feats_d, graded, kw)
            metrics = score_one(gt, pred, score_hide)
            rows.append({
                "game": game_name,
                "dropout": drop_name,
                "candidate": cand_name,
                **metrics,
            })
    return rows


def aggregate_by_candidate(rows):
    """Mean spellF1 / hidden_team_acc across (game, dropout) for each candidate."""
    by = {}
    for row in rows:
        c = row["candidate"]
        by.setdefault(c, []).append(row)
    out = {}
    for c, items in by.items():
        spell = [r["spell_f1_mean"] for r in items]
        hid = [r["hidden_team_acc"] for r in items
               if r["hidden_team_acc"] == r["hidden_team_acc"]]  # not NaN
        out[c] = {
            "mean_spell_f1": float(np.mean(spell)),
            "mean_hidden_team_acc": float(np.mean(hid)) if hid else float("nan"),
            "n": len(items),
        }
    return out


def per_game_means(rows):
    """candidate -> game -> mean spellF1 / hidden over dropouts."""
    bucket = {}
    for row in rows:
        bucket.setdefault(row["candidate"], {}).setdefault(row["game"], []).append(row)
    out = {}
    for c, games in bucket.items():
        out[c] = {}
        for g, items in games.items():
            spell = [r["spell_f1_mean"] for r in items]
            hid = [r["hidden_team_acc"] for r in items
                   if r["hidden_team_acc"] == r["hidden_team_acc"]]
            out[c][g] = {
                "mean_spell_f1": float(np.mean(spell)),
                "mean_hidden_team_acc": float(np.mean(hid)) if hid else float("nan"),
            }
    return out


def pick_winner(rows):
    """Return (winner_name, reason) or (sm_v1_3, reason) if nothing beats it."""
    agg = aggregate_by_candidate(rows)
    per_g = per_game_means(rows)
    base = agg["sm_v1_3"]
    base_spell = base["mean_spell_f1"]
    base_hid = base["mean_hidden_team_acc"]
    games = sorted({r["game"] for r in rows})

    best = None
    best_spell = base_spell
    for name, stats in agg.items():
        if name in ("sm_v1_3", "smooth50"):
            continue
        if stats["mean_spell_f1"] <= base_spell:
            continue
        # require hidden-frame team accuracy >= sm_v1_3 when both defined
        cand_hid = stats["mean_hidden_team_acc"]
        if cand_hid == cand_hid and base_hid == base_hid and cand_hid < base_hid:
            continue
        # leave-one-game-out: beat sm_v1_3 on every game
        ok = True
        for g in games:
            if per_g[name][g]["mean_spell_f1"] <= per_g["sm_v1_3"][g]["mean_spell_f1"]:
                ok = False
                break
        if not ok:
            continue
        if stats["mean_spell_f1"] > best_spell:
            best_spell = stats["mean_spell_f1"]
            best = name

    if best is None:
        return "sm_v1_3", "no candidate beat sm_v1_3 on spellF1+hidden+LOGO"
    return best, "beat sm_v1_3 mean_spell {:.4f} -> {:.4f}".format(
        base_spell, best_spell)


def write_csv(path, rows):
    if not rows:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    keys = list(rows[0].keys())
    with open(path, "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _score_drop(gt, features, graded, hide_mask, candidates):
    """Score one dropout. candidates is a list of (name, kwargs or None for smooth50)."""
    feats_d = apply_dropout(features, hide_mask)
    natural_missing = ~np.asarray(features["ball_present"], dtype=bool)
    score_hide = hide_mask | natural_missing
    rows = []
    for cand_name, kw in candidates:
        if kw is None:
            pred = smooth50_labels(feats_d)
        else:
            pred = run_labeller(feats_d, graded, kw)
        metrics = score_one(gt, pred, score_hide)
        rows.append({"candidate": cand_name, **metrics})
    return rows


def evaluate_vc_splice(root):
    """VC on visible frames, sm_v1_5 labels on missing frames, then one merge."""
    rows = []
    for game_name in GAME_NAMES:
        print("=== {} splice pz=1.0 ===".format(game_name), flush=True)
        feats = extract_metrica_features(
            game_name, root=root, pz_radius_m=1.0, dz_radius_m=1.0)
        n = feats["n_frames"]
        gt = build_metrica_gt(game_name, n, root=root)
        graded = np.ones(n, dtype=bool)
        drops = list(dropout_configs(gt, feats))
        splice_kw = sm_v1_5_kwargs()
        splice_kw["vc_splice"] = True
        cands = (
            ("sm_v1_5", sm_v1_5_kwargs()),
            ("vc_splice", splice_kw),
            ("smooth50", None),
        )
        for drop_name, hide_mask in drops:
            print("  {}".format(drop_name), flush=True)
            for row in _score_drop(gt, feats, graded, hide_mask, cands):
                rows.append({
                    "game": game_name,
                    "dropout": drop_name,
                    "pz_radius_m": 1.0,
                    "dz_radius_m": 1.0,
                    **row,
                })
    return rows


def kwargs_for_version(major, minor):
    """Metrica kwargs for an approved model version. Extend when promoting."""
    if (major,minor)==(2,3):
        kw=kwargs_for_version(1,10)
        kw['boundary_ensemble_model']=DEFAULT_BOUNDARY_MODEL_PATH
        return kw
    if (major,minor) == (2,2):
        kw = kwargs_for_version(1,10)
        kw['residual_ownership_model'] = DEFAULT_RESIDUAL_MODEL_PATH
        return kw
    if (major, minor) == (2, 0):
        kw = kwargs_for_version(1, 10)
        kw['set_ownership_model'] = DEFAULT_SET_OWNERSHIP_PATH
        return kw
    if (major, minor) == (2, 1):
        kw = kwargs_for_version(2, 0)
        kw['set_ownership_policy'] = 'run90_mean90'
        return kw
    if (major, minor) == (1, 10):
        kw = kwargs_for_version(1, 8)
        kw['confirm_frames'] = DEFAULT_CONFIRM_FRAMES
        kw['confirm_passes'] = DEFAULT_CONFIRM_PASSES
        kw['boosted_dead_model'] = DEFAULT_BOOSTED_MODEL_PATH
        return kw
    if (major, minor) == (1, 9):
        kw = kwargs_for_version(1, 8)
        kw['confirm_frames'] = DEFAULT_CONFIRM_FRAMES
        kw['confirm_passes'] = DEFAULT_CONFIRM_PASSES
        kw['learned_dead_model'] = DEFAULT_MODEL_PATH
        return kw
    if (major, minor) == (1, 8):
        kw = sm_v1_5_kwargs()
        kw["vc_splice"] = True
        kw["use_control_confirm"] = False
        kw["use_dead_bridge"] = True
        kw["max_dead_bridge_frames"] = DEFAULT_MAX_DEAD_BRIDGE_FRAMES
        return kw
    if (major, minor) == (1, 7):
        kw = sm_v1_5_kwargs()
        kw["vc_splice"] = True
        kw["use_control_confirm"] = False
        kw["use_dead_bridge"] = False
        return kw
    if (major, minor) in ((1, 5), (1, 6)):
        return sm_v1_5_kwargs()
    raise KeyError(
        "no Metrica kwargs registry for version ({}, {})".format(major, minor))


def current_cand_kwargs():
    """Live predict defaults on Metrica (BAR-ATH kickoff stays off)."""
    kw = sm_v1_5_kwargs()
    kw["vc_splice"] = DEFAULT_VC_SPLICE
    kw["use_control_confirm"] = DEFAULT_USE_CONTROL_CONFIRM
    kw["confirm_frames"] = DEFAULT_CONFIRM_FRAMES
    kw["confirm_passes"] = DEFAULT_CONFIRM_PASSES
    kw["use_dead_bridge"] = DEFAULT_USE_DEAD_BRIDGE
    kw["max_dead_bridge_frames"] = DEFAULT_MAX_DEAD_BRIDGE_FRAMES
    if DEFAULT_USE_BOOSTED_DEAD:
        kw['boosted_dead_model'] = DEFAULT_BOOSTED_MODEL_PATH
    elif DEFAULT_USE_LEARNED_DEAD:
        kw['learned_dead_model'] = DEFAULT_MODEL_PATH
    if DEFAULT_USE_SET_OWNERSHIP:
        kw['set_ownership_model'] = DEFAULT_SET_OWNERSHIP_PATH
        kw['set_ownership_policy'] = DEFAULT_SET_OWNERSHIP_POLICY
    if DEFAULT_USE_BOUNDARY_ENSEMBLE:
        kw['boundary_ensemble_model']=DEFAULT_BOUNDARY_MODEL_PATH
    elif DEFAULT_USE_RESIDUAL_OWNERSHIP:
        kw['residual_ownership_model'] = DEFAULT_RESIDUAL_MODEL_PATH
    return kw


def evaluate_cand_vs_latest(root, cand_kwargs, latest_kwargs):
    """Assess Track B: cand vs LATEST vs smooth50 (same dropouts as splice eval).

    Row candidate ids are always 'cand', 'latest', 'smooth50' (no name collision).
    """
    rows = []
    cands = (
        ("cand", cand_kwargs),
        ("latest", latest_kwargs),
        ("smooth50", None),
    )
    for game_name in GAME_NAMES:
        print("=== {} assess cand vs LATEST pz=1.0 ===".format(game_name), flush=True)
        feats = extract_metrica_features(
            game_name, root=root, pz_radius_m=1.0, dz_radius_m=1.0,
            include_motion=('boosted_dead_model' in cand_kwargs or
                            'boosted_dead_model' in latest_kwargs))
        n = feats["n_frames"]
        gt = build_metrica_gt(game_name, n, root=root)
        graded = np.ones(n, dtype=bool)
        drops = list(dropout_configs(gt, feats))
        for drop_name, hide_mask in drops:
            print("  {}".format(drop_name), flush=True)
            for row in _score_drop(gt, feats, graded, hide_mask, cands):
                rows.append({
                    "game": game_name,
                    "dropout": drop_name,
                    "pz_radius_m": 1.0,
                    "dz_radius_m": 1.0,
                    **row,
                })
    return rows


def pick_cand_vs_latest(rows, cand_id="cand", latest_id="latest"):
    """Whether cand beats LATEST on mean spellF1 + hidden, both games."""
    agg = aggregate_by_candidate(rows)
    per_g = per_game_means(rows)
    if latest_id not in agg:
        return None, "missing LATEST {}".format(latest_id)
    if cand_id not in agg:
        return None, "missing cand {}".format(cand_id)
    base_spell = agg[latest_id]["mean_spell_f1"]
    base_hid = agg[latest_id]["mean_hidden_team_acc"]
    cand = agg[cand_id]
    if cand["mean_spell_f1"] <= base_spell:
        return None, "cand spellF1 {:.4f} did not beat LATEST {:.4f}".format(
            cand["mean_spell_f1"], base_spell)
    cand_hid = cand["mean_hidden_team_acc"]
    if cand_hid == cand_hid and base_hid == base_hid and cand_hid < base_hid:
        return None, "cand hidden {:.4f} below LATEST {:.4f}".format(
            cand_hid, base_hid)
    games = sorted({r["game"] for r in rows})
    for g in games:
        if per_g[cand_id][g]["mean_spell_f1"] <= per_g[latest_id][g]["mean_spell_f1"]:
            return None, "cand did not beat LATEST spellF1 on {}".format(g)
        ref_h = per_g[latest_id][g]["mean_hidden_team_acc"]
        cand_h = per_g[cand_id][g]["mean_hidden_team_acc"]
        if cand_h == cand_h and ref_h == ref_h and cand_h < ref_h:
            return None, "cand hidden below LATEST on {}".format(g)
    return cand_id, "beat LATEST mean_spell {:.4f} -> {:.4f}".format(
        base_spell, cand["mean_spell_f1"])


def evaluate_vc_gap(root):
    """Score the two gap-edge rules at the published 1 m zone.

    Candidates: plain VC, reappear (nearby team, no path test), edge (nearby
    team plus a direction change across the gap). Gate is sm_v1_5 on spellF1
    and hidden-frame team accuracy, both games.
    """
    rows = []
    specs = (
        ("vc_plain", "off"),
        ("vc_reappear", "reappear"),
        ("vc_edge", "edge"),
    )
    for game_name in GAME_NAMES:
        print("=== {} pz=1.0 dz=1.0 ===".format(game_name), flush=True)
        feats = extract_metrica_features(
            game_name, root=root, pz_radius_m=1.0, dz_radius_m=1.0)
        n = feats["n_frames"]
        gt = build_metrica_gt(game_name, n, root=root)
        graded = np.ones(n, dtype=bool)
        drops = list(dropout_configs(gt, feats))
        cands = [("sm_v1_5", sm_v1_5_kwargs())]
        cands.extend((name, vc_on_v15_kwargs(mode)) for name, mode in specs)
        for drop_name, hide_mask in drops:
            print("  {}".format(drop_name), flush=True)
            for row in _score_drop(gt, feats, graded, hide_mask, cands):
                rows.append({
                    "game": game_name,
                    "dropout": drop_name,
                    "pz_radius_m": 1.0,
                    "dz_radius_m": 1.0,
                    **row,
                })
    return rows


def evaluate_vc(root):
    """Score sm_v1_5 against Vidal-Codina radii on both Metrica games.

    Radii are the paper's published set plus databallpy's published possession
    zone. Selection is leave-one-game-out against sm_v1_5, not against BAR-ATH.
    """
    rows = []
    for game_name in GAME_NAMES:
        print("=== {} ===".format(game_name))
        base = extract_metrica_features(game_name, root=root)
        n = base["n_frames"]
        gt = build_metrica_gt(game_name, n, root=root)
        graded = np.ones(n, dtype=bool)
        drops = list(dropout_configs(gt, base))
        base_cands = (
            ("sm_v1_5", sm_v1_5_kwargs()),
            ("sm_v1_3", sm_v1_3_kwargs()),
            ("smooth50", None),
        )
        for drop_name, hide_mask in drops:
            for row in _score_drop(gt, base, graded, hide_mask, base_cands):
                rows.append({
                    "game": game_name,
                    "dropout": drop_name,
                    "pz_radius_m": "",
                    "dz_radius_m": "",
                    **row,
                })
        for cand_name, pz, dz in VC_RADII:
            print("  extract {} pz={} dz={}".format(game_name, pz, dz))
            feats = extract_metrica_features(
                game_name, root=root, pz_radius_m=pz, dz_radius_m=dz)
            present = np.asarray(feats["ball_present"], dtype=bool)
            n_ball = int(present.sum())
            n_duel = int(np.sum(present & np.asarray(feats["pz_duel"], dtype=bool)))
            n_pz = int(np.sum(np.asarray(feats["pz_player"]) >= 0))
            print("    ball={}  pz_hit={} ({:.1f}%)  duel={} ({:.1f}%)".format(
                n_ball, n_pz, 100.0 * n_pz / max(1, n_ball),
                n_duel, 100.0 * n_duel / max(1, n_ball)))
            for drop_name, hide_mask in drops:
                for row in _score_drop(gt, feats, graded, hide_mask,
                                        ((cand_name, vc_on_v15_kwargs()),)):
                    rows.append({
                        "game": game_name,
                        "dropout": drop_name,
                        "pz_radius_m": pz,
                        "dz_radius_m": dz,
                        **row,
                    })
    return rows


def pick_vs_reference(rows, reference="sm_v1_5"):
    """Best candidate that beats `reference` on mean spellF1, hidden acc, and each game."""
    agg = aggregate_by_candidate(rows)
    per_g = per_game_means(rows)
    if reference not in agg:
        return None, "missing reference {}".format(reference)
    base_spell = agg[reference]["mean_spell_f1"]
    base_hid = agg[reference]["mean_hidden_team_acc"]
    games = sorted({r["game"] for r in rows})
    best = None
    best_spell = base_spell
    for name, stats in agg.items():
        if name == reference or not name.startswith("vc_"):
            continue
        if stats["mean_spell_f1"] <= base_spell:
            continue
        cand_hid = stats["mean_hidden_team_acc"]
        if cand_hid == cand_hid and base_hid == base_hid and cand_hid < base_hid:
            continue
        ok = True
        for g in games:
            if per_g[name][g]["mean_spell_f1"] <= per_g[reference][g]["mean_spell_f1"]:
                ok = False
                break
            ref_h = per_g[reference][g]["mean_hidden_team_acc"]
            cand_h = per_g[name][g]["mean_hidden_team_acc"]
            if cand_h == cand_h and ref_h == ref_h and cand_h < ref_h:
                ok = False
                break
        if not ok:
            continue
        if stats["mean_spell_f1"] > best_spell:
            best_spell = stats["mean_spell_f1"]
            best = name
    if best is None:
        return None, "no vc candidate beat {} on spellF1 + hidden, both games".format(reference)
    return best, "beat {} mean_spell {:.4f} -> {:.4f}".format(
        reference, base_spell, best_spell)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_OUT)
    parser.add_argument("--fetch", action="store_true",
                        help="download sample games before evaluating")
    parser.add_argument("--vc", action="store_true",
                        help="score Vidal-Codina radii against sm_v1_5 (skip the old grid)")
    parser.add_argument("--vc-gap", action="store_true",
                        help="score reappear and edge gap rules at the 1.0 m zone")
    parser.add_argument("--vc-splice", action="store_true",
                        help="score VC-on-visible spliced with sm_v1_5 on missing frames")
    parser.add_argument("--sweep", action="store_true",
                        help="full A x B grid (and optional C)")
    parser.add_argument("--long-gap", action="store_true",
                        help="include long-gap D candidates (C)")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "evaluation", "reports", "external_sweep.csv"),
    )
    args = parser.parse_args()

    if args.fetch:
        fetch_all(root=args.root)

    if args.vc_splice or args.vc_gap or args.vc:
        if args.vc_splice:
            all_rows = evaluate_vc_splice(args.root)
        elif args.vc_gap:
            all_rows = evaluate_vc_gap(args.root)
        else:
            all_rows = evaluate_vc(args.root)
        write_csv(args.out, all_rows)
        print("wrote {}".format(args.out))
        agg = aggregate_by_candidate(all_rows)
        per_g = per_game_means(all_rows)
        print("\n=== aggregate (mean over games x dropouts) ===")
        for name in sorted(agg, key=lambda n: -agg[n]["mean_spell_f1"]):
            s = agg[name]
            print("  {:24s}  spellF1={:.4f}  hidden={:.4f}  n={}".format(
                name, s["mean_spell_f1"], s["mean_hidden_team_acc"], s["n"]))
        print("\n=== per game (mean over dropouts) ===")
        for name in sorted(per_g):
            for game in sorted(per_g[name]):
                s = per_g[name][game]
                print("  {:24s}  {:16s}  spellF1={:.4f}  hidden={:.4f}".format(
                    name, game, s["mean_spell_f1"], s["mean_hidden_team_acc"]))
        winner, reason = pick_vs_reference(all_rows, "sm_v1_5")
        print("\nVS sm_v1_5: {} ({})".format(winner or "none", reason))
        return

    all_rows = []
    for game in GAME_NAMES:
        print("=== {} ===".format(game))
        rows = evaluate_game(
            game, args.root,
            include_long_gap=args.long_gap and args.sweep,
        )
        all_rows.extend(rows)
        # Print sm_v1_3 / smooth50 aggregate for this game quickly
        for cand in ("sm_v1_3", "smooth50"):
            items = [r for r in rows if r["candidate"] == cand]
            spell = np.mean([r["spell_f1_mean"] for r in items])
            hid = np.nanmean([r["hidden_team_acc"] for r in items])
            print("  {}: spellF1={:.4f}  hidden_acc={:.4f}".format(cand, spell, hid))

    write_csv(args.out, all_rows)
    print("wrote {}".format(args.out))

    agg = aggregate_by_candidate(all_rows)
    print("\n=== aggregate (mean over games x dropouts) ===")
    for name in sorted(agg, key=lambda n: -agg[n]["mean_spell_f1"]):
        s = agg[name]
        print("  {:40s}  spellF1={:.4f}  hidden={:.4f}  n={}".format(
            name, s["mean_spell_f1"], s["mean_hidden_team_acc"], s["n"]))

    winner, reason = pick_winner(all_rows)
    print("\nWINNER: {} ({})".format(winner, reason))
    print("sm_v1_3 spellF1={:.4f} hidden={:.4f}".format(
        agg["sm_v1_3"]["mean_spell_f1"], agg["sm_v1_3"]["mean_hidden_team_acc"]))
    if winner in agg:
        print("winner  spellF1={:.4f} hidden={:.4f}".format(
            agg[winner]["mean_spell_f1"], agg[winner]["mean_hidden_team_acc"]))


if __name__ == "__main__":
    main()
