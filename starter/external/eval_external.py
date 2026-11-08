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
    label_possession,
)
from baseline import smooth  # noqa: E402
from features import HOME, AWAY, NOBODY  # noqa: E402

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
    )


def run_labeller(features, graded, kwargs):
    return label_possession(
        features,
        game="BAR-ATH",  # unused when graded is passed
        graded=graded,
        **kwargs
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_OUT)
    parser.add_argument("--fetch", action="store_true",
                        help="download sample games before evaluating")
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
