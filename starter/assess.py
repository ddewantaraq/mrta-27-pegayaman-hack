"""Dual-track assessment: local structure (A) + Metrica H/A (B).

Both tracks: cand vs LATEST (approved best) vs smooth50.

Track A — BAR-ATH CSV structure (n_spells, mean duration, short <12f / <25f,
  D, X). No BAR-ATH GT.

Track B — same challenge spellF1 code on Metrica event labels + dropouts,
  plus hidden-frame team accuracy. Re-runs kwargs for cand (code defaults)
  and LATEST (version registry); does not score the BAR-ATH CSV on Metrica.

    .venv/bin/python starter/assess.py --cand evaluation/results/sm_v1_7_compliant.csv
    bash assess.sh evaluation/results/sm_v1_7_compliant.csv
"""

from __future__ import print_function

import argparse
import os
import sys
import threading

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(HERE, "external"))

from model_latest import (  # noqa: E402
    format_version_label,
    latest_version,
    parse_sm_version,
    read_latest,
)
from state_machine import graded_mask  # noqa: E402
from vc_possession import live_structure  # noqa: E402

DEFAULT_SMOOTH = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")
DEFAULT_METRICA_OUT = os.path.join(
    ROOT, "evaluation", "reports", "assess_metrica.csv")


def _structure_row(path):
    labels = pd.read_csv(path)["label"].astype(str).to_numpy()
    s = live_structure(labels, graded_mask(len(labels)))
    return {
        "name": os.path.basename(path),
        "n_spells": s["n_spells"],
        "mean_s": s["mean_s"],
        "short12": s["short12"],
        "short25": s["short25"],
        "D": s["counts"]["D"],
        "X": s["counts"]["X"],
    }


def run_track_a(cand_path, ref_path, smooth_path):
    print("=== Track A: local structure (no BAR-ATH GT) ===")
    print("cand vs LATEST vs smooth50")
    rows = []
    for path in (cand_path, ref_path, smooth_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        rows.append(_structure_row(path))
    fmt = "{:<28} {:>6} {:>7} {:>4} {:>4} {:>6} {:>5}"
    print(fmt.format("file", "spells", "mean_s", "s12", "s25", "D", "X"))
    for r in rows:
        print(fmt.format(
            r["name"][:28],
            r["n_spells"],
            "{:.2f}".format(r["mean_s"]),
            r["short12"],
            r["short25"],
            r["D"],
            r["X"],
        ))
    print("Gate: prefer fewer s12/s25 and longer mean vs LATEST; clearly better than smooth50.")
    return rows


def _cand_display_name(cand_path):
    parsed = parse_sm_version(cand_path)
    if parsed is not None:
        return format_version_label(*parsed)
    return os.path.basename(cand_path)


def run_track_b(metrica_root, out_csv, cand_path, fetch=False):
    from fetch_metrica import fetch_all, DEFAULT_OUT  # noqa: E402
    from eval_external import (  # noqa: E402
        aggregate_by_candidate,
        current_cand_kwargs,
        evaluate_cand_vs_latest,
        kwargs_for_version,
        per_game_means,
        pick_cand_vs_latest,
        write_csv,
    )

    root = metrica_root or DEFAULT_OUT
    maj, minor = latest_version()
    latest_label = format_version_label(maj, minor)
    cand_label = _cand_display_name(cand_path)
    display = {"cand": cand_label, "latest": latest_label, "smooth50": "smooth50"}

    print("=== Track B: Metrica H/A (external spellF1 + hidden team acc) ===")
    print("spellF1 = same possession_metrics as the challenge; GT = Metrica events.")
    print("Model versions: {} vs {} vs smooth50.".format(cand_label, latest_label))
    if fetch:
        fetch_all(root=root)
    latest_kw = kwargs_for_version(maj, minor)
    cand_kw = current_cand_kwargs()
    rows = evaluate_cand_vs_latest(root, cand_kw, latest_kw)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(out_csv, rows)
    print("wrote {}".format(out_csv))
    agg = aggregate_by_candidate(rows)
    per_g = per_game_means(rows)
    print("\n--- aggregate (mean over games x dropouts) ---")
    order = ("cand", "latest", "smooth50")
    for name in sorted(agg, key=lambda n: -agg[n]["mean_spell_f1"]):
        s = agg[name]
        print("  {:16s}  spellF1={:.4f}  hidden={:.4f}  n={}".format(
            display.get(name, name), s["mean_spell_f1"], s["mean_hidden_team_acc"], s["n"]))
    print("\n--- per game ---")
    for name in order:
        if name not in per_g:
            continue
        for game in sorted(per_g[name]):
            s = per_g[name][game]
            print("  {:16s}  {:16s}  spellF1={:.4f}  hidden={:.4f}".format(
                display.get(name, name), game, s["mean_spell_f1"], s["mean_hidden_team_acc"]))
    winner, reason = pick_cand_vs_latest(rows)
    print("\nVS LATEST ({}): {} ({})".format(
        latest_label,
        display.get(winner, "none") if winner else "none",
        reason))
    print("Gate: beat LATEST on spellF1 AND hidden, both games.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cand", required=True,
        help="candidate BAR-ATH submission CSV (Track A)")
    parser.add_argument(
        "--ref", default=None,
        help="LATEST / structure reference CSV (default from evaluation/results/LATEST)")
    parser.add_argument(
        "--smooth", default=DEFAULT_SMOOTH,
        help="smooth50 baseline CSV")
    parser.add_argument(
        "--skip-metrica", action="store_true",
        help="run Track A only")
    parser.add_argument(
        "--fetch", action="store_true",
        help="download Metrica sample games before Track B")
    parser.add_argument(
        "--metrica-root", default=None,
        help="Metrica data root (default starter/external cache)")
    parser.add_argument(
        "--metrica-out", default=DEFAULT_METRICA_OUT,
        help="CSV path for Track B row dump")
    parser.add_argument(
        "--public-log", action="store_true",
        help="optional: also run public 5-min scorer (log only, not a keep gate)")
    args = parser.parse_args(argv)

    cand = os.path.abspath(args.cand)
    if not os.path.isfile(cand):
        print("FAIL  missing --cand {}".format(cand), file=sys.stderr)
        return 1
    ref_path = os.path.abspath(args.ref) if args.ref else read_latest()

    track_b_err = []

    def _b():
        try:
            run_track_b(
                args.metrica_root, args.metrica_out, cand, fetch=args.fetch)
        except Exception as exc:
            track_b_err.append(exc)

    b_thread = None
    if not args.skip_metrica:
        b_thread = threading.Thread(target=_b)
        b_thread.start()

    try:
        run_track_a(cand, ref_path, os.path.abspath(args.smooth))
    except Exception as exc:
        print("FAIL  Track A: {}".format(exc), file=sys.stderr)
        if b_thread is not None:
            b_thread.join()
        return 1

    if b_thread is not None:
        b_thread.join()
        if track_b_err:
            print("FAIL  Track B: {}".format(track_b_err[0]), file=sys.stderr)
            return 1

    if args.public_log:
        print("\n=== optional public log (not a keep gate) ===")
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="assess_public_")
        try:
            shutil.copy(cand, os.path.join(tmp, os.path.basename(cand)))
            rc = os.system(
                '"{}" "{}" --results "{}"'.format(
                    sys.executable,
                    os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                    tmp,
                ))
            if rc != 0:
                print("WARN  public log exited {}".format(rc), file=sys.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\nSee docs/ASSESS.md for the two-track keep ritual.")
    print("OK  assess {}".format(cand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
