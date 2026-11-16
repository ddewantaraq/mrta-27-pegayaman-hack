"""Three-track assessment: local structure (A) + Metrica (B) + Sportec (C).

All external tracks: cand vs reference vs smooth50 where applicable.

Track A — BAR-ATH CSV structure (n_spells, mean duration, short <12f / <25f,
  D, X). No BAR-ATH GT. Columns: cand vs --ref (default LATEST) vs smooth50.

Track B — same challenge spellF1 code on Metrica event labels + dropouts,
  plus hidden-frame team accuracy. Re-runs kwargs for cand and reference
  versions (from CSV names when parseable); does not score the BAR-ATH CSV
  on Metrica.

Track C — Sportec Open DFL excluded-fit matches (default J03WMX/WPY/WQQ/WR9)
  under the full-timeline natural + 2/5/10/30s block-dropout suite used for
  sm_v1_10 / sm_v2_x promotion. Same spellF1 + hidden metrics as Track B.

    .venv/bin/python starter/assess.py --cand evaluation/results/sm_v2_3_compliant.csv \\
        --ref evaluation/results/sm_v2_2_compliant.csv
    bash assess.sh evaluation/results/sm_v2_3_compliant.csv --ref evaluation/results/sm_v2_2_compliant.csv
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
DEFAULT_SPORTEC_OUT = os.path.join(
    ROOT, "evaluation", "reports", "assess_sportec.csv")
DEFAULT_SKILLCORNER_OUT = os.path.join(
    ROOT, "evaluation", "reports", "assess_skillcorner.csv")
DEFAULT_SPORTEC_MATCHES = ("J03WMX", "J03WPY", "J03WQQ", "J03WR9")


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
    print("cand vs ref vs smooth50")
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
    print("Gate: prefer fewer s12/s25 and longer mean vs ref; clearly better than smooth50.")
    return rows


def _cand_display_name(cand_path):
    parsed = parse_sm_version(cand_path)
    if parsed is not None:
        return format_version_label(*parsed)
    return os.path.basename(cand_path)


def _kwargs_for_path(path, fallback="current"):
    """Labeller kwargs from a submission filename, else current or LATEST."""
    from eval_external import current_cand_kwargs, kwargs_for_version  # noqa: E402

    parsed = parse_sm_version(path) if path else None
    if parsed is not None:
        try:
            return kwargs_for_version(*parsed), format_version_label(*parsed)
        except KeyError as exc:
            # Auto-bumped sm_v* after LATEST: predict just wrote current defaults.
            if fallback == "latest":
                raise
            print("WARN  {}: using current predict defaults for cand".format(exc),
                  flush=True)
            return current_cand_kwargs(), format_version_label(*parsed)
    if fallback == "latest":
        maj, minor = latest_version()
        return kwargs_for_version(maj, minor), format_version_label(maj, minor)
    return current_cand_kwargs(), "cand-defaults"


def _print_external_summary(rows, cand_label, ref_label, include_smooth=True):
    from eval_external import (  # noqa: E402
        aggregate_by_candidate,
        per_game_means,
        pick_cand_vs_latest,
    )

    display = {"cand": cand_label, "latest": ref_label, "smooth50": "smooth50"}
    agg = aggregate_by_candidate(rows)
    per_g = per_game_means(rows)
    print("\n--- aggregate (mean over games x dropouts) ---")
    for name in sorted(agg, key=lambda n: -agg[n]["mean_spell_f1"]):
        s = agg[name]
        print("  {:16s}  spellF1={:.4f}  hidden={:.4f}  n={}".format(
            display.get(name, name), s["mean_spell_f1"], s["mean_hidden_team_acc"], s["n"]))
    print("\n--- per game ---")
    order = ("cand", "latest", "smooth50") if include_smooth else ("cand", "latest")
    for name in order:
        if name not in per_g:
            continue
        for game in sorted(per_g[name]):
            s = per_g[name][game]
            print("  {:16s}  {:16s}  spellF1={:.4f}  hidden={:.4f}".format(
                display.get(name, name), game, s["mean_spell_f1"], s["mean_hidden_team_acc"]))
    winner, reason = pick_cand_vs_latest(rows)
    print("\nVS REF ({}): {} ({})".format(
        ref_label,
        display.get(winner, "none") if winner else "none",
        reason))
    return rows


def run_leakage_check(cand_path, ref_path):
    """Print in-sample flags per track; return violations for the candidate."""
    from leakage_guard import report  # noqa: E402

    print("=== Leakage: model training matches vs evaluation tracks ===")
    cand_kw, cand_label = _kwargs_for_path(cand_path, fallback="current")
    ref_kw, ref_label = _kwargs_for_path(ref_path, fallback="latest")
    cand_lines, bad = report(cand_kw, cand_label, parse_sm_version(cand_path))
    ref_lines, _ = report(ref_kw, ref_label, parse_sm_version(ref_path))
    for line in cand_lines + ref_lines:
        print(line)
    if bad:
        print("Gate: candidate adds in-sample evaluation matches: {}".format(bad))
    return bad


def run_track_b(metrica_root, out_csv, cand_path, ref_path, fetch=False):
    from fetch_metrica import fetch_all, DEFAULT_OUT  # noqa: E402
    from eval_external import evaluate_cand_vs_latest, write_csv  # noqa: E402

    root = metrica_root or DEFAULT_OUT
    cand_kw, cand_label = _kwargs_for_path(cand_path, fallback="current")
    ref_kw, ref_label = _kwargs_for_path(ref_path, fallback="latest")

    print("=== Track B: Metrica H/A (external spellF1 + hidden team acc) ===")
    print("spellF1 = same possession_metrics as the challenge; GT = Metrica events.")
    print("Model versions: {} vs {} vs smooth50.".format(cand_label, ref_label))
    if fetch:
        fetch_all(root=root)
    rows = evaluate_cand_vs_latest(root, cand_kw, ref_kw)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(out_csv, rows)
    print("wrote {}".format(out_csv))
    _print_external_summary(rows, cand_label, ref_label, include_smooth=True)
    print("Gate: beat ref on spellF1 AND hidden, both games.")
    return rows


def _block_suite(match_list, cand_kw, ref_kw):
    """Natural + 2/5/10/30s block dropouts, cand vs ref, one row per cell."""
    import numpy as np
    from eval_external import run_labeller, score_one  # noqa: E402
    from dropout import apply_dropout  # noqa: E402
    from external_dataset import dataset  # noqa: E402
    from sportec_experiment import block_dropout  # noqa: E402

    rows = []
    for game in match_list:
        feats, gt, graded = dataset(game)
        for sec in (0, 2, 5, 10, 30):
            hide = (
                np.zeros(feats["n_frames"], dtype=bool)
                if sec == 0
                else block_dropout(feats["n_frames"], sec)
            )
            dropped = apply_dropout(feats, hide)
            score_hide = hide | (~np.asarray(feats["ball_present"], dtype=bool))
            for name, kw in (("cand", cand_kw), ("latest", ref_kw)):
                pred = run_labeller(dropped, graded, kw)
                rows.append(dict(
                    game=game,
                    dropout=("natural" if sec == 0 else "{}s".format(sec)),
                    candidate=name,
                    **score_one(gt, pred, score_hide),
                ))
            print("  {} {}s".format(game, sec), flush=True)
    return rows


def run_track_d(out_csv, cand_path, ref_path):
    """SkillCorner hold-out (broadcast tracking); H/A frames only, D/X unlabelled."""
    from eval_external import write_csv  # noqa: E402
    from leakage_guard import track_d  # noqa: E402

    matches = list(track_d())
    if not matches:
        raise FileNotFoundError(
            "no Track D split; run research/experiments/skillcorner_label_check.py")
    cand_kw, cand_label = _kwargs_for_path(cand_path, fallback="current")
    ref_kw, ref_label = _kwargs_for_path(ref_path, fallback="latest")
    print("=== Track D: SkillCorner H/A hold-out (broadcast, H/A frames only) ===")
    print("Matches: {}".format(", ".join(matches)))
    print("Model versions: {} vs {}.".format(cand_label, ref_label))
    rows = _block_suite(matches, cand_kw, ref_kw)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(out_csv, rows)
    print("wrote {}".format(out_csv))
    _print_external_summary(rows, cand_label, ref_label, include_smooth=False)
    print("Gate: no worse than ref on spellF1 and hidden across the hold-out.")
    return rows


def run_track_c(out_csv, cand_path, ref_path, matches, fetch=False):
    """Full-timeline Sportec suite: cand vs ref (no smooth50 column)."""
    from eval_external import write_csv  # noqa: E402

    cand_kw, cand_label = _kwargs_for_path(cand_path, fallback="current")
    ref_kw, ref_label = _kwargs_for_path(ref_path, fallback="latest")
    match_list = list(matches)

    print("=== Track C: Sportec H/A (full-timeline suite) ===")
    print("Matches (excluded from v2.3 fit): {}".format(", ".join(match_list)))
    print("Dropouts: natural + 2/5/10/30s blocks (seed 41, p=0.41).")
    print("Model versions: {} vs {}.".format(cand_label, ref_label))

    if fetch:
        import subprocess
        cmd = [
            sys.executable,
            os.path.join(HERE, "external", "fetch_sportec.py"),
            "--matches",
        ] + match_list
        print("fetch:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=ROOT)

    sportec_root = os.path.join(ROOT, "data_external", "sportec")
    missing = [
        m for m in match_list
        if not any(
            name.endswith(".xml") and m in name
            for name in (os.listdir(sportec_root) if os.path.isdir(sportec_root) else [])
        )
    ]
    if missing:
        raise FileNotFoundError(
            "Sportec match files missing for {}: {}. "
            "Run: python starter/external/fetch_sportec.py --matches {} "
            "(or pass --fetch-sportec). Use --skip-sportec to omit Track C.".format(
                sportec_root, ", ".join(missing), " ".join(missing)))

    rows = _block_suite(match_list, cand_kw, ref_kw)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_csv(out_csv, rows)
    print("wrote {}".format(out_csv))
    _print_external_summary(rows, cand_label, ref_label, include_smooth=False)
    print("Gate: beat ref on spellF1 AND hidden across the Sportec suite matches.")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cand", required=True,
        help="candidate BAR-ATH submission CSV (Track A)")
    parser.add_argument(
        "--ref", default=None,
        help="structure + kwargs reference CSV (default from evaluation/results/LATEST)")
    parser.add_argument(
        "--smooth", default=DEFAULT_SMOOTH,
        help="smooth50 baseline CSV")
    parser.add_argument(
        "--skip-metrica", action="store_true",
        help="skip Track B")
    parser.add_argument(
        "--skip-sportec", action="store_true",
        help="skip Track C")
    parser.add_argument(
        "--fetch", action="store_true",
        help="download Metrica sample games before Track B")
    parser.add_argument(
        "--fetch-sportec", action="store_true",
        help="download Track C Sportec matches before scoring")
    parser.add_argument(
        "--metrica-root", default=None,
        help="Metrica data root (default starter/external cache)")
    parser.add_argument(
        "--metrica-out", default=DEFAULT_METRICA_OUT,
        help="CSV path for Track B row dump")
    parser.add_argument(
        "--sportec-out", default=DEFAULT_SPORTEC_OUT,
        help="CSV path for Track C row dump")
    parser.add_argument(
        "--sportec-matches", nargs="+", default=list(DEFAULT_SPORTEC_MATCHES),
        help="Sportec match IDs for Track C")
    parser.add_argument(
        "--skillcorner", action="store_true",
        help="also run Track D (SkillCorner hold-out; needs fetch_skillcorner.py)")
    parser.add_argument(
        "--skillcorner-out", default=DEFAULT_SKILLCORNER_OUT,
        help="CSV path for Track D row dump")
    parser.add_argument(
        "--allow-leakage", action="store_true",
        help="run even if the candidate trains on evaluation-track matches")
    parser.add_argument(
        "--public-log", action="store_true",
        help="optional: also run public 5-min scorer (log only, not a keep gate)")
    args = parser.parse_args(argv)

    cand = os.path.abspath(args.cand)
    if not os.path.isfile(cand):
        print("FAIL  missing --cand {}".format(cand), file=sys.stderr)
        return 1
    ref_path = os.path.abspath(args.ref) if args.ref else read_latest()

    leaks = run_leakage_check(cand, ref_path)
    if leaks and not args.allow_leakage:
        print("FAIL  leakage guard (pass --allow-leakage for diagnostics only)",
              file=sys.stderr)
        return 1

    track_errs = []

    def _b():
        try:
            run_track_b(
                args.metrica_root, args.metrica_out, cand, ref_path,
                fetch=args.fetch)
        except Exception as exc:
            track_errs.append(("B", exc))

    def _c():
        try:
            run_track_c(
                args.sportec_out, cand, ref_path, args.sportec_matches,
                fetch=args.fetch_sportec)
        except Exception as exc:
            track_errs.append(("C", exc))

    def _d():
        try:
            run_track_d(args.skillcorner_out, cand, ref_path)
        except Exception as exc:
            track_errs.append(("D", exc))

    threads = []
    if args.skillcorner:
        t = threading.Thread(target=_d)
        t.start()
        threads.append(t)
    if not args.skip_metrica:
        t = threading.Thread(target=_b)
        t.start()
        threads.append(t)
    if not args.skip_sportec:
        t = threading.Thread(target=_c)
        t.start()
        threads.append(t)

    try:
        run_track_a(cand, ref_path, os.path.abspath(args.smooth))
    except Exception as exc:
        print("FAIL  Track A: {}".format(exc), file=sys.stderr)
        for t in threads:
            t.join()
        return 1

    for t in threads:
        t.join()
    if track_errs:
        for name, exc in track_errs:
            print("FAIL  Track {}: {}".format(name, exc), file=sys.stderr)
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

    print("\nSee docs/ASSESS.md for the three-track keep ritual.")
    print("OK  assess {}".format(cand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
