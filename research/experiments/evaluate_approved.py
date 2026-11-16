"""Compare frozen / named-kwargs candidates on external matches (NumPy only).

No challenge reference is accepted. Full-timeline block dropout includes D,
unlike the older live-window-only Metrica dropout suite.

Reports natural spellF1 separately from the dropout suite and prints a
leave-one-match-out style pass/fail against the reference.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one,
                           write_csv, aggregate_by_candidate, per_game_means,
                           smooth50_labels)

SUITE_MATCHES = ['J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9', 'Sample_Game_2']
NATURAL_WATCH = ['J03WMX', 'Sample_Game_2', 'J03WR9']
ALL_MATCHES = SUITE_MATCHES + ['Sample_Game_3']


def version(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def resolve_kwargs(spec):
    """version string like '2.3', 'smooth50', or a Python dict literal of kwargs."""
    if spec == 'smooth50':
        return None
    if spec.startswith('{'):
        return ast.literal_eval(spec)
    return kwargs_for_version(*version(spec))


def score_match(game, candidates):
    f, gt, graded = dataset(game)
    rows = []
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(
            f['n_frames'], sec, seed=41)
        dropped = apply_dropout(f, hide)
        score_hide = hide | ~f['ball_present']
        for name, kw in candidates:
            if kw is None:
                pred = smooth50_labels(dropped)
            else:
                pred = run_labeller(dropped, graded, kw)
            rows.append(dict(game=game, dropout=str(sec), candidate=name,
                             **score_one(gt, pred, score_hide)))
        print(game, sec, flush=True)
    return rows


def equal_match_suite(rows, names, matches):
    """Equal-match mean of per-match suite means (all dropouts including natural)."""
    per = per_game_means(rows)
    out = {}
    for name in names:
        spells, hids = [], []
        for game in matches:
            if game not in per.get(name, {}):
                continue
            spells.append(per[name][game]['mean_spell_f1'])
            hids.append(per[name][game]['mean_hidden_team_acc'])
        out[name] = dict(
            mean_spell_f1=float(np.mean(spells)) if spells else float('nan'),
            mean_hidden_team_acc=float(np.mean(hids)) if hids else float('nan'),
            n_matches=len(spells))
    return out


def natural_only(rows):
    return [r for r in rows if r['dropout'] == '0']


def gate_report(rows, cand='candidate', ref='reference',
                suite_matches=None, all_matches=None, natural_watch=None):
    suite_matches = suite_matches or SUITE_MATCHES
    all_matches = all_matches or ALL_MATCHES
    natural_watch = natural_watch or NATURAL_WATCH
    per = per_game_means(rows)
    nat = per_game_means(natural_only(rows))
    suite = equal_match_suite(rows, [cand, ref], suite_matches)
    present = [g for g in all_matches if g in per.get(cand, {}) and g in per.get(ref, {})]
    wins = 0
    non_lose = 0
    max_loss = 0.0
    match_deltas = {}
    for game in present:
        delta = per[cand][game]['mean_spell_f1'] - per[ref][game]['mean_spell_f1']
        match_deltas[game] = delta
        if delta > 1e-12:
            wins += 1
            non_lose += 1
        elif delta >= -1e-12:
            non_lose += 1
        max_loss = max(max_loss, -delta)
    natural_ok = True
    natural_deltas = {}
    for game in natural_watch:
        if game not in nat.get(cand, {}) or game not in nat.get(ref, {}):
            continue
        delta = nat[cand][game]['mean_spell_f1'] - nat[ref][game]['mean_spell_f1']
        natural_deltas[game] = delta
        if delta < -1e-12:
            natural_ok = False
    suite_spell_ok = suite[cand]['mean_spell_f1'] >= suite[ref]['mean_spell_f1'] - 1e-12
    suite_hid_ok = (suite[cand]['mean_hidden_team_acc']
                    >= suite[ref]['mean_hidden_team_acc'] - 1e-12)
    # Beat or tie on >= n-1 matches; no match loses more than 0.01.
    loo_ok = non_lose >= max(1, len(present) - 1) and max_loss <= 0.01
    smooth = None
    if 'smooth50' in per:
        smooth = equal_match_suite(rows, [cand, 'smooth50'], suite_matches)
        smooth_ok = (smooth[cand]['mean_spell_f1'] > smooth['smooth50']['mean_spell_f1']
                     and smooth[cand]['mean_hidden_team_acc']
                     > smooth['smooth50']['mean_hidden_team_acc'])
    else:
        smooth_ok = None
    passed = suite_spell_ok and suite_hid_ok and loo_ok and natural_ok
    if smooth_ok is False:
        passed = False
    return dict(
        passed=passed,
        suite_spell_ok=suite_spell_ok,
        suite_hid_ok=suite_hid_ok,
        loo_ok=loo_ok,
        natural_ok=natural_ok,
        smooth50_ok=smooth_ok,
        wins=wins,
        non_lose=non_lose,
        n_present=len(present),
        max_loss=max_loss,
        match_deltas=match_deltas,
        natural_deltas=natural_deltas,
        suite=suite,
        smooth50=smooth,
        natural=nat,
        per_game=per,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='2.3',
                        help="version '2.3', 'smooth50', or kwargs dict literal")
    parser.add_argument('--reference', default='2.2')
    parser.add_argument('--candidate-name', default='candidate')
    parser.add_argument('--reference-name', default='reference')
    parser.add_argument('--include-smooth50', action='store_true')
    parser.add_argument('--matches', nargs='+', default=None)
    parser.add_argument('--out', default='evaluation/reports/approved_full_timeline.csv')
    parser.add_argument('--gate-json', default=None,
                        help='optional path for pass/fail JSON summary')
    args = parser.parse_args()
    matches = args.matches or ALL_MATCHES
    candidates = [
        (args.candidate_name, resolve_kwargs(args.candidate)),
        (args.reference_name, resolve_kwargs(args.reference)),
    ]
    if args.include_smooth50:
        candidates.append(('smooth50', None))
    rows = []
    for game in matches:
        try:
            rows.extend(score_match(game, candidates))
        except FileNotFoundError as exc:
            print('SKIP', game, exc, flush=True)
    write_csv(args.out, rows)
    report = gate_report(rows, args.candidate_name, args.reference_name,
                         suite_matches=[m for m in SUITE_MATCHES if any(
                             r['game'] == m for r in rows)],
                         all_matches=[m for m in ALL_MATCHES if any(
                             r['game'] == m for r in rows)])
    summary = dict(
        aggregate=aggregate_by_candidate(rows),
        per_game=report['per_game'],
        natural=report['natural'],
        gate=report,
    )
    print(json.dumps(summary, indent=2))
    print('GATE passed={} suite_spell={} suite_hid={} loo={} natural={} smooth50={}'.format(
        report['passed'], report['suite_spell_ok'], report['suite_hid_ok'],
        report['loo_ok'], report['natural_ok'], report['smooth50_ok']), flush=True)
    if args.gate_json:
        Path(args.gate_json).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
