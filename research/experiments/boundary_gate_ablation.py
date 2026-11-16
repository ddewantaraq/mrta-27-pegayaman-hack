"""Grid boundary blend / scale / gate on top of sm_v2_3 (NumPy only).

Logits are computed once per match/dropout; only decode policies vary.
Defaults reproduce v2.3. Selection uses evaluate_approved.gate_report.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from motion_dead_experiment import dataset
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import (kwargs_for_version, run_labeller, score_one, write_csv)
from evaluate_approved import ALL_MATCHES, SUITE_MATCHES, gate_report
from player_node_data import load_player_nodes
from numpy_set_temporal import load_parameters
from numpy_residual_set import predict_logits
from numpy_boundary_ownership import predict_dual, decode_boundary
from residual_ownership_model import DEFAULT_RESIDUAL_MODEL_PATH
from boundary_ensemble_model import DEFAULT_BOUNDARY_MODEL_PATH
from dead_ball_model import decode_dead_logits
from state_machine import _merge_short_spells


def grid_specs():
    for blend in (0.3, 0.5, 0.7):
        for scale in (0.5, 1.0):
            for gate in ('none', 'missing_only', 'confidence'):
                name = 'b{}_s{}_{}'.format(blend, scale, gate)
                yield name, dict(boundary_blend=blend, boundary_scale=scale,
                                 boundary_gate=gate, boundary_confidence=1.0)


def decode_policy(features, reference, graded, old, newer, boundary,
                  blend, boundary_scale, gate, confidence_threshold, minimum=18):
    labels = np.asarray(reference).copy()
    graded = np.asarray(graded, bool)
    signal = (1.0 - blend) * old + blend * newer
    live = graded & np.isin(labels, ['H', 'A'])
    present = np.asarray(features['ball_present'], bool)
    if gate == 'none':
        home = decode_boundary(
            signal, boundary, live, features.get('period'),
            boundary_scale=float(boundary_scale))
    elif gate == 'missing_only':
        home = decode_dead_logits(signal, live, features.get('period'))
        missing_live = live & ~present
        home_b = decode_boundary(
            signal, boundary, missing_live, features.get('period'),
            boundary_scale=float(boundary_scale))
        home = home.copy()
        home[missing_live] = home_b[missing_live]
    else:
        home = decode_dead_logits(signal, live, features.get('period'))
        uncertain = live & (np.abs(signal) < float(confidence_threshold))
        home_b = decode_boundary(
            signal, boundary, uncertain, features.get('period'),
            boundary_scale=float(boundary_scale))
        home = home.copy()
        home[uncertain] = home_b[uncertain]
    home = home.copy()
    home[present] = signal[present] >= 0
    labels[live] = np.where(home[live], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=ALL_MATCHES)
    parser.add_argument('--out', default='evaluation/reports/boundary_gate_ablation.csv')
    parser.add_argument('--summary', default='evaluation/reports/boundary_gate_ablation.json')
    args = parser.parse_args()
    base_kw = kwargs_for_version(2, 3)
    # Strip boundary so run_labeller returns the v1.10 / residual reference stack.
    ref_kw = dict(base_kw)
    ref_kw.pop('boundary_ensemble_model', None)
    # For v2.3 reference labels we need residual ownership as the base for ensemble.
    # Actually kwargs_for_version(2,3) is v1.10 + boundary. Without boundary it's v1.10.
    # Ensemble uses boosted baseline as reference labels. Match apply_boundary_ensemble:
    # reference = run_labeller without boundary = v1.10 boosted path.
    residual = load_parameters(DEFAULT_RESIDUAL_MODEL_PATH)
    boundary_params = load_parameters(DEFAULT_BOUNDARY_MODEL_PATH)
    specs = [('v23', dict(boundary_blend=0.5, boundary_scale=1.0,
                          boundary_gate='none', boundary_confidence=1.0))]
    specs.extend(list(grid_specs()))
    rows = []
    for game in args.matches:
        try:
            f, gt, graded = dataset(game)
        except FileNotFoundError as exc:
            print('SKIP', game, exc, flush=True)
            continue
        nodes = load_player_nodes(game)
        for sec in (0, 2, 5, 10, 30):
            hide = np.zeros(f['n_frames'], bool) if sec == 0 else block_dropout(
                f['n_frames'], sec, seed=41)
            dropped = apply_dropout(f, hide)
            score_hide = hide | ~f['ball_present']
            reference = run_labeller(dropped, graded, ref_kw)
            old = predict_logits(dropped, graded, nodes, reference, residual)
            newer, boundary = predict_dual(dropped, graded, nodes, reference,
                                           boundary_params)
            for name, overrides in specs:
                pred = decode_policy(
                    dropped, reference, graded, old, newer, boundary,
                    blend=overrides['boundary_blend'],
                    boundary_scale=overrides['boundary_scale'],
                    gate=overrides['boundary_gate'],
                    confidence_threshold=overrides['boundary_confidence'],
                    minimum=base_kw.get('min_spell_frames', 18))
                rows.append(dict(game=game, dropout=str(sec), candidate=name,
                                 **score_one(gt, pred, score_hide)))
            print(game, sec, flush=True)
    write_csv(args.out, rows)
    results = []
    for name, _ in specs:
        if name == 'v23':
            continue
        report = gate_report(
            [r for r in rows if r['candidate'] in (name, 'v23')],
            cand=name, ref='v23',
            suite_matches=[m for m in SUITE_MATCHES if any(r['game'] == m for r in rows)],
            all_matches=[m for m in ALL_MATCHES if any(r['game'] == m for r in rows)])
        results.append(dict(name=name, **{k: report[k] for k in (
            'passed', 'suite_spell_ok', 'suite_hid_ok', 'loo_ok', 'natural_ok',
            'wins', 'n_present', 'max_loss', 'match_deltas', 'natural_deltas',
            'suite')}))
        print('CAND', name, 'passed', report['passed'],
              'suite', report['suite'][name]['mean_spell_f1'],
              'vs', report['suite']['v23']['mean_spell_f1'], flush=True)
    passed = [r for r in results if r['passed']]
    if passed:
        best = max(passed, key=lambda r: r['suite'][r['name']]['mean_spell_f1'])
    else:
        natural = [r for r in results if r['natural_ok']]
        pool = natural or results
        best = max(pool, key=lambda r: r['suite'][r['name']]['mean_spell_f1'])
    summary = dict(best=best['name'], best_passed=bool(best['passed']), results=results)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(dict(best=best['name'], passed=best['passed'],
                          suite=best['suite']), indent=2), flush=True)


if __name__ == '__main__':
    main()
