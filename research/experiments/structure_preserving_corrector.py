"""Structure-preserving retune of the sm_v2_4 error corrector.

Reuses leave-one-match-out fold models under evaluation/reports/crossfit_corrector/.
Sweeps margin / window / min_island (+ optional structure_guard) for a Pareto win
vs sm_v2_3: mean natural spellF1 up, Track-A-like structure not worse on folds,
then challenge Track A + assess B/C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'starter'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'starter' / 'external'))

from crossfit_common import (  # noqa: E402
    MATCHES, REPORTS, load_match, score_natural, summarize_rows, predict_version)
from pipeline_090 import v23_team_logits  # noqa: E402
from error_corrector import design_corrector, load_logits  # noqa: E402
from state_machine import _merge_short_spells  # noqa: E402
from vc_possession import live_structure  # noqa: E402
from error_corrector_model import (  # noqa: E402
    _window_agree, _suppress_short_islands, _apply_with_structure_guard)

FOLD_DIR = REPORTS / 'crossfit_corrector'
OUT_JSON = REPORTS / 'structure_preserving_corrector.json'


def apply_from_corr(baseline, graded, present, corr, margin, window, min_island,
                    structure_guard, minimum=18):
    labels = np.asarray(baseline).copy()
    graded = np.asarray(graded, bool)
    live = graded & present & np.isin(labels, ['H', 'A'])
    corr_home = corr >= 0
    base_home = labels == 'H'
    propose = live & (corr_home != base_home) & (np.abs(corr) >= float(margin))
    propose_lab = np.where(corr_home, 'H', 'A')
    want = _window_agree(propose, propose_lab, corr, labels, margin, int(window))
    want = _suppress_short_islands(want, labels, graded, int(min_island))
    nflips = int(want.sum())
    if structure_guard:
        out = _apply_with_structure_guard(
            labels, graded, want, corr_home, corr, minimum)
        return out, nflips
    labels[want] = np.where(corr_home[want], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum), nflips


def cache_fold(holdout):
    path = FOLD_DIR / ('corr_' + holdout + '.json')
    if not path.exists():
        raise FileNotFoundError('missing fold model ' + str(path))
    data = load_match(holdout)
    f, graded, nodes = data['features'], data['graded'], data['nodes']
    baseline = predict_version(f, graded, (2, 3))
    team_logits = v23_team_logits(f, graded, nodes, baseline)
    corr = load_logits(f, baseline, team_logits, path)
    return dict(
        game=holdout, features=f, gt=data['gt'], graded=graded, nodes=nodes,
        baseline=baseline, corr=corr, present=np.asarray(f['ball_present'], bool),
        v23_structure=live_structure(baseline, graded),
        v23_metrics=score_natural(data['gt'], baseline, f),
    )


def structure_pass(cand_struct, ref_struct):
    return (cand_struct['short12'] <= ref_struct['short12']
            and cand_struct['short25'] <= ref_struct['short25']
            and cand_struct['mean_s'] + 1e-9 >= ref_struct['mean_s']
            and cand_struct['n_spells'] <= ref_struct['n_spells'] + 2)


def evaluate_params(caches, margin, window, min_island, structure_guard):
    rows = []
    for c in caches:
        pred, nflips = apply_from_corr(
            c['baseline'], c['graded'], c['present'], c['corr'],
            margin, window, min_island, structure_guard)
        metrics = score_natural(c['gt'], pred, c['features'])
        struct = live_structure(pred, c['graded'])
        ok = structure_pass(struct, c['v23_structure'])
        delta = metrics['spell_f1_mean'] - c['v23_metrics']['spell_f1_mean']
        hid = metrics['hidden_team_acc']
        hid_v = c['v23_metrics']['hidden_team_acc']
        hid_delta = (hid - hid_v) if (hid == hid and hid_v == hid_v) else float('nan')
        rows.append(dict(
            game=c['game'], flips=nflips,
            spell=metrics['spell_f1_mean'], spell_delta=delta,
            hidden=hid, hidden_delta=hid_delta,
            n_spells=struct['n_spells'], mean_s=struct['mean_s'],
            s12=struct['short12'], s25=struct['short25'],
            structure_ok=ok,
            v23_spell=c['v23_metrics']['spell_f1_mean'],
            v23_s25=c['v23_structure']['short25'],
            v23_mean_s=c['v23_structure']['mean_s'],
        ))
    mean_spell = float(np.mean([r['spell'] for r in rows]))
    mean_v23 = float(np.mean([r['v23_spell'] for r in rows]))
    worst = float(np.min([r['spell_delta'] for r in rows]))
    n_struct_ok = int(sum(1 for r in rows if r['structure_ok']))
    hid_deltas = [r['hidden_delta'] for r in rows if r['hidden_delta'] == r['hidden_delta']]
    mean_hid_delta = float(np.mean(hid_deltas)) if hid_deltas else float('nan')
    return dict(
        margin=margin, window=window, min_island=min_island,
        structure_guard=bool(structure_guard),
        mean_spell=mean_spell, mean_spell_v23=mean_v23,
        mean_spell_delta=mean_spell - mean_v23,
        worst_spell_delta=worst,
        n_structure_ok=n_struct_ok, n_matches=len(rows),
        mean_hidden_delta=mean_hid_delta,
        rows=rows,
    )


def nested_sweep(caches, grid=None):
    grid = grid or dict(
        margin=[1.5, 2.0, 2.5, 3.0],
        window=[0, 3, 5],
        min_island=[1, 8, 12, 18],
        structure_guard=[False, True],
    )
    results = []
    for margin in grid['margin']:
        for window in grid['window']:
            for min_island in grid['min_island']:
                for guard in grid['structure_guard']:
                    res = evaluate_params(
                        caches, margin, window, min_island, guard)
                    results.append(res)
                    print(
                        'm={:.1f} w={} isl={} guard={} spell={:.4f} Δ={:+.4f} '
                        'worst={:+.4f} struct_ok={}/{} hidΔ={:+.4f}'.format(
                            margin, window, min_island, guard,
                            res['mean_spell'], res['mean_spell_delta'],
                            res['worst_spell_delta'], res['n_structure_ok'],
                            res['n_matches'],
                            res['mean_hidden_delta']
                            if res['mean_hidden_delta'] == res['mean_hidden_delta']
                            else float('nan')),
                        flush=True)
    return results


def pick_best(results, mean_eps=0.0, worst_floor=-0.02):
    """Pareto: structure OK on all folds, spell ≥ v2.3, worst Δ ≥ floor, hid ≥."""
    feasible = []
    for r in results:
        if r['n_structure_ok'] < r['n_matches']:
            continue
        if r['mean_spell_delta'] < mean_eps:
            continue
        if r['worst_spell_delta'] < worst_floor:
            continue
        hid = r['mean_hidden_delta']
        if hid == hid and hid < -1e-6:
            continue
        feasible.append(r)
    if not feasible:
        # Fallback: maximize spell among structure-all-ok; else best spell.
        struct_all = [r for r in results if r['n_structure_ok'] == r['n_matches']]
        pool = struct_all or results
        return max(pool, key=lambda r: (
            r['n_structure_ok'], r['mean_spell_delta'], r['worst_spell_delta'])), False
    return max(feasible, key=lambda r: (
        r['mean_spell_delta'], r['worst_spell_delta'], -r['margin'])), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='smaller grid for a fast probe')
    args = parser.parse_args()
    print('caching {} folds…'.format(len(MATCHES)), flush=True)
    caches = [cache_fold(g) for g in MATCHES]
    grid = None
    if args.quick:
        grid = dict(
            margin=[1.5, 2.0, 2.5],
            window=[0, 5],
            min_island=[1, 12, 18],
            structure_guard=[True],
        )
    results = nested_sweep(caches, grid=grid)
    best, pareto = pick_best(results)
    slim = [{k: v for k, v in r.items() if k != 'rows'} for r in results]
    payload = dict(
        pareto_feasible=pareto,
        best={k: v for k, v in best.items() if k != 'rows'},
        best_rows=best['rows'],
        n_tried=len(results),
        all=slim,
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['best'], indent=2))
    print('pareto_feasible', pareto, 'wrote', OUT_JSON)


if __name__ == '__main__':
    main()
