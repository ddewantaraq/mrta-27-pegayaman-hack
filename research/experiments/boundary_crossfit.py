"""Deterministic leave-one-match-out crossfit for boundary ownership retrains.

Trains each fold on the other 9 matches with a stable seed, exports npz, and
scores natural tracking against v2.3 and v2.5 baselines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural, summarize_rows,
    mean_ignore_nan)
from boundary_ownership_experiment import train as train_boundary
from eval_external import write_csv, kwargs_for_version, run_labeller
from missing_ownership_crossfit import export_pt_to_npz


ROOT = REPORTS / 'boundary_cf'


def stable_seed(tag, holdout, base=900):
    """Process-stable seed (avoids randomised hash())."""
    digest = 0
    for ch in '{}:{}'.format(tag, holdout):
        digest = (digest * 131 + ord(ch)) % 100000
    return int(base + digest % 1000)


def score_baselines(holdout):
    data = load_match(holdout)
    out = {}
    for name, ver in (('v23', (2, 3)), ('v25', (2, 5))):
        pred = run_labeller(data['features'], data['graded'],
                            kwargs_for_version(*ver))
        out[name] = score_natural(data['gt'], pred, data['features'])
    return out


def soup(pts, out):
    """Average floating weights of runs that share the teacher initialisation."""
    blobs = [torch.load(p, map_location='cpu', weights_only=False) for p in pts]
    merged = dict(blobs[0])
    state = {}
    for key, value in blobs[0]['state_dict'].items():
        if value.is_floating_point():
            state[key] = torch.stack([b['state_dict'][key] for b in blobs]).mean(0)
        else:
            state[key] = value
    merged['state_dict'] = state
    merged['soup_seeds'] = [int(b['seed']) for b in blobs]
    torch.save(merged, out)
    return out


def run_fold(holdout, tag, train_kwargs, force=False, seeds=1, train_games=None):
    fold_dir = ROOT / tag
    fold_dir.mkdir(parents=True, exist_ok=True)
    pt = fold_dir / ('boundary_' + holdout + '.pt')
    npz = fold_dir / ('boundary_' + holdout + '.npz')
    train_games = train_games or fold_train_games(holdout)
    import boundary_ownership_experiment as boe
    import ownership_context as oc
    old_games = boe.GAMES
    boe.GAMES = train_games
    use_transition = bool(train_kwargs.get('transition_context', False))
    try:
        if force or not npz.exists():
            pts = []
            for k in range(seeds):
                seed = stable_seed(tag if k == 0 else '{}#{}'.format(tag, k), holdout)
                path = pt if seeds == 1 else pt.with_name('{}.s{}.pt'.format(pt.stem, k))
                kw = dict(train_kwargs)
                kw['seed'] = seed
                kw['model_path'] = path
                print('train', tag, holdout, 'seed', seed, 'games', len(train_games),
                      flush=True)
                # Drop stale checkpoint so resume cannot mix recipes.
                last = path.with_suffix('.last.pt')
                if last.exists() and force:
                    last.unlink()
                train_boundary(**kw)
                pts.append(path)
            if seeds > 1:
                soup(pts, pt)
            export_pt_to_npz(pt, npz)
    finally:
        boe.GAMES = old_games
    data = load_match(holdout)
    kw = kwargs_for_version(2, 3)
    kw['boundary_ensemble_model'] = str(npz.resolve())
    prev = oc.USE_TRANSITION_CONTEXT
    oc.USE_TRANSITION_CONTEXT = use_transition
    try:
        pred = run_labeller(data['features'], data['graded'], kw)
    finally:
        oc.USE_TRANSITION_CONTEXT = prev
    metrics = score_natural(data['gt'], pred, data['features'])
    bases = score_baselines(holdout)
    row = dict(
        game=holdout, candidate=tag, tag=tag,
        spell_f1_mean=metrics['spell_f1_mean'],
        hidden_team_acc=metrics['hidden_team_acc'],
        hidden_n=metrics.get('hidden_n', metrics.get('n_natural_missing_live')),
        v23_spell=bases['v23']['spell_f1_mean'],
        v25_spell=bases['v25']['spell_f1_mean'],
        v23_hidden=bases['v23']['hidden_team_acc'],
        v25_hidden=bases['v25']['hidden_team_acc'],
        delta_v23=metrics['spell_f1_mean'] - bases['v23']['spell_f1_mean'],
        delta_v25=metrics['spell_f1_mean'] - bases['v25']['spell_f1_mean'],
        npz=str(npz),
    )
    print('holdout', holdout, 'spell', round(row['spell_f1_mean'], 4),
          'Δv23', round(row['delta_v23'], 4),
          'Δv25', round(row['delta_v25'], 4), flush=True)
    return row


def summarize_vs(rows, ref_key='v25_spell', delta_key='delta_v25'):
    mean_spell = float(np.mean([r['spell_f1_mean'] for r in rows]))
    mean_ref = float(np.mean([r[ref_key] for r in rows]))
    worst = float(np.min([r[delta_key] for r in rows]))
    hid = mean_ignore_nan([r['hidden_team_acc'] for r in rows])
    hid_ref = mean_ignore_nan([r['v25_hidden' if 'v25' in ref_key else 'v23_hidden']
                               for r in rows])
    return dict(
        mean_spell=mean_spell,
        mean_ref=mean_ref,
        mean_delta=mean_spell - mean_ref,
        worst_delta=worst,
        mean_hidden=hid,
        mean_hidden_ref=hid_ref,
        n_matches=len(rows),
    )


def crossfit(tag, train_kwargs, matches=None, force=False, seeds=1):
    matches = matches or MATCHES
    rows = [run_fold(h, tag, train_kwargs, force=force, seeds=seeds) for h in matches]
    fold_dir = ROOT / tag
    write_csv(fold_dir / 'crossfit.csv', rows)
    summary = dict(
        tag=tag,
        train_kwargs={k: (str(v) if isinstance(v, Path) else v)
                      for k, v in train_kwargs.items()},
        vs_v23=summarize_vs(rows, 'v23_spell', 'delta_v23'),
        vs_v25=summarize_vs(rows, 'v25_spell', 'delta_v25'),
        aggregate=summarize_rows(rows),
        rows=rows,
    )
    Path(fold_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(tag=tag, vs_v23=summary['vs_v23'], vs_v25=summary['vs_v25']),
                     indent=2))
    return summary


DEFAULT_CONTROL = dict(
    missing_weight=1.0,
    long_dropout_boost=False,
    boundary_loss_weight=0.2,
    boundary_pos_weight=10.0,
    boundary_sigma=5.0,
    boundary_trunc=15,
    contested_boost=1.0,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--missing-weight', type=float, default=1.0)
    parser.add_argument('--long-dropout-boost', action='store_true')
    parser.add_argument('--boundary-loss-weight', type=float, default=0.2)
    parser.add_argument('--boundary-pos-weight', type=float, default=10.0)
    parser.add_argument('--boundary-sigma', type=float, default=5.0)
    parser.add_argument('--boundary-trunc', type=int, default=15)
    parser.add_argument('--contested-boost', type=float, default=1.0)
    parser.add_argument('--transition-context', action='store_true')
    parser.add_argument('--matches', nargs='+', default=None)
    parser.add_argument('--seeds', type=int, default=1,
                        help='train N seeds per fold and average their weights')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train_kwargs = dict(
        missing_weight=args.missing_weight,
        long_dropout_boost=args.long_dropout_boost,
        boundary_loss_weight=args.boundary_loss_weight,
        boundary_pos_weight=args.boundary_pos_weight,
        boundary_sigma=args.boundary_sigma,
        boundary_trunc=args.boundary_trunc,
        contested_boost=args.contested_boost,
        transition_context=args.transition_context,
    )
    crossfit(args.tag, train_kwargs, matches=args.matches, force=args.force,
             seeds=args.seeds)


if __name__ == '__main__':
    main()
