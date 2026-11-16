"""Cross-fit missing-ball boundary ownership (wider net optional).

Trains on 9 matches with missing-frame loss upweight; scores natural holdout
by swapping the boundary ensemble head into the v2.3 stack.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural, summarize_rows)
from boundary_ownership_experiment import train as train_boundary
from export_boundary_ownership import main as _export_main  # not used
from eval_external import write_csv, kwargs_for_version, run_labeller
import torch


def export_pt_to_npz(pt_path, npz_path):
    from boundary_ownership_model import load_boundary_model
    from numpy_set_temporal import load_parameters
    from numpy_boundary_ownership import forward_dual
    net, mean, scale = load_boundary_model(pt_path)
    p = {key: value.numpy() for key, value in net.state_dict().items()}
    p['temporal.project.weight'] = p['temporal.project.weight'][
        :, np.r_[np.arange(57), 157, np.arange(57, 157)]]
    p.update(mean=mean, scale=scale, layers=np.array(len(net.temporal.blocks)),
             baseline_prior=np.array(net.prior))
    np.savez_compressed(npz_path, **p)
    # Quick parity probe
    p = load_parameters(npz_path)
    rng = np.random.RandomState(0)
    x = rng.randn(17, 57).astype(np.float32)
    nodes = rng.rand(17, 2, 11, 8).astype(np.float32)
    nodes[..., 6] = nodes[..., 6] > .3
    ctx = rng.rand(17, 8).astype(np.float32)
    base = rng.choice([-1., 0., 1.], 17).astype(np.float32)
    with torch.inference_mode():
        expected = net(torch.from_numpy(x.T.copy())[None],
                       torch.from_numpy(nodes)[None],
                       torch.from_numpy(ctx)[None],
                       torch.from_numpy(base)[None])
    team, boundary = forward_dual(np.column_stack([x, base]), nodes, ctx, p)
    for k, actual in enumerate((team + net.prior * base, boundary)):
        np.testing.assert_allclose(actual, expected[k][0].numpy(), rtol=2e-5, atol=2e-5)


def crossfit_missing(matches=None, missing_weight=2.0, width_note='w32'):
    matches = matches or MATCHES
    fold_dir = REPORTS / 'crossfit_missing'
    fold_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        # Monkeypatch GAMES for this fold by calling train with custom path;
        # boundary_ownership_experiment.train uses module-level GAMES — override.
        import boundary_ownership_experiment as boe
        old_games = boe.GAMES
        boe.GAMES = train_games
        pt = fold_dir / ('boundary_' + holdout + '.pt')
        npz = fold_dir / ('boundary_' + holdout + '.npz')
        try:
            if not npz.exists():
                print('train missing fold', holdout, flush=True)
                train_boundary(missing_weight=missing_weight, seed=400 + abs(hash(holdout)) % 1000,
                               model_path=pt, long_dropout_boost=True)
                export_pt_to_npz(pt, npz)
        finally:
            boe.GAMES = old_games
        kw = kwargs_for_version(2, 3)
        kw['boundary_ensemble_model'] = str(npz.resolve())
        data = load_match(holdout)
        pred = run_labeller(data['features'], data['graded'], kw)
        metrics = score_natural(data['gt'], pred, data['features'])
        rows.append(dict(game=holdout, candidate='missing_mw2', **metrics))
        print('holdout', holdout, metrics['spell_f1_mean'], flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    parser.add_argument('--train-all', action='store_true')
    parser.add_argument('--missing-weight', type=float, default=2.0)
    args = parser.parse_args()
    if args.train_all:
        import boundary_ownership_experiment as boe
        boe.GAMES = MATCHES
        pt = REPORTS / 'boundary_ownership_all10.pt'
        npz = REPORTS / 'boundary_ownership_all10.npz'
        train_boundary(missing_weight=args.missing_weight, seed=811,
                       model_path=pt, long_dropout_boost=True)
        export_pt_to_npz(pt, npz)
        print('trained', npz, flush=True)
    if args.crossfit or not args.train_all:
        rows = crossfit_missing(missing_weight=args.missing_weight)
        write_csv(REPORTS / 'crossfit_missing.csv', rows)
        summary = summarize_rows(rows)
        Path(REPORTS / 'crossfit_missing.json').write_text(
            json.dumps(dict(aggregate=summary, rows=rows), indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
