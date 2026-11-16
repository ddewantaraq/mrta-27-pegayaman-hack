"""Crossfit HSMM decoder using v2.3 team logits + boosted dead logits."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, REPORTS, load_match, fold_train_games, score_natural,
    summarize_rows, predict_version)
from hsmm_decoder import fit_duration_from_games, decode_hsmm
from pipeline_090 import v23_team_logits
from dead_crossfit import predict_dead_logits, train_dead, export_dead
from eval_external import write_csv
from state_machine import _merge_short_spells
import joblib


def crossfit(matches=None, max_len=60):
    matches = matches or MATCHES
    fold_dir = REPORTS / 'crossfit_hsmm'
    fold_dir.mkdir(parents=True, exist_ok=True)
    # Reuse dead fold models from crossfit_dead when present.
    dead_dir = REPORTS / 'crossfit_dead'
    rows = []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        dur_path = fold_dir / ('duration_' + holdout + '.npz')
        if not dur_path.exists():
            logp = fit_duration_from_games(load_match, train_games, max_len=100)
            np.savez_compressed(dur_path, H=logp['H'], A=logp['A'], D=logp['D'])
        dead_path = dead_dir / ('dead_' + holdout + '.json')
        if not dead_path.exists():
            model, prior = train_dead(train_games)
            joblib.dump(model, dead_dir / ('dead_' + holdout + '.joblib'))
            export_dead(model, dead_path, train_games, prior=prior)
        data = load_match(holdout)
        f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
        ref = predict_version(f, graded, (2, 3))
        team = v23_team_logits(f, graded, nodes, ref)
        dead = predict_dead_logits(f, dead_path)
        with np.load(dur_path) as z:
            duration = {s: z[s].copy() for s in ('H', 'A', 'D')}
        labels = decode_hsmm(team, dead, graded, f.get('period'),
                             duration_logp=duration, max_len=max_len, stride=5)
        labels = _merge_short_spells(labels, graded, 18)
        metrics = score_natural(gt, labels, f)
        rows.append(dict(game=holdout, candidate='hsmm_v23dead', **metrics))
        print(holdout, round(metrics['spell_f1_mean'], 4), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crossfit', action='store_true')
    args = parser.parse_args()
    rows = crossfit()
    write_csv(REPORTS / 'crossfit_hsmm.csv', rows)
    summary = summarize_rows(rows)
    Path(REPORTS / 'crossfit_hsmm.json').write_text(
        json.dumps(dict(aggregate=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
