"""Cross-fitting harness: score each match with models that never trained on it.

For frozen approved models (v2.3) and smooth50, "crossfit" is just natural
scoring on all matches (they were not trained on the hold-out set in the
original protocol for most matches; we still report the equal-match mean).

For trainable candidates, pass a `train_fn(train_games) -> predict_fn` factory.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import (
    MATCHES, METRICA, SPORTEC, REPORTS, load_match, fold_train_games,
    score_natural, summarize_rows, predict_version, predict_smooth50)
from eval_external import write_csv
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import score_one, kwargs_for_version, run_labeller


def score_anchors(matches=None):
    matches = matches or MATCHES
    rows = []
    for game in matches:
        print('anchor', game, flush=True)
        data = load_match(game)
        f, gt, graded = data['features'], data['gt'], data['graded']
        for name, pred in (
            ('v23', predict_version(f, graded, (2, 3))),
            ('smooth50', predict_smooth50(f)),
        ):
            metrics = score_natural(gt, pred, f)
            row = dict(game=game, candidate=name, protocol='natural', **metrics)
            # Sportec natural hidden is usually empty; add a fixed 5s dropout
            # hidden accuracy so Sportec is not ignored.
            if game in SPORTEC:
                hide5 = block_dropout(f['n_frames'], 5, seed=41)
                dropped = apply_dropout(f, hide5)
                if name == 'v23':
                    pred5 = run_labeller(dropped, graded, kwargs_for_version(2, 3))
                else:
                    pred5 = predict_smooth50(dropped)
                m5 = score_one(gt, pred5, hide5 | ~f['ball_present'])
                row['hidden_5s_acc'] = m5['hidden_team_acc']
                row['hidden_5s_n'] = m5['hidden_n']
            rows.append(row)
    return rows


def crossfit_train_predict(train_fn, matches=None, candidate_name='cand'):
    """train_fn(train_games) -> callable(features, graded, nodes) -> labels."""
    matches = matches or MATCHES
    rows = []
    for holdout in matches:
        train_games = fold_train_games(holdout)
        print('fold holdout', holdout, 'train', len(train_games), flush=True)
        predict_fn = train_fn(train_games)
        data = load_match(holdout)
        pred = predict_fn(data['features'], data['graded'], data['nodes'])
        metrics = score_natural(data['gt'], pred, data['features'])
        rows.append(dict(game=holdout, candidate=candidate_name,
                         protocol='natural', **metrics))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=MATCHES)
    parser.add_argument('--out', default=str(REPORTS / 'crossfit_anchors.csv'))
    parser.add_argument('--summary', default=str(REPORTS / 'crossfit_anchors.json'))
    args = parser.parse_args()
    rows = score_anchors(args.matches)
    write_csv(args.out, rows)
    summary = dict(
        aggregate=summarize_rows(rows),
        per_game={r['candidate']: {} for r in rows},
    )
    for r in rows:
        summary['per_game'].setdefault(r['candidate'], {})[r['game']] = dict(
            spell_f1_mean=r['spell_f1_mean'],
            hidden_team_acc=r['hidden_team_acc'],
            hidden_n=r['hidden_n'],
            hidden_5s_acc=r.get('hidden_5s_acc'),
        )
    # Metrica-only hidden headline
    metrica_hidden = [
        r['hidden_team_acc'] for r in rows
        if r['candidate'] == 'v23' and r['game'] in METRICA
        and r['hidden_team_acc'] == r['hidden_team_acc']]
    summary['v23_metrica_mean_hidden'] = float(np.mean(metrica_hidden)) if metrica_hidden else None
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['aggregate'], indent=2))
    print('v23_metrica_mean_hidden', summary['v23_metrica_mean_hidden'])


if __name__ == '__main__':
    main()
