"""Does SkillCorner training data help the ownership models? Like-for-like test.

For each FIT holdout, retrain with and without the 16 SkillCorner training
matches (down-weighted) and score the holdout:
  * boundary ownership model (replaces the v2.3 boundary npz)
  * live-team error corrector (applied to v2.3 at the honest margin)
Track B/C/D matches are never trained on or used for selection.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, score_natural, predict_version
from boundary_crossfit import run_fold, DEFAULT_CONTROL
from error_corrector import training_matrix, export_model
from honest_corrector import FIT, corrected
from error_corrector_model import (
    design_corrector_features, ensemble_team_logits, load_corrector,
    HONEST_CORRECTOR_MARGIN)
from boosted_dead_model import boosted_logits
from event_first import skillcorner_train

OUT_DIR = REPORTS / 'skillcorner_retrain'


def boundary_rows(sc_sampling):
    sc = skillcorner_train()
    rows = []
    for holdout in FIT:
        base = [g for g in FIT if g != holdout]
        for tag, games, kw in (
                ('fit_nosc', base, dict(DEFAULT_CONTROL)),
                ('fit_sc{}'.format(sc_sampling), base + sc,
                 dict(DEFAULT_CONTROL, sc_sampling=sc_sampling))):
            row = run_fold(holdout, tag, kw, train_games=games)
            rows.append(dict(model='boundary', candidate=tag, game=holdout,
                             spell_f1_mean=row['spell_f1_mean'],
                             hidden_team_acc=row['hidden_team_acc'],
                             delta_v23=row['delta_v23']))
    return rows


def corrector_rows(sc_weight, margin=HONEST_CORRECTOR_MARGIN):
    sc = skillcorner_train()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}

    def matrix(games):
        key = tuple(games)
        if key not in cache:
            cache[key] = training_matrix(list(games))
        return cache[key]

    rows = []
    for holdout in FIT:
        base = [g for g in FIT if g != holdout]
        data = load_match(holdout)
        f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
        ref = predict_version(f, graded, (2, 3))
        logits = ensemble_team_logits(f, graded, nodes, ref)
        rows.append(dict(model='corrector', candidate='v2_3', game=holdout,
                         **score_natural(gt, ref, f)))
        for tag, games, weight in (('fit_nosc', base, 1.), ('fit_sc{}'.format(sc_weight),
                                                           base + sc, sc_weight)):
            xs, ys, ws = [], [], []
            for g in games:
                x, y = matrix((g,))
                xs.append(x)
                ys.append(y)
                ws.append(np.full(len(y), weight if g.startswith('SC_') else 1.))
            model = HistGradientBoostingClassifier(
                learning_rate=.05, max_iter=250, max_leaf_nodes=63, min_samples_leaf=30,
                l2_regularization=1., max_bins=127, early_stopping=False, random_state=73)
            with threadpool_limits(limits=4):
                model.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=np.concatenate(ws))
            path = OUT_DIR / 'corr_{}_{}.json'.format(tag, holdout)
            export_model(model, path, games)
            corr = boosted_logits(design_corrector_features(f, ref, logits),
                                  load_corrector(str(path)))
            pred, flips = corrected(f, ref, graded, corr, margin)
            rows.append(dict(model='corrector', candidate=tag, game=holdout, flips=flips,
                             **score_natural(gt, pred, f)))
            print('corrector', tag, holdout, flips, round(rows[-1]['spell_f1_mean'], 4),
                  flush=True)
    return rows


def summarize(rows):
    out = {}
    for r in rows:
        out.setdefault(r['model'], {}).setdefault(r['candidate'], []).append(r)
    return {m: {c: dict(spell=float(np.mean([r['spell_f1_mean'] for r in v])),
                        hidden=float(np.nanmean([r['hidden_team_acc'] for r in v])),
                        n=len(v)) for c, v in cands.items()} for m, cands in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sc-weight', type=float, default=0.25)
    parser.add_argument('--skip-boundary', action='store_true')
    parser.add_argument('--out', default=str(REPORTS / 'skillcorner_retrain.json'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = corrector_rows(args.sc_weight)
    if not args.skip_boundary:
        rows += boundary_rows(args.sc_weight)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps(dict(sc_weight=args.sc_weight, fit=FIT,
                                              skillcorner_train=skillcorner_train(),
                                              summary=summary, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
