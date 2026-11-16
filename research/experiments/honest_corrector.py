"""Error corrector trained and tuned on the fit set only (no Track B/C matches).

Margin is chosen by leave-one-match-out over FIT, scored with the shipped
NumPy apply path. The final model is trained on all of FIT.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, score_natural, predict_version
from error_corrector import train, export_model
from error_corrector_model import (
    design_corrector_features, ensemble_team_logits, load_corrector)
from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells

FIT = ['J03WOH', 'J03WN1', 'J03WOY', 'Sample_Game_3']
MARGINS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
FOLD_DIR = REPORTS / 'honest_corrector'
SHIP_PATH = Path('starter/models/error_corrector_v2.json')


def corrected(features, reference, graded, corr, margin, minimum=18):
    labels = np.asarray(reference).copy()
    live = graded & np.asarray(features['ball_present'], bool) & np.isin(labels, ['H', 'A'])
    corr_home = corr >= 0
    want = live & (corr_home != (labels == 'H')) & (np.abs(corr) >= margin)
    labels[want] = np.where(corr_home[want], 'H', 'A')
    return _merge_short_spells(labels, graded, minimum), int(want.sum())


def fold_rows(holdout, margins=MARGINS, seed=73):
    path = FOLD_DIR / 'corr_{}.json'.format(holdout)
    train_games = [g for g in FIT if g != holdout]
    if not path.exists():
        FOLD_DIR.mkdir(parents=True, exist_ok=True)
        export_model(train(train_games, seed=seed), path, train_games)
    data = load_match(holdout)
    f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
    base = predict_version(f, graded, (2, 3))
    corr = boosted_logits(
        design_corrector_features(f, base, ensemble_team_logits(f, graded, nodes, base)),
        load_corrector(str(path)))
    rows = [dict(game=holdout, candidate='v2_3', flips=0, **score_natural(gt, base, f))]
    for margin in margins:
        pred, flips = corrected(f, base, graded, corr, margin)
        rows.append(dict(game=holdout, candidate='m{}'.format(margin), flips=flips,
                         **score_natural(gt, pred, f)))
    for row in rows:
        print(holdout, row['candidate'], 'flips', row['flips'],
              'spell', round(row['spell_f1_mean'], 4),
              'hidden', round(row['hidden_team_acc'], 4), flush=True)
    return rows


def choose_margin(rows):
    """Best mean spellF1 with mean hidden >= v2.3 and no fold worse than -0.005."""
    by = {}
    for row in rows:
        by.setdefault(row['candidate'], {})[row['game']] = row
    base = by['v2_3']
    mean = lambda cand, key: float(np.nanmean([by[cand][g][key] for g in FIT]))
    best, best_spell = None, mean('v2_3', 'spell_f1_mean')
    table = {}
    for cand in by:
        worst = min(by[cand][g]['spell_f1_mean'] - base[g]['spell_f1_mean'] for g in FIT)
        table[cand] = dict(spell=mean(cand, 'spell_f1_mean'),
                           hidden=mean(cand, 'hidden_team_acc'), worst_delta=worst)
        if cand == 'v2_3':
            continue
        ok = (table[cand]['hidden'] + 1e-9 >= mean('v2_3', 'hidden_team_acc')
              and worst >= -0.005)
        if ok and table[cand]['spell'] > best_spell:
            best, best_spell = cand, table[cand]['spell']
    return (float(best[1:]) if best else None), table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'honest_corrector.json'))
    args = parser.parse_args()
    rows = [r for g in FIT for r in fold_rows(g)]
    margin, table = choose_margin(rows)
    print(json.dumps(dict(chosen_margin=margin, table=table), indent=2))
    if margin is not None:
        export_model(train(FIT), SHIP_PATH, FIT)
        data = json.loads(SHIP_PATH.read_text())
        data.update(margin=margin, fit_matches=FIT,
                    selection='leave-one-match-out over fit matches only',
                    excluded_evaluation_matches=['Sample_Game_1', 'Sample_Game_2',
                                                 'J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9'])
        SHIP_PATH.write_text(json.dumps(data, separators=(',', ':')) + '\n')
    Path(args.out).write_text(json.dumps(
        dict(fit=FIT, chosen_margin=margin, table=table, rows=rows), indent=2))


if __name__ == '__main__':
    main()
