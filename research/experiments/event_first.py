"""Train and cross-fit the event-first touch model (starter/event_first_model.py).

Touch team labels come from external ground truth only (Sportec BallPossession,
Metrica events, SkillCorner possession.group on the training split). Decoder
switch cost and margin are chosen by leave-one-match-out over FIT matches.
Track B/C/D matches are never used for training or selection.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, score_natural, predict_version
from error_corrector_model import ensemble_team_logits
from event_first_model import (FEATURE_VERSION, touch_candidates, touch_design,
                               decode_event_first)
from boosted_dead_model import boosted_logits
from state_machine import _merge_short_spells

FIT = ['J03WOH', 'J03WN1', 'J03WOY', 'Sample_Game_3']
ANCHOR = (2, 6)
LABEL_LEAD = 3
SWITCH_COSTS = (0.0, 0.5, 1.0)
MARGINS = (1.0, 2.0, 3.0, 4.0, 6.0)
CACHE = REPORTS / 'event_first'
SHIP_PATH = Path('starter/models/event_first_v1.json')


def skillcorner_train():
    split = Path('starter/external/skillcorner_split.json')
    return json.loads(split.read_text())['train'] if split.exists() else []


def game_cache(game):
    path = CACHE / '{}_v{}_{}.npz'.format(game, *ANCHOR)
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            ref, logits = z['reference'], z['team_logits']
    else:
        ref = predict_version(f, graded, ANCHOR)
        logits = ensemble_team_logits(f, graded, data['nodes'], ref)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, reference=np.asarray(ref, dtype='<U1'), team_logits=logits)
    idx, kin = touch_candidates(f, graded)
    x = touch_design(f, ref, logits, idx, kin)
    lead = gt[np.clip(idx + LABEL_LEAD, 0, len(gt) - 1)]
    keep = np.isin(lead, ['H', 'A'])
    return dict(game=game, f=f, gt=gt, graded=graded, ref=ref, idx=idx, x=x,
                y=(lead == 'H').astype(float), keep=keep)


def fit_model(items, seed=29, sc_weight=1.0):
    x = np.concatenate([d['x'][d['keep']] for d in items])
    y = np.concatenate([d['y'][d['keep']] for d in items])
    w = np.concatenate([np.full(int(d['keep'].sum()),
                                sc_weight if d['game'].startswith('SC_') else 1.)
                        for d in items])
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_iter=200, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1., max_bins=127, early_stopping=False, random_state=seed)
    with threadpool_limits(limits=4):
        model.fit(x, y, sample_weight=w)
    return model


def packed(model):
    trees = []
    for iteration in model._predictors:
        nodes = iteration[0].nodes
        fields = dict(feature='feature_idx', threshold='num_threshold', value='value',
                      left='left', right='right', leaf='is_leaf', missing_left='missing_go_to_left')
        trees.append({k: nodes[s].tolist() for k, s in fields.items()})
    return dict(baseline=float(model._baseline_prediction[0, 0]), trees=trees)


def as_arrays(pack):
    return dict(baseline=pack['baseline'], trees=[
        {k: np.asarray(v, dtype=np.float64 if k in ('threshold', 'value') else np.int32)
         for k, v in t.items()} for t in pack['trees']])


def parity(model, x):
    """Max |sklearn raw score - NumPy tree logit|."""
    if not len(x):
        return 0.
    expected = model._raw_predict(x)[:, 0]
    return float(np.max(np.abs(expected - boosted_logits(x, as_arrays(packed(model))))))


def decode(d, logit, cost, margin):
    labels = decode_event_first(d['f'], d['ref'], d['graded'], logit, d['idx'], cost, margin)
    return _merge_short_spells(labels, d['graded'], 18)


def score_holdout(d, model, domain):
    logit = boosted_logits(d['x'], as_arrays(packed(model)))
    acc = float(np.mean((logit[d['keep']] >= 0) == (d['y'][d['keep']] > .5)))
    base = score_natural(d['gt'], d['ref'], d['f'])
    rows = [dict(game=d['game'], domain=domain, candidate='anchor', touch_acc=acc,
                 parity=parity(model, d['x']), **base)]
    for cost in SWITCH_COSTS:
        for margin in MARGINS:
            m = score_natural(d['gt'], decode(d, logit, cost, margin), d['f'])
            rows.append(dict(game=d['game'], domain=domain,
                             candidate='c{}_m{}'.format(cost, margin), touch_acc=acc, **m))
    print(domain, d['game'], 'touches', int(d['keep'].sum()), 'acc', round(acc, 4),
          'anchor', round(base['spell_f1_mean'], 4),
          'best', round(max(r['spell_f1_mean'] for r in rows), 4), flush=True)
    return rows


def lomo(items, extra, sc_weight=1.0, sc_groups=4):
    """Fit-match LOMO plus grouped folds over the SkillCorner training split."""
    pool = extra if sc_weight > 0 else []
    rows = []
    for d in items:
        train = [e for e in items if e['game'] != d['game']] + pool
        rows += score_holdout(d, fit_model(train, sc_weight=sc_weight), 'fit')
    for k in range(sc_groups):
        group = extra[k::sc_groups]
        names = {g['game'] for g in group}
        train = items + [e for e in pool if e['game'] not in names]
        model = fit_model(train, sc_weight=sc_weight)
        for d in group:
            rows += score_holdout(d, model, 'sc')
    return rows


def choose(rows):
    """Best fit-mean spellF1 that is also no worse on the SkillCorner folds."""
    by = {}
    for r in rows:
        by.setdefault(r['candidate'], {}).setdefault(r['domain'], {})[r['game']] = r
    anchor = by['anchor']

    def mean(c, dom, key):
        cells = by[c].get(dom, {})
        return float(np.nanmean([v[key] for v in cells.values()])) if cells else float('nan')

    table, best, best_spell = {}, None, mean('anchor', 'fit', 'spell_f1_mean')
    for c in by:
        worst = min(by[c]['fit'][g]['spell_f1_mean'] - anchor['fit'][g]['spell_f1_mean']
                    for g in FIT)
        table[c] = dict(spell=mean(c, 'fit', 'spell_f1_mean'),
                        hidden=mean(c, 'fit', 'hidden_team_acc'), worst_delta=worst,
                        sc_spell=mean(c, 'sc', 'spell_f1_mean'),
                        sc_hidden=mean(c, 'sc', 'hidden_team_acc'))
        if c == 'anchor':
            continue
        sc_ok = 'sc' not in by[c] or (
            table[c]['sc_spell'] + 1e-9 >= mean('anchor', 'sc', 'spell_f1_mean')
            and table[c]['sc_hidden'] + 1e-9 >= mean('anchor', 'sc', 'hidden_team_acc'))
        if (worst >= -0.005 and sc_ok and table[c]['spell'] > best_spell
                and table[c]['hidden'] + 1e-9 >= mean('anchor', 'fit', 'hidden_team_acc')):
            best, best_spell = c, table[c]['spell']
    return best, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-skillcorner', action='store_true')
    parser.add_argument('--sc-weight', type=float, default=1.0)
    parser.add_argument('--no-export', action='store_true')
    parser.add_argument('--out', default=str(REPORTS / 'event_first_crossfit.json'))
    args = parser.parse_args()
    items = [game_cache(g) for g in FIT]
    sc = [] if args.no_skillcorner else [game_cache(g) for g in skillcorner_train()]
    rows = lomo(items, sc, args.sc_weight)
    best, table = choose(rows)
    print(json.dumps(dict(best=best, anchor=table['anchor'],
                          best_row=table.get(best)), indent=2))
    result = dict(fit=FIT, skillcorner_train=[d['game'] for d in sc], anchor=list(ANCHOR),
                  sc_weight=args.sc_weight, best=best, table=table, rows=rows)
    if best is not None and not args.no_export:
        cost, margin = (float(p[1:]) for p in best.split('_'))
        pool = sc if args.sc_weight > 0 else []
        model = fit_model(items + pool, sc_weight=args.sc_weight)
        pack = packed(model)
        all_x = np.concatenate([d['x'] for d in items])
        result['ship_parity_max_abs'] = parity(model, all_x)
        SHIP_PATH.write_text(json.dumps(dict(
            feature_version=FEATURE_VERSION, switch_cost=cost, margin=margin,
            n_features=int(model.n_features_in_), label_lead_frames=LABEL_LEAD,
            skillcorner_sample_weight=args.sc_weight if pool else None,
            selection_matches=FIT + [d['game'] for d in sc],
            training_matches=FIT + [d['game'] for d in pool],
            data_sources=['https://doi.org/10.1038/s41597-025-04505-y',
                          'https://github.com/metrica-sports/sample-data',
                          'https://github.com/SkillCorner/opendata'],
            selection='LOMO over fit matches + grouped folds over SkillCorner training split',
            **pack),
            separators=(',', ':')) + '\n')
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
