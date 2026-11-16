"""How much spellF1 does the short-spell merge cap? External matches only.

Reports, per match (natural tracking):
  * share of GT H/A spells (and frames) shorter than 8/12/18/25 frames
  * recall of GT short spells by the anchor prediction
  * oracle spellF1 of GT merged at each threshold (the cap the merge imposes)
  * anchor spellF1 with min_spell_frames swept over THRESHOLDS
The sweep is chosen on FIT matches only; evaluation matches are reported.
No BAR-ATH labels are read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import MATCHES, REPORTS, load_match, score_natural
from eval_external import kwargs_for_version, run_labeller
from state_machine import _merge_short_spells

THRESHOLDS = (8, 12, 18, 25)
FIT = ('J03WOH', 'J03WN1', 'J03WOY', 'Sample_Game_3')


def live_runs(labels, graded):
    runs, i, n = [], 0, len(labels)
    while i < n:
        if not graded[i] or labels[i] not in ('H', 'A'):
            i += 1
            continue
        j = i + 1
        while j < n and graded[j] and labels[j] == labels[i]:
            j += 1
        runs.append((i, j, labels[i]))
        i = j
    return runs


def match_report(game, version):
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    runs = live_runs(gt, graded)
    lengths = np.array([b - a for a, b, _ in runs])
    live_frames = max(int(lengths.sum()), 1)
    base_kw = kwargs_for_version(*version)
    preds = {}
    for k in THRESHOLDS:
        kw = dict(base_kw, min_spell_frames=k)
        preds[k] = run_labeller(f, graded, kw)
    anchor = preds[base_kw.get('min_spell_frames', 18)]
    out = dict(game=game, n_gt_spells=len(runs))
    for k in THRESHOLDS:
        short = [(a, b, t) for a, b, t in runs if b - a < k]
        hit = sum(1 for a, b, t in short if np.mean(anchor[a:b] == t) > .5)
        out['short{}_share'.format(k)] = len(short) / max(len(runs), 1)
        out['short{}_frame_share'.format(k)] = float(
            sum(b - a for a, b, _ in short)) / live_frames
        out['short{}_recall'.format(k)] = hit / len(short) if short else float('nan')
        oracle = _merge_short_spells(np.asarray(gt).copy(), graded, k)
        out['oracle{}_spell'.format(k)] = score_natural(gt, oracle, f)['spell_f1_mean']
        m = score_natural(gt, preds[k], f)
        out['pred{}_spell'.format(k)] = m['spell_f1_mean']
        out['pred{}_hidden'.format(k)] = m['hidden_team_acc']
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items()}), flush=True)
    return out


def summarize(rows, games):
    pick = [r for r in rows if r['game'] in games]
    keys = [k for k in rows[0] if k not in ('game',)]
    return {k: float(np.nanmean([r[k] for r in pick])) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='2.6')
    parser.add_argument('--out', default=str(REPORTS / 'short_spell_ceiling.json'))
    args = parser.parse_args()
    version = tuple(int(p) for p in args.version.split('.'))
    rows = [match_report(g, version) for g in MATCHES]
    fit = summarize(rows, FIT)
    held = summarize(rows, [g for g in MATCHES if g not in FIT])
    best = max(THRESHOLDS, key=lambda k: (fit['pred{}_spell'.format(k)], k == 18))
    result = dict(version=args.version, fit_matches=list(FIT), thresholds=THRESHOLDS,
                  fit_mean=fit, heldout_mean=held, all_mean=summarize(rows, MATCHES),
                  best_min_spell_on_fit=best, rows=rows)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    for name, s in (('fit', fit), ('heldout', held)):
        print(name, ' '.join('k{}: oracle {:.4f} pred {:.4f} hid {:.4f} short {:.3f}'.format(
            k, s['oracle{}_spell'.format(k)], s['pred{}_spell'.format(k)],
            s['pred{}_hidden'.format(k)], s['short{}_share'.format(k)]) for k in THRESHOLDS))
    print('best min_spell_frames on fit:', best)


if __name__ == '__main__':
    main()
