"""Duel-margin external eval: shipped touch model, decoder-only change.

Ref = v2.6 pipeline per match; shipped event_first_v1 logits; decode at
base margin 1.0 vs duel margin 0.5. Eval matches are never in event-first
training (FIT + SC train). Writes touch_duel_margin_eval.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, predict_version, score_natural
from error_corrector_model import ensemble_team_logits
from event_first_model import load_event_first, touch_candidates, touch_design
from boosted_dead_model import boosted_logits
from touch_duel_margin_experiment import decode_duel_margin

GAMES = ['Sample_Game_2', 'J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9',
         'SC_1953632', 'SC_2006229', 'SC_2007448', 'SC_2013725']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_duel_margin_eval.json'))
    args = parser.parse_args()
    model = load_event_first()
    rows = []
    for game in GAMES:
        d = load_match(game)
        f, gt, graded = d['features'], d['gt'], d['graded']
        ref = predict_version(f, graded, (2, 6))
        logits_team = ensemble_team_logits(f, graded, d['nodes'], ref)
        idx, kin = touch_candidates(f, graded)
        x = touch_design(f, ref, logits_team, idx, kin)
        logit = boosted_logits(x, model)
        dd = dict(f=f, gt=gt, graded=graded, ref=ref, idx=idx)
        for name, dm in (('base_m1.0', 1.0), ('duel_m0.5', 0.5)):
            labels = decode_duel_margin(dd, logit, dm)
            m = score_natural(gt, labels, f)
            rows.append(dict(game=game, candidate=name, **m))
            print(game, name, round(m['spell_f1_mean'], 4),
                  round(m['hidden_team_acc'], 4) if m['hidden_team_acc'] == m['hidden_team_acc'] else 'nan',
                  flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')
    for name in ('base_m1.0', 'duel_m0.5'):
        vals = [r for r in rows if r['candidate'] == name]
        print(name, 'mean spell', round(float(np.mean([r['spell_f1_mean'] for r in vals])), 4),
              'mean hidden', round(float(np.nanmean([r['hidden_team_acc'] for r in vals])), 4))


if __name__ == '__main__':
    main()
