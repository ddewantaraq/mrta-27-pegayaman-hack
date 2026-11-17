"""P4 probe: are missing-span team errors onset handoff or mid-span drift?

For each hidden live span (synthetic 5s-block dropout, plus natural
missing on Metrica), report wrong-team rate by distance from span start
and the pure-drift span count (onset correct, later wrong). If errors are
onset-dominated, mid-span models (possessor heads, imputation) cannot
help; fix the visible handoff instead. Diagnostics only, v2.7 labeller.
Writes evaluation/reports/missing_span_audit.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import kwargs_for_version, run_labeller

MATCHES = ['J03WMX', 'J03WPY', 'J03WQQ', 'J03WR9', 'Sample_Game_2',
           'Sample_Game_3']


def spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def audit(game, hide, label):
    data = load_match(game)
    f, gt, graded = data['features'], np.asarray(data['gt']), data['graded']
    dropped = apply_dropout(dict(f), np.asarray(hide, bool))
    pred = np.asarray(run_labeller(dropped, graded, kwargs_for_version(2, 7)))
    hidden = np.asarray(hide, bool) & graded & np.isin(gt, ['H', 'A'])
    buckets = {'d0_4': [0, 0], 'd5_24': [0, 0], 'd25p': [0, 0]}
    n_spans = n_pure_drift = n_onset_wrong = 0
    for a, b in spans(hidden):
        seg_gt, seg_pred = gt[a:b], pred[a:b]
        right = seg_pred == seg_gt
        n_spans += 1
        onset_ok = bool(right[:5].all())
        if not onset_ok:
            n_onset_wrong += 1
        elif not bool(right.all()):
            n_pure_drift += 1
        for d, ok in enumerate(right):
            key = 'd0_4' if d < 5 else ('d5_24' if d < 25 else 'd25p')
            buckets[key][1] += 1
            buckets[key][0] += int(ok)
    return dict(game=game, condition=label, n_spans=n_spans,
                n_onset_wrong_spans=n_onset_wrong, n_pure_drift_spans=n_pure_drift,
                acc_by_distance={k: (v[0] / v[1] if v[1] else None)
                                 for k, v in buckets.items()},
                n_frames={k: v[1] for k, v in buckets.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'missing_span_audit.json'))
    args = parser.parse_args()
    rows = []
    for game in MATCHES:
        data = load_match(game)
        f = data['features']
        rows.append(audit(game, block_dropout(f['n_frames'], 5, seed=41), 'drop5s'))
        natural = ~np.asarray(f['ball_present'], bool)
        if natural.any():
            rows.append(audit(game, natural, 'natural'))
        print('audited', game, flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(r['game'], r['condition'], 'spans', r['n_spans'],
              'onset_wrong', r['n_onset_wrong_spans'],
              'pure_drift', r['n_pure_drift_spans'],
              {k: round(v, 3) if v is not None else None
               for k, v in r['acc_by_distance'].items()})


if __name__ == '__main__':
    main()
