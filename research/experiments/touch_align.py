"""P1 step 1: align unused event files to touch candidates and GT switches.

J03WMX (Track C, eval-only) via Sportec events_raw XML; SC_1874553
(SkillCorner train split, never Track D) via dynamic_events.csv.
Parse-and-align report only: no fitting, no selection.
Writes evaluation/reports/touch_supervision_align.json.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match
from event_first_model import touch_candidates
from touch_supervision import (
    accel_candidates,
    align_hits,
    parse_skillcorner_dynamic,
    parse_sportec_events_raw,
    sportec_frame_offset,
)

SPORTEC_ROOT = Path('data_external/sportec')
SC_ROOT = Path('data_external/skillcorner')
TOLS = (5, 12)


def gt_switches(gt):
    gt = np.asarray(gt)
    live = np.isin(gt, ['H', 'A'])
    edge = live[:-1] & live[1:] & (gt[:-1] != gt[1:])
    return np.flatnonzero(edge) + 1


def team_agreement(touches, idx, gt, lead=3):
    gt = np.asarray(gt)
    pairs = [(t.team, gt[min(i + lead, len(gt) - 1)])
             for t, i in zip(touches, idx) if t.team is not None]
    pairs = [(team, g) for team, g in pairs if g in ('H', 'A')]
    if not pairs:
        return dict(n=0, agree=None)
    agree = sum((team == 0 and g == 'H') or (team == 1 and g == 'A')
                for team, g in pairs)
    return dict(n=len(pairs), agree=agree / len(pairs))


def analyze(game, touches, idx, features, gt, graded):
    n = len(gt)
    keep = [(t, i) for t, i in zip(touches, idx) if 0 <= i < n and graded[i]]
    touches = [t for t, _ in keep]
    idx = [i for _, i in keep]
    cands = touch_candidates(features, graded)[0].tolist()
    accel = accel_candidates(features, graded).tolist()
    union = sorted(set(cands) | set(accel))
    switches = gt_switches(gt).tolist()
    out = dict(n_touches=len(idx), kinds=dict(Counter(t.kind for t in touches)),
               n_candidates=len(cands), n_accel=len(accel), n_union=len(union),
               n_switches=len(switches))
    for tol in TOLS:
        pc, rc, _ = align_hits(cands, idx, tol)
        pa, ra, _ = align_hits(accel, idx, tol)
        pu, ru, _ = align_hits(union, idx, tol)
        pe, re, _ = align_hits(idx, switches, tol)
        ps, rs, _ = align_hits(cands, switches, tol)
        out['tol{}'.format(tol)] = dict(
            cand_vs_event=dict(precision=pc, recall=rc),
            accel_vs_event=dict(precision=pa, recall=ra),
            union_vs_event=dict(precision=pu, recall=ru),
            event_vs_switch=dict(precision=pe, recall=re),
            cand_vs_switch=dict(precision=ps, recall=rs))
    out['event_team_vs_gt_lead3'] = team_agreement(touches, idx, gt)
    for kind in sorted({t.kind for t in touches}):
        sub = [(t, i) for t, i in zip(touches, idx) if t.kind == kind]
        out['team_vs_gt_lead3_' + kind] = team_agreement(
            [t for t, _ in sub], [i for _, i in sub], gt)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'touch_supervision_align.json'))
    args = parser.parse_args()
    report = {}

    data = load_match('J03WOH')
    offset = sportec_frame_offset('J03WOH', SPORTEC_ROOT)
    touches = parse_sportec_events_raw('J03WOH', SPORTEC_ROOT)
    idx = [t.frame_n - offset for t in touches]
    report['J03WOH_fit'] = dict(offset=offset,
                                **analyze('J03WOH', touches, idx, data['features'],
                                          data['gt'], data['graded']))

    data = load_match('J03WMX')
    offset = sportec_frame_offset('J03WMX', SPORTEC_ROOT)
    touches = parse_sportec_events_raw('J03WMX', SPORTEC_ROOT)
    idx = [t.frame_n - offset for t in touches]
    report['J03WMX'] = dict(offset=offset,
                            **analyze('J03WMX', touches, idx, data['features'],
                                      data['gt'], data['graded']))

    data = load_match('SC_1874553')
    touches = parse_skillcorner_dynamic('SC_1874553', SC_ROOT)
    idx = [t.frame25 for t in touches]
    report['SC_1874553'] = analyze('SC_1874553', touches, idx, data['features'],
                                   data['gt'], data['graded'])

    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
