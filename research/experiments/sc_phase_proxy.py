"""P6 probe: are SkillCorner phases_of_play a dead-ball proxy?

SC has no dead-ball label: frames without possession.group are X (~45%).
If group gaps align with phase gaps, phases give a stoppage proxy for
future SC dead-ball training. Analysis only, one train match, no fitting.
Writes evaluation/reports/sc_phase_proxy.json.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skillcorner_features import read_source

ROOT = Path('data_external/skillcorner')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', default='SC_1874553')
    parser.add_argument('--out', default=str('evaluation/reports/sc_phase_proxy.json'))
    args = parser.parse_args()
    sid = args.game[len('SC_'):]
    src = read_source(args.game, ROOT)
    in_period = src['period'] > 0
    group = src['group']
    n = len(group)
    in_phase = np.zeros(n, bool)
    n_phases = 0
    with open(ROOT / sid / (sid + '_phases_of_play.csv'), newline='') as handle:
        for row in csv.DictReader(handle):
            a, b = int(row['frame_start']), int(row['frame_end'])
            in_phase[max(a, 0):min(b + 1, n)] = True
            n_phases += 1
    live = in_period & (group > 0)
    gap = in_period & (group == 0)
    report = dict(
        game=args.game, n_phases=n_phases,
        p_gap_given_phase=float(gap[in_phase].mean()) if in_phase.any() else None,
        p_gap_given_no_phase=float(gap[~in_phase & in_period].mean()),
        p_phase_given_gap=float(in_phase[gap].mean()) if gap.any() else None,
        frac_gap=float(gap.sum() / in_period.sum()),
        frac_phase=float((in_phase & in_period).sum() / in_period.sum()))
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
