"""B/C evaluate structure-preserving corrector params with full v2.3 kwargs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'starter'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'starter' / 'external'))

from eval_external import kwargs_for_version, write_csv, aggregate_by_candidate
from error_corrector_model import V25_CORRECTOR_PATH
from evaluate_approved import (
    score_match, gate_report, ALL_MATCHES, SUITE_MATCHES)


def make_cand(margin, window=0, min_island=1, structure_guard=False):
    kw = kwargs_for_version(2, 3)
    kw['error_corrector_model'] = V25_CORRECTOR_PATH
    kw['error_corrector_margin'] = float(margin)
    kw['error_corrector_window'] = int(window)
    kw['error_corrector_min_island'] = int(min_island)
    kw['error_corrector_structure_guard'] = bool(structure_guard)
    return kw


def run_one(name, kw, ref_kw, out_prefix):
    rows = []
    for game in ALL_MATCHES:
        rows.extend(score_match(game, [(name, kw), ('reference', ref_kw)]))
    write_csv(Path('evaluation/reports') / (out_prefix + '.csv'), rows)
    report = gate_report(rows, name, 'reference')
    Path('evaluation/reports').joinpath(out_prefix + '_gate.json').write_text(
        json.dumps(dict(aggregate=aggregate_by_candidate(rows), gate=report),
                   indent=2) + '\n')
    g = report
    print(name, 'GATE', g['passed'],
          'suite_spell', g['suite_spell_ok'],
          'suite_hid', g['suite_hid_ok'],
          'loo', g['loo_ok'],
          'natural', g['natural_ok'])
    print('suite', g['suite'])
    return report


def main():
    ref = kwargs_for_version(2, 3)
    configs = [
        ('v24_m15', make_cand(1.5, 0, 1, False), 'spc_bc_v24_m15'),
        ('spc_m3', make_cand(3.0, 0, 1, False), 'spc_bc_m3'),
    ]
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        configs = [c for c in configs if c[0] in want]
    for name, kw, prefix in configs:
        run_one(name, kw, ref, prefix)


if __name__ == '__main__':
    main()
