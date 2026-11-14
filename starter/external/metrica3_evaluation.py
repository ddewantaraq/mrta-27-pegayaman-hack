"""One reserved-match evaluation of the previously frozen ownership candidate.

JSON-to-legacy reference conversion only: uses exactly labels_from_events,
including its known limitations, rather than changing reference semantics.
"""
import hashlib
import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrica3_geometry import ROOT, cached_geometry
from metrica_gt import labels_from_events
from player_node_data import load_player_nodes
from sportec_experiment import block_dropout
from dropout import apply_dropout
from eval_external import kwargs_for_version, run_labeller, score_one, write_csv, aggregate_by_candidate
from numpy_set_temporal import load_parameters, predict_logits
from state_machine import _merge_short_spells


def reference(root, n_frames):
    meta = ET.parse(Path(root)/'Sample_Game_3_metadata.xml').getroot()
    team_ids = [t.get('id') for t in meta.iter('Team')]
    if len(team_ids) != 2:
        raise ValueError('Expected two teams in EPTS metadata')
    data = json.loads((Path(root)/'Sample_Game_3_events.json').read_text())['data']
    events = []
    for row in data:
        team = row.get('team')
        if team is None or team['id'] not in team_ids:
            raise ValueError('Unknown event team')
        events.append({'Team': 'Home' if team['id'] == team_ids[0] else 'Away',
                       'Type': row['type']['name'].strip().upper(),
                       'Subtype': '', 'Period': int(row['period']),
                       'Start Frame': int(row['start']['frame']),
                       'End Frame': int((row.get('end') or row['start'])['frame'])})
    return labels_from_events(events, n_frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', default='evaluation/reports/set_temporal_frozen_decision.json')
    parser.add_argument('--policy', default='full', choices=('full','run90_mean90'))
    parser.add_argument('--out-prefix', default='evaluation/reports/set_temporal_reserved')
    parser.add_argument('--residual-model-file',default=None)
    args = parser.parse_args()
    decision = json.loads(Path(args.frozen).read_text())
    for path, expected in decision['hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen candidate changed: '+path)
    f, graded = cached_geometry()
    gt = reference(ROOT, f['n_frames'])
    # Previous Metrica runs pass all tracking frames to inference. score_one
    # excludes reference X internally; never reveal event boundaries to models.
    graded = np.ones(f['n_frames'], bool)
    nodes = load_player_nodes('Sample_Game_3')
    model = load_parameters('evaluation/reports/set_temporal_v1.npz')
    rows = []
    for sec in (0, 2, 5, 10, 30):
        hide = np.zeros(len(gt), bool) if sec == 0 else block_dropout(len(gt), sec, seed=41)
        dropped = apply_dropout(f, hide)
        baseline = run_labeller(dropped, graded, kwargs_for_version(1, 10))
        if args.residual_model_file:
            from residual_ownership_model import apply_residual_ownership
            pred = apply_residual_ownership(dropped,baseline,graded,nodes,args.residual_model_file)
        else:
            logits = predict_logits(dropped, graded, nodes, model)
            pred = baseline.copy()
            mask = graded & np.isin(baseline, ['H', 'A'])
            if args.policy != 'full':
                from ownership_gates import replacement_mask
                mask = replacement_mask(dropped, baseline, graded, logits, args.policy)
            pred[mask] = np.where(logits[mask] >= 0, 'H', 'A')
            pred = _merge_short_spells(pred, graded, 18)
        for name, labels in [('reference', baseline), ('frozen_full', pred)]:
            rows.append(dict(game='Sample_Game_3', dropout=str(sec), candidate=name,
                             **score_one(gt, labels, hide | ~f['ball_present'])))
        print('reserved evaluation', sec, flush=True)
    write_csv(args.out_prefix+'.csv', rows)
    summary = aggregate_by_candidate(rows)
    print(json.dumps(summary, indent=2))
    old, new = summary['reference'], summary['frozen_full']
    passed = (new['mean_spell_f1'] > old['mean_spell_f1'] and
              new['mean_hidden_team_acc'] >= old['mean_hidden_team_acc'])
    if decision.get('require_natural_nonregression'):
        natural = {r['candidate']: r for r in rows if r['dropout'] == '0'}
        passed &= (natural['frozen_full']['spell_f1_mean'] >= natural['reference']['spell_f1_mean']
                   and natural['frozen_full']['hidden_team_acc'] >= natural['reference']['hidden_team_acc'])
    Path(args.out_prefix+'_decision.json').write_text(
        json.dumps(dict(passed=passed, scores=summary, frozen_decision=decision), indent=2))
    print('Reserved promotion criteria passed:', passed)


if __name__ == '__main__':
    main()
