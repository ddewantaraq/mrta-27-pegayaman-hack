"""Label-noise floor + oracle stack for the 0.90 natural target.

No challenge ground truth. Counterfactual oracles are diagnostics, not models.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import MATCHES, load_match, score_natural, predict_version, REPORTS
from diagnose_v22 import confusion
from eval_external import score_one


def shift_labels(labels, shift):
    out = np.asarray(labels).copy()
    if shift == 0:
        return out
    n = len(out)
    if shift > 0:
        out[shift:] = labels[:n - shift]
        out[:shift] = labels[0]
    else:
        s = -shift
        out[:n - s] = labels[s:]
        out[n - s:] = labels[-1]
    return out


def perfect_team(pred, gt):
    out = np.asarray(pred).copy()
    live = np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
    out[live] = gt[live]
    return out


def perfect_dead(pred, gt):
    out = np.asarray(pred).copy()
    out[gt == 'D'] = 'D'
    restore = np.isin(gt, ['H', 'A']) & (pred == 'D')
    # Keep previous live team if available; else use GT team.
    out[restore] = gt[restore]
    return out


def perfect_both(pred, gt):
    out = perfect_dead(pred, gt)
    return perfect_team(out, gt)


def perfect_visible_team(pred, gt, present):
    out = np.asarray(pred).copy()
    live = present & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
    out[live] = gt[live]
    return out


def parse_version(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def audit_match(game, version=(2, 5)):
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    present = np.asarray(f['ball_present'], bool)
    pred = predict_version(f, graded, version)
    hide = ~present
    valid = graded & (gt != 'X')

    noise = {}
    for shift in (0, 3, 5, 10, -3, -5, -10):
        shifted = shift_labels(gt, shift)
        noise[str(shift)] = score_one(gt, shifted, hide)['spell_f1_mean']

    oracles = {
        'actual': score_natural(gt, pred, f),
        'perfect_team': score_natural(gt, perfect_team(pred, gt), f),
        'perfect_dead': score_natural(gt, perfect_dead(pred, gt), f),
        'perfect_both': score_natural(gt, perfect_both(pred, gt), f),
        'perfect_visible_team': score_natural(
            gt, perfect_visible_team(pred, gt, present), f),
    }
    errors = dict(
        all_frames=confusion(gt, pred, valid),
        visible_frames=confusion(gt, pred, valid & present),
        missing_frames=confusion(gt, pred, valid & ~present),
    )
    # Reachable target: 90% of perfect_both spell, capped by noise floor at ±5.
    perfect_spell = oracles['perfect_both']['spell_f1_mean']
    noise5 = min(noise['5'], noise['-5'])
    ceiling = min(perfect_spell, noise5)
    reachable = 0.9 * ceiling
    return dict(
        game=game,
        label_noise_spell_f1=noise,
        oracles={k: dict(spell_f1_mean=v['spell_f1_mean'],
                         hidden_team_acc=v['hidden_team_acc'],
                         hidden_n=v['hidden_n']) for k, v in oracles.items()},
        errors=errors,
        ceiling_spell_f1=ceiling,
        reachable_90pct_of_ceiling=reachable,
        absolute_0_90_feasible=bool(ceiling >= 0.90),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=MATCHES)
    parser.add_argument('--version', default='2.5',
                        help='labeller version major.minor (default 2.5)')
    parser.add_argument('--out', default=str(REPORTS / 'ceiling_audit.json'))
    args = parser.parse_args()
    version = parse_version(args.version)
    rows = []
    for game in args.matches:
        print('audit', game, 'v{}.{}'.format(*version), flush=True)
        row = audit_match(game, version=version)
        rows.append(row)
        print(' ', game, 'actual', round(row['oracles']['actual']['spell_f1_mean'], 4),
              'perfect_both', round(row['oracles']['perfect_both']['spell_f1_mean'], 4),
              'ceiling', round(row['ceiling_spell_f1'], 4),
              'reachable90', round(row['reachable_90pct_of_ceiling'], 4), flush=True)
    ceilings = [r['ceiling_spell_f1'] for r in rows]
    reachables = [r['reachable_90pct_of_ceiling'] for r in rows]
    actuals = [r['oracles']['actual']['spell_f1_mean'] for r in rows]
    summary = dict(
        version='{}.{}'.format(*version),
        n_matches=len(rows),
        mean_actual_spell_f1=float(np.mean(actuals)),
        mean_ceiling_spell_f1=float(np.mean(ceilings)),
        mean_reachable_90pct=float(np.mean(reachables)),
        absolute_0_90_feasible_all=all(r['absolute_0_90_feasible'] for r in rows),
        target_note=(
            'Headline target remains 0.90 if mean ceiling >= 0.90; '
            'otherwise aim for mean_reachable_90pct of the measured ceiling.'),
        matches=rows,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    note = Path(args.out).with_name(Path(args.out).stem + '_note.md')
    lines = [
        '# Ceiling audit (natural tracking, sm_v{}.{})'.format(*version),
        '',
        'Mean actual spellF1: {:.4f}'.format(summary['mean_actual_spell_f1']),
        'Mean ceiling spellF1: {:.4f}'.format(summary['mean_ceiling_spell_f1']),
        'Mean 90% of ceiling: {:.4f}'.format(summary['mean_reachable_90pct']),
        'Absolute 0.90 feasible on every match: {}'.format(
            summary['absolute_0_90_feasible_all']),
        '',
        '| Match | Actual | Perfect both | Ceiling | 90% of ceiling |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
            r['game'], r['oracles']['actual']['spell_f1_mean'],
            r['oracles']['perfect_both']['spell_f1_mean'],
            r['ceiling_spell_f1'], r['reachable_90pct_of_ceiling']))
    note.write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'version', 'mean_actual_spell_f1', 'mean_ceiling_spell_f1',
        'mean_reachable_90pct', 'absolute_0_90_feasible_all')}, indent=2))


if __name__ == '__main__':
    main()
