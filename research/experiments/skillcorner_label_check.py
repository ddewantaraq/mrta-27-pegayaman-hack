"""Does SkillCorner's possession.group mean the same thing as our H/A?

Runs the anchor labeller on SkillCorner training-split matches (never Track D)
and reports frame agreement on labelled H/A frames plus the signed offset
between each SkillCorner team switch and the nearest predicted switch of the
same direction. A consistent offset becomes a fixed label shift; nothing is
fitted on Track B/C/D or BAR-ATH.
Also writes the committed Track D split (lowest 4 sha256 of the match id).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS, load_match, score_natural
from eval_external import kwargs_for_version, run_labeller
from fetch_skillcorner import MATCH_IDS
from leakage_guard import SKILLCORNER_SPLIT

N_TRACK_D = 4
WINDOW = 75


def split():
    games = ['SC_' + m for m in MATCH_IDS]
    ranked = sorted(games, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    return dict(track_d=sorted(ranked[:N_TRACK_D]), train=sorted(ranked[N_TRACK_D:]),
                rule='lowest {} sha256(match id) are Track D'.format(N_TRACK_D))


def write_split():
    data = split()
    Path(SKILLCORNER_SPLIT).write_text(json.dumps(data, indent=1) + '\n')
    return data


def team_switches(labels, graded):
    """(frame, new team) where the team changes, bridging X/D gaps inside a period."""
    out, last = [], None
    for i in np.flatnonzero(graded & np.isin(labels, ['H', 'A'])):
        if last is not None and labels[i] != labels[last] and graded[last:i + 1].all():
            out.append((int(i), labels[i]))
        last = i
    return out


def match_check(game, version):
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    pred = run_labeller(f, graded, kwargs_for_version(*version))
    lab = np.isin(gt, ['H', 'A'])
    agree = float(np.mean(pred[lab] == gt[lab]))
    pred_live = float(np.mean(np.isin(pred[lab], ['H', 'A'])))
    team_agree = float(np.mean(pred[lab & np.isin(pred, ['H', 'A'])] ==
                               gt[lab & np.isin(pred, ['H', 'A'])]))
    ps = team_switches(pred, graded)
    pf = np.array([p for p, _ in ps])
    pt = np.array([t for _, t in ps])
    offsets = []
    for frame, team in team_switches(gt, graded):
        if not len(pf):
            break
        cand = np.flatnonzero((pt == team) & (np.abs(pf - frame) <= WINDOW))
        if len(cand):
            k = cand[np.argmin(np.abs(pf[cand] - frame))]
            offsets.append(int(pf[k] - frame))
    offsets = np.array(offsets)
    row = dict(game=game, frame_agree=agree, team_agree_when_live=team_agree,
               pred_live_share=pred_live, n_gt_switches=len(team_switches(gt, graded)),
               n_matched=len(offsets),
               median_offset=float(np.median(offsets)) if len(offsets) else float('nan'),
               mean_offset=float(np.mean(offsets)) if len(offsets) else float('nan'),
               **{'spell_' + k: v for k, v in score_natural(gt, pred, f).items()
                  if k in ('spell_f1_mean', 'hidden_team_acc')})
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in row.items()}), flush=True)
    return row, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='2.6')
    parser.add_argument('--out', default=str(REPORTS / 'skillcorner_label_check.json'))
    args = parser.parse_args()
    version = tuple(int(p) for p in args.version.split('.'))
    sp = write_split()
    rows, all_off = [], []
    for game in sp['train']:
        row, off = match_check(game, version)
        rows.append(row)
        all_off.extend(off.tolist())
    all_off = np.array(all_off)
    summary = dict(
        frame_agree=float(np.mean([r['frame_agree'] for r in rows])),
        team_agree_when_live=float(np.mean([r['team_agree_when_live'] for r in rows])),
        pred_live_share=float(np.mean([r['pred_live_share'] for r in rows])),
        switch_match_rate=float(sum(r['n_matched'] for r in rows)) /
        max(sum(r['n_gt_switches'] for r in rows), 1),
        median_offset=float(np.median(all_off)),
        offset_iqr=[float(np.percentile(all_off, 25)), float(np.percentile(all_off, 75))],
        spell_f1=float(np.mean([r['spell_spell_f1_mean'] for r in rows])),
    )
    # Fixed shift only if the lag is consistent (IQR does not straddle zero).
    lo, hi = summary['offset_iqr']
    summary['label_shift_frames'] = int(round(summary['median_offset'])) if lo * hi > 0 else 0
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps(dict(version=args.version, split=sp,
                                              summary=summary, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
