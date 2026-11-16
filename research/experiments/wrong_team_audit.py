"""Bucket sm_v2_4 live H↔A errors to set anti-flicker priors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import MATCHES, REPORTS, load_match
from eval_external import kwargs_for_version, run_labeller
from error_corrector_model import ensemble_team_logits


def island_lengths(mask):
    """Length of the contiguous True island covering each True index."""
    n = len(mask)
    lengths = np.zeros(n, np.int32)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        lengths[i:j] = j - i
        i = j
    return lengths


def dist_to_switch(gt, live):
    """Frames to nearest GT H↔A switch within the same live run context."""
    n = len(gt)
    switch = np.zeros(n, bool)
    switch[1:] = live[1:] & live[:-1] & (gt[1:] != gt[:-1]) & np.isin(gt[1:], ['H', 'A']) & np.isin(gt[:-1], ['H', 'A'])
    idx = np.flatnonzero(switch)
    dist = np.full(n, 9999, np.int32)
    if len(idx) == 0:
        return dist
    for i in range(n):
        dist[i] = int(np.min(np.abs(idx - i)))
    return dist


def parse_version(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def audit_match(game, version=(2, 5)):
    data = load_match(game)
    f, gt, graded, nodes = data['features'], data['gt'], data['graded'], data['nodes']
    pred = run_labeller(f, graded, kwargs_for_version(*version))
    # Ensemble logits are defined on the boundary stack (v2.3).
    base23 = run_labeller(f, graded, kwargs_for_version(2, 3))
    logits = ensemble_team_logits(f, graded, nodes, base23)
    present = np.asarray(f['ball_present'], bool)
    live = graded & np.isin(gt, ['H', 'A']) & np.isin(pred, ['H', 'A'])
    wrong = live & (pred != gt)
    nearest = f['nearest_team']
    majority = f['majority_team']
    nearest_lab = np.where(nearest == 0, 'H', np.where(nearest == 1, 'A', 'X'))
    majority_lab = np.where(majority == 0, 'H', np.where(majority == 1, 'A', 'X'))
    lengths = island_lengths(wrong)
    dist = dist_to_switch(gt, graded & np.isin(gt, ['H', 'A']))
    abs_logit = np.abs(logits)

    def subset(mask):
        if not np.any(mask):
            return dict(n=0)
        L = lengths[mask]
        return dict(
            n=int(mask.sum()),
            mean_island=float(np.mean(L)),
            median_island=float(np.median(L)),
            pct_island_le_12=float(np.mean(L <= 12)),
            pct_island_le_25=float(np.mean(L <= 25)),
            pct_near_switch_le_12=float(np.mean(dist[mask] <= 12)),
            pct_near_switch_le_25=float(np.mean(dist[mask] <= 25)),
            pct_disagree_nearest=float(np.mean(nearest_lab[mask] != pred[mask])),
            pct_disagree_majority=float(np.mean(majority_lab[mask] != pred[mask])),
            mean_abs_logit=float(np.mean(abs_logit[mask])),
            pct_abs_logit_lt_1=float(np.mean(abs_logit[mask] < 1.0)),
            pct_abs_logit_lt_1_5=float(np.mean(abs_logit[mask] < 1.5)),
            pct_visible=float(np.mean(present[mask])),
        )

    visible_wrong = wrong & present
    missing_wrong = wrong & ~present
    return dict(
        game=game,
        live_n=int(live.sum()),
        wrong_n=int(wrong.sum()),
        wrong_rate=float(wrong.sum() / max(1, live.sum())),
        all_wrong=subset(wrong),
        visible_wrong=subset(visible_wrong),
        missing_wrong=subset(missing_wrong),
    )


def recommend_priors(rows):
    # Aggregate visible wrong stats
    islands = []
    near = []
    for r in rows:
        vw = r['visible_wrong']
        if vw.get('n', 0) == 0:
            continue
        islands.append(vw['pct_island_le_12'])
        near.append(vw['pct_near_switch_le_12'])
    mean_short = float(np.mean(islands)) if islands else 0.5
    mean_near = float(np.mean(near)) if near else 0.5
    # If many errors are short islands, prefer stronger min_island / window.
    min_island = 12 if mean_short >= 0.4 else 8
    window = 5 if mean_short >= 0.4 or mean_near >= 0.4 else 3
    margin = 1.5
    return dict(
        margin=margin,
        window=window,
        min_island=min_island,
        rationale=dict(
            mean_pct_island_le_12=mean_short,
            mean_pct_near_switch_le_12=mean_near,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=MATCHES)
    parser.add_argument('--version', default='2.5',
                        help='labeller version major.minor (default 2.5)')
    parser.add_argument('--out', default=str(REPORTS / 'wrong_team_audit.json'))
    args = parser.parse_args()
    version = parse_version(args.version)
    rows = []
    for game in args.matches:
        print('audit', game, 'v{}.{}'.format(*version), flush=True)
        row = audit_match(game, version=version)
        rows.append(row)
        print(' ', game, 'wrong', row['wrong_n'], '/', row['live_n'],
              'rate', round(row['wrong_rate'], 4),
              'short_isl', round(row['visible_wrong'].get('pct_island_le_12', 0), 3), flush=True)
    priors = recommend_priors(rows)
    summary = dict(version='{}.{}'.format(*version), matches=rows, priors=priors,
                   mean_wrong_rate=float(np.mean([r['wrong_rate'] for r in rows])))
    Path(args.out).write_text(json.dumps(summary, indent=2))
    note = Path(args.out).with_name(Path(args.out).stem + '_note.md')
    lines = [
        '# Wrong-team audit (sm_v{}.{})'.format(*version),
        '',
        'Mean wrong-team rate: {:.4f}'.format(summary['mean_wrong_rate']),
        'Recommended priors: margin={}, window={}, min_island={}'.format(
            priors['margin'], priors['window'], priors['min_island']),
        'Rationale: {}'.format(priors['rationale']),
        '',
        '| Match | Wrong rate | Vis short≤12f | Near switch≤12f | Mean |logit| |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        vw = r['visible_wrong']
        lines.append('| {} | {:.4f} | {:.3f} | {:.3f} | {:.3f} |'.format(
            r['game'], r['wrong_rate'],
            vw.get('pct_island_le_12', float('nan')),
            vw.get('pct_near_switch_le_12', float('nan')),
            vw.get('mean_abs_logit', float('nan'))))
    note.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(version=summary['version'], **priors), indent=2))


if __name__ == '__main__':
    main()
