"""Audit sm_v2_3 H↔A switch timing vs GT to set switch-editor priors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import MATCHES, REPORTS, load_match, predict_version
from contested_band_features import contested_score, ha_switches


MATCH_RADIUS = 50


def match_switches(pred_idx, gt_idx, radius=MATCH_RADIUS):
    """Greedy nearest matching within radius. Returns offsets and leftovers."""
    pred_idx = np.asarray(pred_idx, np.int32)
    gt_idx = np.asarray(gt_idx, np.int32)
    used_gt = np.zeros(len(gt_idx), bool)
    matched_offsets = []
    matched_pred = []
    matched_gt = []
    false_pos = []
    for p in pred_idx:
        if len(gt_idx) == 0:
            false_pos.append(int(p))
            continue
        dists = np.abs(gt_idx - p)
        dists = np.where(used_gt, 10**9, dists)
        j = int(np.argmin(dists))
        if dists[j] <= radius:
            used_gt[j] = True
            matched_offsets.append(int(p - gt_idx[j]))
            matched_pred.append(int(p))
            matched_gt.append(int(gt_idx[j]))
        else:
            false_pos.append(int(p))
    missed = [int(g) for g, u in zip(gt_idx, used_gt) if not u]
    return dict(
        offsets=matched_offsets,
        matched_pred=matched_pred,
        matched_gt=matched_gt,
        false_pos=false_pos,
        missed=missed,
    )


def neighborhood_contested(score, centers, radius=12):
    if len(centers) == 0:
        return dict(n=0, mean_peak=float('nan'), pct_peak_ge_0_5=float('nan'))
    peaks = []
    for c in centers:
        a = max(0, c - radius)
        b = min(len(score), c + radius + 1)
        peaks.append(float(np.max(score[a:b])))
    peaks = np.asarray(peaks)
    return dict(
        n=len(peaks),
        mean_peak=float(np.mean(peaks)),
        pct_peak_ge_0_5=float(np.mean(peaks >= 0.5)),
    )


def parse_version(text):
    major, minor = text.split('.')
    return int(major), int(minor)


def audit_match(game, version=(2, 5)):
    data = load_match(game)
    f, gt, graded = data['features'], data['gt'], data['graded']
    pred = predict_version(f, graded, version)
    pred_sw = ha_switches(pred, graded)
    gt_sw = ha_switches(gt, graded)
    matched = match_switches(pred_sw, gt_sw)
    offsets = np.asarray(matched['offsets'], np.int32)
    score = contested_score(f)
    early = int(np.sum(offsets < 0)) if len(offsets) else 0
    late = int(np.sum(offsets > 0)) if len(offsets) else 0
    exact = int(np.sum(offsets == 0)) if len(offsets) else 0
    return dict(
        game=game,
        n_pred=int(len(pred_sw)),
        n_gt=int(len(gt_sw)),
        n_matched=len(matched['offsets']),
        n_false=len(matched['false_pos']),
        n_missed=len(matched['missed']),
        early=early,
        late=late,
        exact=exact,
        median_abs_offset=float(np.median(np.abs(offsets))) if len(offsets) else float('nan'),
        mean_abs_offset=float(np.mean(np.abs(offsets))) if len(offsets) else float('nan'),
        pct_abs_le_12=float(np.mean(np.abs(offsets) <= 12)) if len(offsets) else float('nan'),
        pct_abs_le_25=float(np.mean(np.abs(offsets) <= 25)) if len(offsets) else float('nan'),
        contested_matched=neighborhood_contested(score, matched['matched_pred']),
        contested_false=neighborhood_contested(score, matched['false_pos']),
        contested_missed=neighborhood_contested(score, matched['missed']),
        offsets_sample=matched['offsets'][:40],
    )


def recommend_priors(rows):
    abs_medians = [r['median_abs_offset'] for r in rows if r['median_abs_offset'] == r['median_abs_offset']]
    med = float(np.median(abs_medians)) if abs_medians else 12.0
    # Retarget radius: cover ~most matched offsets, capped.
    r = int(np.clip(np.ceil(med * 2), 12, 40))
    false_rates = [r['n_false'] / max(1, r['n_pred']) for r in rows]
    miss_rates = [r['n_missed'] / max(1, r['n_gt']) for r in rows]
    contested_miss = [r['contested_missed'].get('pct_peak_ge_0_5', 0) for r in rows
                      if r['contested_missed'].get('n', 0) > 0]
    return dict(
        retarget_radius=r,
        margin=0.55,  # predict_proba confidence threshold
        min_island=18,
        match_radius=MATCH_RADIUS,
        mean_false_rate=float(np.mean(false_rates)),
        mean_miss_rate=float(np.mean(miss_rates)),
        mean_median_abs_offset=med,
        mean_missed_contested_ge_0_5=float(np.mean(contested_miss)) if contested_miss else 0.,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches', nargs='+', default=MATCHES)
    parser.add_argument('--version', default='2.5',
                        help='labeller version major.minor (default 2.5)')
    parser.add_argument('--out', default=str(REPORTS / 'switch_timing_audit.json'))
    args = parser.parse_args()
    version = parse_version(args.version)
    rows = []
    for game in args.matches:
        print('audit', game, 'v{}.{}'.format(*version), flush=True)
        row = audit_match(game, version=version)
        rows.append(row)
        print(' ', game, 'pred', row['n_pred'], 'gt', row['n_gt'],
              'match', row['n_matched'], 'false', row['n_false'],
              'miss', row['n_missed'], 'med|off|', round(row['median_abs_offset'], 2),
              flush=True)
    priors = recommend_priors(rows)
    summary = dict(version='{}.{}'.format(*version), matches=rows, priors=priors)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    note_path = Path(args.out).with_name(Path(args.out).stem + '_note.md')
    dominant = 'false_positive' if priors['mean_false_rate'] >= priors['mean_miss_rate'] else 'missed_gt'
    if priors['mean_median_abs_offset'] >= 8:
        dominant = 'timing_offset'
    lines = [
        '# Switch-timing audit (sm_v{}.{})'.format(*version),
        '',
        'Mean median |offset| (matched): {:.2f}'.format(priors['mean_median_abs_offset']),
        'Mean false rate: {:.3f}  mean miss rate: {:.3f}'.format(
            priors['mean_false_rate'], priors['mean_miss_rate']),
        'Missed peaks contested≥0.5: {:.3f}'.format(priors['mean_missed_contested_ge_0_5']),
        'Dominant mode: **{}**'.format(dominant),
        'Recommended priors: R={}, margin={}, min_island={}'.format(
            priors['retarget_radius'], priors['margin'], priors['min_island']),
        '',
        '| Match | Pred | GT | Match | False | Miss | Med |off| | ≤12f |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append('| {} | {} | {} | {} | {} | {} | {:.1f} | {:.3f} |'.format(
            r['game'], r['n_pred'], r['n_gt'], r['n_matched'], r['n_false'],
            r['n_missed'], r['median_abs_offset'], r['pct_abs_le_12']))
    note_path.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(version=summary['version'], dominant=dominant, **priors), indent=2))


if __name__ == '__main__':
    main()
