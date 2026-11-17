"""BAR-ATH video smoke: frozen FootAndBall vs ATD agreement (aggregates only).

Boundaries: docs/algorithms/video-smoke-boundaries.md (frozen before
this run). Detector/thresholds/segment/kill are pre-registered; output
is counts and quantiles only. Per-frame positions never leave memory.
Writes evaluation/reports/barath_smoke.json (aggregates only).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'data_external/footandball/repo')
sys.path.insert(0, 'starter')
from crossfit_common import REPORTS  # noqa: E402
from games import asset_paths  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
import data.augmentation as augmentations  # noqa: E402
from data.augmentation import BALL_LABEL  # noqa: E402
import network.footandball as footandball  # noqa: E402

SEG_START, SEG_LEN = 11185, 2000
WEIGHTS = 'data_external/footandball/model_20201019_1416_final.pth'
LENGTH, WIDTH = 105.3, 68.0


def atd_ball(segment):
    """Ball xy (m) for segment frames from the ATD raw stream (code-only)."""
    raw = asset_paths()['atd_raw']
    out = {}
    end = SEG_START + SEG_LEN
    with open(raw) as handle:
        for line in handle:
            parts = line.split(':')
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if frame < SEG_START:
                continue
            if frame >= end:
                break
            ball = parts[2].strip()
            if ball and ball[0] != 'N':
                x, y, _ = (float(v) for v in ball.split(','))
                out[frame] = (x * LENGTH, y * WIDTH)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'barath_smoke.json'))
    args = parser.parse_args()
    matrices, reproj, valid = load_homography()
    atd = atd_ball(range(SEG_START, SEG_START + SEG_LEN))
    model = footandball.model_factory('fb1', 'detect', ball_threshold=0.7,
                                      player_threshold=0.7)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.eval()
    cap = cv2.VideoCapture(asset_paths()['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, SEG_START)
    dists, n_det, n_frames = [], 0, 0
    for i in range(SEG_LEN):
        ok, frame = cap.read()
        if not ok:
            break
        n_frames += 1
        video_frame = SEG_START + i
        tensor = augmentations.numpy2tensor(frame).unsqueeze(0)
        with torch.no_grad():
            det = model(tensor)[0]
        balls = [(b, s) for b, l, s in zip(det['boxes'], det['labels'], det['scores'])
                 if int(l) == BALL_LABEL]
        if not balls:
            continue
        n_det += 1
        if video_frame not in atd or not valid[video_frame]:
            continue
        box, _ = max(balls, key=lambda bs: float(bs[1]))
        cx, cy = (float(box[0]) + float(box[2])) / 2., (float(box[1]) + float(box[3])) / 2.
        nx, ny = unproject(matrices[video_frame], cx, cy)
        ax, ay = atd[video_frame]
        dists.append(float(np.hypot(nx * LENGTH - ax, ny * WIDTH - ay)))
    cap.release()
    dists = np.array(dists)
    if n_frames == 0 or len(dists) == 0:
        verdict = 'INCONCLUSIVE (video unreadable/absent or no compared frames)'
    else:
        verdict = 'STOP' if np.median(dists) > 2. else 'PASS'
    report = dict(segment=[SEG_START, SEG_START + SEG_LEN - 1], n_frames=n_frames,
                  atd_present_rate=float(len(atd) / SEG_LEN),
                  detector_present_rate=float(n_det / max(n_frames, 1)),
                  n_compared=int(len(dists)),
                  median_disagreement_m=float(np.median(dists)) if len(dists) else None,
                  p90_disagreement_m=float(np.percentile(dists, 90)) if len(dists) else None,
                  pct_gt_1p5m=float(np.mean(dists > 1.5)) if len(dists) else None,
                  kill_criterion='median > 2m',
                  verdict=verdict)
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
