"""BAR-ATH smoke v2: FootAndBall (ISSIA-selected) + 5m ROI gate.

Boundaries: docs/algorithms/video-smoke-v2-boundaries.md. Identical
segment/kill to v1; gate pre-registered at 5m around ATD ball.
Aggregates only. Writes evaluation/reports/barath_smoke_v2.json.
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
sys.path.insert(0, '.')
from crossfit_common import REPORTS  # noqa: E402
from games import asset_paths  # noqa: E402
from load_homography import load as load_homography, unproject  # noqa: E402
from barath_smoke import LENGTH, WIDTH, SEG_LEN, SEG_START, WEIGHTS, atd_ball  # noqa: E402
import data.augmentation as augmentations  # noqa: E402
from data.augmentation import BALL_LABEL  # noqa: E402
import network.footandball as footandball  # noqa: E402

GATE_M = 5.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(REPORTS / 'barath_smoke_v2.json'))
    args = parser.parse_args()
    matrices, reproj, valid = load_homography()
    atd = atd_ball(range(SEG_START, SEG_START + SEG_LEN))
    model = footandball.model_factory('fb1', 'detect', ball_threshold=0.7,
                                      player_threshold=0.7)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.eval()
    cap = cv2.VideoCapture(asset_paths()['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, SEG_START)
    dists, n_det, n_gated, n_frames = [], 0, 0, 0
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
        d = float(np.hypot(nx * LENGTH - ax, ny * WIDTH - ay))
        if d <= GATE_M:
            n_gated += 1
            dists.append(d)
    cap.release()
    dists = np.array(dists)
    if len(dists) == 0:
        verdict = 'STOP (no gated detections to compare)'
    else:
        verdict = 'STOP' if np.median(dists) > 2. else 'PASS'
    report = dict(segment=[SEG_START, SEG_START + SEG_LEN - 1], n_frames=n_frames,
                  atd_present_rate=float(len(atd) / SEG_LEN),
                  detector_present_rate_ungated=float(n_det / max(n_frames, 1)),
                  gated_compared=int(len(dists)),
                  median_disagreement_m=float(np.median(dists)) if len(dists) else None,
                  kill_criterion='median > 2m (gated)',
                  verdict=verdict)
    print(json.dumps(report, indent=2))
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
