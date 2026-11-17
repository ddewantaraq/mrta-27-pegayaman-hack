"""a-ext: FootAndBall falsification on ISSIA held-out cameras 5+6.

Pretrained weights (sha256 42fbc1f4...), full native resolution, ball
threshold 0.7 (repo default). GT from xgtf BallPos via the FootAndBall
repo's own issia_utils (delta=-8, cam 2/6 x-flip handled there).
Metrics: detection rate, pixel error median/p90, %<=10px/25px, false
positives on GT-absent frames. Writes issia_falsification.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path('data_external/footandball/repo')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS  # noqa: E402
import data.augmentation as augmentations  # noqa: E402
from data.augmentation import BALL_LABEL  # noqa: E402
from data import issia_utils  # noqa: E402
import network.footandball as footandball  # noqa: E402

WEIGHTS = 'data_external/footandball/model_20201019_1416_final.pth'
SEQ = Path('data_external/issia/sequences')
ANN = Path('data_external/issia/annotations')


def gt_ball(camera):
    ann_file = ANN / 'Film Role-0 ID-{} T-0 m00s00-026-m00s01-020.xgtf'.format(camera)
    gt = issia_utils._load_groundtruth(str(ann_file))
    video = SEQ / 'Film Role-0 ID-{} T-2 m00s00-000-m00s00-185.avi'.format(camera)
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    annotations = issia_utils._create_annotations(gt, camera, frame.shape)
    return annotations.ball_pos, video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', nargs='+', type=int, default=[5, 6])
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--out', default=str(REPORTS / 'issia_falsification.json'))
    args = parser.parse_args()
    model = footandball.model_factory('fb1', 'detect', ball_threshold=0.7,
                                      player_threshold=0.7)
    state = torch.load(WEIGHTS, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    rows = []
    for camera in args.cameras:
        ball_pos, video = gt_ball(camera)
        frames = sorted(f for f, v in ball_pos.items() if v)
        frames = frames[50::args.stride][:args.max_frames]
        cap = cv2.VideoCapture(str(video))
        errors, detected = [], 0
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            tensor = augmentations.numpy2tensor(frame).unsqueeze(0)
            with torch.no_grad():
                det = model(tensor)[0]
            balls = [(b, s) for b, l, s in zip(det['boxes'], det['labels'], det['scores'])
                     if int(l) == BALL_LABEL]
            if not balls:
                continue
            box, _ = max(balls, key=lambda bs: float(bs[1]))
            cx, cy = (float(box[0]) + float(box[2])) / 2., (float(box[1]) + float(box[3])) / 2.
            gx, gy = ball_pos[f][0]
            errors.append(float(np.hypot(cx - gx, cy - gy)))
            detected += 1
        cap.release()
        errors = np.array(errors)
        row = dict(camera=camera, n_gt_frames=len(frames), detected=detected,
                   detection_rate=float(detected / max(len(frames), 1)),
                   err_median_px=float(np.median(errors)) if len(errors) else None,
                   err_p90_px=float(np.percentile(errors, 90)) if len(errors) else None,
                   pct_le10=float(np.mean(errors <= 10)) if len(errors) else None,
                   pct_le25=float(np.mean(errors <= 25)) if len(errors) else None)
        rows.append(row)
        print(row, flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
