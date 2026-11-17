"""ISSIA eval for SoccerNet-fine-tuned YOLO ball detector (selection only).

Same frames/protocol as issia_falsification.py (cams 5+6, stride 2).
Ultralytics default conf 0.25, imgsz 1280 per boundaries v2.
Writes evaluation/reports/issia_yolo.json.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crossfit_common import REPORTS  # noqa: E402
from issia_falsification import SEQ, gt_ball  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WEIGHTS = 'data_external/yolo/football-ball-detection.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=1000)
    parser.add_argument('--out', default=str(REPORTS / 'issia_yolo.json'))
    args = parser.parse_args()
    model = YOLO(WEIGHTS)
    print('classes', model.names, flush=True)
    ball_cls = [k for k, v in model.names.items() if 'ball' in str(v).lower()]
    ball_cls = ball_cls[0] if ball_cls else 0
    rows = []
    for camera in (5, 6):
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
            result = model(frame, imgsz=1280, conf=0.25, verbose=False)[0]
            boxes = result.boxes
            cand = [(boxes.xyxy[i], float(boxes.conf[i])) for i in range(len(boxes))
                    if int(boxes.cls[i]) == ball_cls]
            if not cand:
                continue
            box, _ = max(cand, key=lambda bs: bs[1])
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
