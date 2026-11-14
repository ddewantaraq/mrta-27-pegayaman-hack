"""Discover camera alignment on ATD only; no challenge labels are loaded."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from camera_alignment import find_anchors, fuse_camera_ball
from games import require, source_frames, GAMES
from features import _parse_xyz


def ball_track(source):
    n = source_frames(source)
    xy = np.full((n, 2), np.nan, np.float32)
    present = np.zeros(n, bool)
    raw = require(['atd_raw'], source=source)[0]
    length, width = GAMES['BAR-ATH']['pitch']
    with open(raw) as handle:
        for line in handle:
            fields = line.split(':')
            if len(fields) != 3:
                continue
            i = int(fields[0])-1
            point = _parse_xyz(fields[2].strip())
            if point is not None and 0 <= i < n:
                xy[i] = point[0]*length, point[1]*width
                present[i] = True
    return xy, present


def main():
    target, tp = ball_track('tactical')
    source, sp = ball_track('broadcast')
    pitch = np.asarray(GAMES['BAR-ATH']['pitch'])
    candidates = []
    for flip_x, flip_y in ((False, False), (True, False), (False, True), (True, True)):
        transformed = source.copy()
        if flip_x:
            transformed[:, 0] = pitch[0]-transformed[:, 0]
        if flip_y:
            transformed[:, 1] = pitch[1]-transformed[:, 1]
        anchors = find_anchors(target, tp, transformed, sp)
        xy, present, fill, mapping = fuse_camera_ball(target, tp, transformed, sp, anchors)
        summary = dict(flip_x=flip_x, flip_y=flip_y, anchors=len(anchors),
                       recovered=int(fill.sum()), mapped=int(np.sum(mapping >= 0)),
                       median_anchor_error=float(np.median(anchors[:, 2])) if len(anchors) else None)
        print(json.dumps(summary), flush=True)
        candidates.append((int(fill.sum()), summary, xy, present, fill, anchors, mapping))
    best = max(candidates, key=lambda item: item[0])
    out = Path('evaluation/reports/camera_alignment')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'fusion_candidate.npz', xy=best[2], present=best[3],
                        fill=best[4], anchors=best[5], mapping=best[6])
    (out/'summary.json').write_text(json.dumps(best[1], indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
