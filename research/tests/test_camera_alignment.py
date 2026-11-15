import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_alignment import find_anchors, mapping_from_anchors, fuse_camera_ball


class CameraAlignmentTests(unittest.TestCase):
    def test_inconsistent_offsets_do_not_create_mapping(self):
        anchors = np.array([[10, 50, 0], [100, 300, 0]], float)
        self.assertTrue((mapping_from_anchors(anchors, 150) == -1).all())

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'optional alignment dependency')
    def test_shifted_track_recovery_without_changing_observed_points(self):
        time = np.arange(1500)/25.
        original = np.column_stack([40+20*np.sin(time/4), 30+10*np.cos(time/3)])
        source = np.vstack([np.full((50, 2), np.nan), original])
        sp = np.isfinite(source).all(axis=1)
        target = original.copy()
        present = np.ones(1500, bool)
        present[400:410] = False
        target[~present] = np.nan
        anchors = find_anchors(target, present, source, sp)
        self.assertGreater(len(anchors), 4)
        self.assertAlmostEqual(np.median(anchors[:, 1]-anchors[:, 0]), 50.)
        output, _, fill, _ = fuse_camera_ball(target, present, source, sp, anchors)
        self.assertTrue(fill[400:410].all())
        np.testing.assert_array_equal(output[present], original[present])
        np.testing.assert_allclose(output[fill], original[fill])


if __name__ == '__main__':
    unittest.main()
