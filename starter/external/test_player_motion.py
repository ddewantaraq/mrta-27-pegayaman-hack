import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from player_motion import summarize_motion


class PlayerMotionTests(unittest.TestCase):
    def moving_players(self):
        xy = np.zeros((80, 2, 2), np.float32)
        xy[:, 0, 0] = np.arange(80)/25.
        xy[:, 1, 0] = 20+np.arange(80)*2/25.
        return xy

    def test_units_and_player_order(self):
        xy = self.moving_players()
        a = summarize_motion(xy)
        b = summarize_motion(xy[:, ::-1])
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertAlmostEqual(a[30, 0]*10, 1.5, places=4)
        self.assertAlmostEqual(a[30, 2]*10, 1.5, places=4)

    def test_missing_tracks_are_not_stationary_players(self):
        xy = np.full((80, 2, 2), np.nan, np.float32)
        out = summarize_motion(xy)
        self.assertTrue(np.isfinite(out).all())
        self.assertFalse(out.any())

    def test_motion_does_not_cross_period_boundary(self):
        xy = self.moving_players()
        period = np.r_[np.ones(40), np.full(40, 2)]
        out = summarize_motion(xy, period)
        self.assertTrue((out[35:45, 9] == 0).all())
        self.assertGreater(out[20, 9], 0)


if __name__ == '__main__':
    unittest.main()
