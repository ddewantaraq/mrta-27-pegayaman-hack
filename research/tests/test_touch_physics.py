import unittest

import numpy as np

from touch_physics_experiment import intercept_features


class InterceptFeatureTests(unittest.TestCase):
    def test_home_closing_player_wins_intercept_race(self):
        n = 30
        features = dict(ball_x_m=np.full(n, 50.), ball_y_m=np.full(n, 34.),
                        ball_present=np.ones(n, bool), pitch=(105.3, 68.))
        nodes = np.zeros((10, 2, 11, 8), np.float32)
        nodes[:, :, :, 7] = 1.  # sign +
        # home player0 at (40,34) running +x at 5 m/s; away at (90,34) still
        nodes[:, 0, 0] = [40 / 105.3, 34 / 68., .5, 0., .5, 1., 1., 1.]
        nodes[:, 1, 0] = [90 / 105.3, 34 / 68., 0., 0., 0., 1., 1., 1.]
        out = intercept_features(features, nodes, np.array([10]))
        th_min, ta_min = out[0, 0], out[0, 1]
        closing_home = out[0, 4]
        self.assertLess(th_min, ta_min)
        self.assertGreater(closing_home, 0.)


if __name__ == '__main__':
    unittest.main()
