import unittest
import numpy as np
from starter.pass_travel_t27 import detect_same_team_travel


class T27ContactDurationRegressionTests(unittest.TestCase):

    def test_long_first_contact_does_not_count_as_intermediate(self):
        n = 20
        features = {
            "n_frames": n,
            "ball_present": np.zeros(n, dtype=bool),
            "ball_x_m": np.full(n, np.nan),
            "ball_y_m": np.full(n, np.nan),
            "nearest_team": np.full(n, -1),
            "nearest_dist": np.full(n, np.nan),
        }

        for frame, x in [(2, 10.0), (3, 11.0),
                         (4, 12.0), (8, 16.0)]:
            features["ball_present"][frame] = True
            features["ball_x_m"][frame] = x
            features["ball_y_m"][frame] = 20.0
            features["nearest_team"][frame] = 0
            features["nearest_dist"][frame] = 0.5

        candidates = detect_same_team_travel(features)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].start_frame, 4)
        self.assertEqual(candidates[0].end_frame, 8)
        self.assertEqual(candidates[0].missing_frames, 3)


if __name__ == "__main__":
    unittest.main()
