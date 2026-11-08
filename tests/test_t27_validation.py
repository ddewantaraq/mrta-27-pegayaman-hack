import unittest
import numpy as np

from starter.pass_travel_t27 import detect_same_team_travel


class T27ValidationTests(unittest.TestCase):

    def make_features(self):
        n = 10
        return {
            "n_frames": n,
            "ball_present": np.zeros(n, dtype=bool),
            "ball_x_m": np.full(n, np.nan),
            "ball_y_m": np.full(n, np.nan),
            "nearest_team": np.full(n, -1),
            "nearest_dist": np.full(n, np.nan),
        }

    def test_zero_fps_rejected(self):
        features = self.make_features()

        with self.assertRaises(ValueError):
            detect_same_team_travel(features, fps=0)

    def test_negative_contact_radius_rejected(self):
        features = self.make_features()

        with self.assertRaises(ValueError):
            detect_same_team_travel(
                features,
                contact_radius_m=-1.0
            )

    def test_invalid_gap_range_rejected(self):
        features = self.make_features()

        with self.assertRaises(ValueError):
            detect_same_team_travel(
                features,
                min_gap_frames=10,
                max_gap_frames=5
            )

    def test_wrong_graded_length_rejected(self):
        features = self.make_features()
        graded = np.ones(5, dtype=bool)

        with self.assertRaises(ValueError):
            detect_same_team_travel(
                features,
                graded=graded
            )


if __name__ == "__main__":
    unittest.main()
