import unittest
import numpy as np

from starter.pass_travel_t27 import detect_same_team_travel


class T27ContactEdgeTests(unittest.TestCase):

    def make_features(self):
        n = 20
        bp = np.zeros(n, dtype=bool)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        team = np.full(n, -1)
        dist = np.full(n, np.nan)

        return {
            "n_frames": n,
            "ball_present": bp,
            "ball_x_m": x,
            "ball_y_m": y,
            "nearest_team": team,
            "nearest_dist": dist,
        }

    def add_contact(self, features, frame, x):
        features["ball_present"][frame] = True
        features["ball_x_m"][frame] = x
        features["ball_y_m"][frame] = 20.0
        features["nearest_team"][frame] = 0
        features["nearest_dist"][frame] = 0.5

    def test_continuous_contact_not_counted_as_pass(self):
        features = self.make_features()

        for frame in range(2, 9):
            self.add_contact(features, frame, 10.0 + frame - 2)

        candidates = detect_same_team_travel(features)

        self.assertEqual(len(candidates), 0)

    def test_two_separate_contacts(self):
        features = self.make_features()

        self.add_contact(features, 2, 10.0)
        self.add_contact(features, 8, 16.0)

        candidates = detect_same_team_travel(features)

        self.assertEqual(len(candidates), 1)

    def test_intermediate_contact_rejected(self):
        features = self.make_features()

        self.add_contact(features, 2, 10.0)
        self.add_contact(features, 5, 13.0)
        self.add_contact(features, 8, 16.0)

        candidates = detect_same_team_travel(features)

        self.assertFalse(
            any(
                c.start_frame == 2 and c.end_frame == 8
                for c in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
