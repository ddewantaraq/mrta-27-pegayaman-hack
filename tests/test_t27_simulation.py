import unittest
import numpy as np

from starter.t27_integration import analyze_t27


class T27SimulationTests(unittest.TestCase):

    def make_features(self, second_team=0):
        n = 12

        ball_present = np.zeros(n, dtype=bool)
        ball_x = np.full(n, np.nan)
        ball_y = np.full(n, np.nan)
        nearest_team = np.full(n, -1, dtype=int)
        nearest_dist = np.full(n, np.nan)

        # First observed contact: home team
        ball_present[1] = True
        ball_x[1] = 10.0
        ball_y[1] = 20.0
        nearest_team[1] = 0
        nearest_dist[1] = 0.5

        # Second observed contact
        ball_present[6] = True
        ball_x[6] = 16.0
        ball_y[6] = 20.0
        nearest_team[6] = second_team
        nearest_dist[6] = 0.5

        return {
            "n_frames": n,
            "ball_present": ball_present,
            "ball_x_m": ball_x,
            "ball_y_m": ball_y,
            "nearest_team": nearest_team,
            "nearest_dist": nearest_dist,
        }

    def test_same_team_candidate(self):
        features = self.make_features(second_team=0)
        labels = np.array(["H"] * 12)

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].team, 0)
        self.assertEqual(candidates[0].start_frame, 1)
        self.assertEqual(candidates[0].end_frame, 6)

    def test_opposing_team_rejected(self):
        features = self.make_features(second_team=1)
        labels = np.array(["H"] * 12)

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        self.assertEqual(len(candidates), 0)

    def test_dead_ball_boundary(self):
        features = self.make_features(second_team=0)
        labels = np.array(["H"] * 12)
        labels[4] = "D"

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        self.assertEqual(len(candidates), 0)

    def test_ungraded_boundary(self):
        features = self.make_features(second_team=0)
        labels = np.array(["H"] * 12)
        graded = np.ones(12, dtype=bool)
        graded[4] = False

        candidates = analyze_t27(
            features,
            labels,
            graded=graded
        )

        self.assertEqual(len(candidates), 0)

    def test_labels_not_modified(self):
        features = self.make_features(second_team=0)
        labels = np.array(["H"] * 12)
        original = labels.copy()

        analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        np.testing.assert_array_equal(labels, original)


if __name__ == "__main__":
    unittest.main()
