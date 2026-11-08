import unittest
import numpy as np

from starter.t27_integration import analyze_t27


class T27AdvancedTests(unittest.TestCase):

    def make_features(self):
        n = 15
        bp = np.zeros(n, dtype=bool)
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        team = np.full(n, -1)
        dist = np.full(n, np.nan)

        for frame, pos in [(2, 10.0), (8, 16.0)]:
            bp[frame] = True
            x[frame] = pos
            y[frame] = 20.0
            team[frame] = 0
            dist[frame] = 0.5

        return {
            "n_frames": n,
            "ball_present": bp,
            "ball_x_m": x,
            "ball_y_m": y,
            "nearest_team": team,
            "nearest_dist": dist,
        }

    def test_valid_same_team_travel(self):
        features = self.make_features()
        labels = np.array(["H"] * 15)
        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))
        self.assertEqual(len(candidates), 1)

    def test_halftime_ungraded_boundary(self):
        features = self.make_features()
        labels = np.array(["H"] * 15)
        graded = np.ones(15, dtype=bool)
        graded[5:7] = False

        candidates = analyze_t27(
            features, labels, graded=graded
        )
        self.assertEqual(len(candidates), 0)

    def test_dead_ball_in_middle(self):
        features = self.make_features()
        labels = np.array(["H"] * 15)
        labels[5] = "D"

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))
        self.assertEqual(len(candidates), 0)

    def test_team_switch_in_middle(self):
        features = self.make_features()
        labels = np.array(["H"] * 15)
        labels[5] = "A"

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))
        self.assertEqual(len(candidates), 0)

    def test_no_mutation(self):
        features = self.make_features()
        labels = np.array(["H"] * 15)
        before = labels.copy()

        analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        np.testing.assert_array_equal(labels, before)


if __name__ == "__main__":
    unittest.main()
