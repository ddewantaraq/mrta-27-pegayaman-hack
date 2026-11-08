import unittest
import numpy as np

from starter.t27_integration import analyze_t27


class T27IntegrationTests(unittest.TestCase):

    def test_empty_candidates_preserve_labels(self):
        features = {
            "n_frames": 5,
            "ball_present": np.zeros(5, dtype=bool),
            "ball_x_m": np.full(5, np.nan),
            "ball_y_m": np.full(5, np.nan),
            "nearest_team": np.full(5, -1),
            "nearest_dist": np.full(5, np.nan),
        }

        labels = np.array(["H", "H", "H", "H", "H"])
        before = labels.copy()

        candidates = analyze_t27(features, labels, graded=np.ones(len(labels), dtype=bool))

        self.assertEqual(candidates, [])
        np.testing.assert_array_equal(labels, before)


if __name__ == "__main__":
    unittest.main()
