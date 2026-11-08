import unittest
import numpy as np

from starter.t27_integration import analyze_t27


class T27GradedSafetyTests(unittest.TestCase):

    def test_missing_graded_mask_rejected(self):
        features = {"n_frames": 5}
        labels = np.array(["H"] * 5)

        with self.assertRaisesRegex(ValueError, "graded mask is required"):
            analyze_t27(features, labels)

    def test_wrong_graded_mask_length_rejected(self):
        features = {"n_frames": 5}
        labels = np.array(["H"] * 5)
        graded = np.ones(4, dtype=bool)

        with self.assertRaisesRegex(ValueError, "graded length mismatch"):
            analyze_t27(features, labels, graded=graded)


if __name__ == "__main__":
    unittest.main()
