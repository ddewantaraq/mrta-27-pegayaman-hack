import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boosted_dead_model import boosted_logits


class BoostedInferenceTests(unittest.TestCase):
    def test_numeric_threshold_and_missing_routing(self):
        tree = dict(feature=np.array([0, 0, 0]), threshold=np.array([1., 0., 0.]),
                    value=np.array([0., -2., 3.]), left=np.array([1, 0, 0]),
                    right=np.array([2, 0, 0]), leaf=np.array([0, 1, 1]),
                    missing_left=np.array([1, 0, 0]))
        model = dict(baseline=.5, trees=[tree])
        x = np.array([[0.], [1.], [2.], [np.nan]])
        np.testing.assert_array_equal(boosted_logits(x, model), [-1.5, -1.5, 3.5, -1.5])


if __name__ == '__main__':
    unittest.main()
