"""Boundary checks for the experimental live-island filter."""
import unittest

import numpy as np

from short_spell_experiment import suppress_live_islands


class LiveIslandTests(unittest.TestCase):
    def test_only_dead_enclosed_islands(self):
        labels = np.array(list('DDHHDDHAAHXXAD'))
        out = suppress_live_islands(labels, np.ones(len(labels), bool), 3)
        self.assertEqual(''.join(out), 'DDDDDDHAAHXXAD')
        self.assertEqual(''.join(labels), 'DDHHDDHAAHXXAD')

    def test_threshold_is_strict(self):
        labels = np.array(list('DHHHD'))
        np.testing.assert_array_equal(
            suppress_live_islands(labels, np.ones(5, bool), 3), labels)

    def test_ungraded_boundary_is_not_crossed(self):
        labels = np.array(list('DHD'))
        for graded in ([False, True, True], [True, True, False], [True, False, True]):
            np.testing.assert_array_equal(
                suppress_live_islands(labels, graded, 12), labels)

    def test_missing_ball_island_is_preserved(self):
        labels = np.array(list('DHD'))
        np.testing.assert_array_equal(suppress_live_islands(
            labels, np.ones(3, bool), 12, np.array([True, False, True])), labels)


if __name__ == '__main__':
    unittest.main()
