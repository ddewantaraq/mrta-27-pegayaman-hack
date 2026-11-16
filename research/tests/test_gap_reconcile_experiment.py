import unittest

import numpy as np

from gap_reconcile_experiment import reconcile_gaps


class GapReconcileTests(unittest.TestCase):
    def test_same_team_endpoints_replace_stale_hidden_state(self):
        labels = np.array(list('HAAH'))
        out = reconcile_gaps(labels, [True, False, False, True], [True]*4)
        self.assertEqual(''.join(out), 'HHHH')
        self.assertEqual(''.join(labels), 'HAAH')

    def test_opposing_endpoints_split_at_midpoint(self):
        labels = np.array(list('HHHHA'))
        out = reconcile_gaps(labels, [True, False, False, False, True], [True]*5)
        self.assertEqual(''.join(out), 'HHAAA')
        np.testing.assert_array_equal(reconcile_gaps(
            labels, [True, False, False, False, True], [True]*5, 'same'), labels)

    def test_dead_unknown_and_ungraded_are_barriers(self):
        for middle in ('D', 'X'):
            labels = np.array(list('HA'+middle+'AH'))
            np.testing.assert_array_equal(reconcile_gaps(
                labels, [True, False, False, False, True], [True]*5), labels)
        labels = np.array(list('HAAH'))
        np.testing.assert_array_equal(reconcile_gaps(
            labels, [True, False, False, True], [True, True, False, True]), labels)

    def test_no_extrapolation_at_edges(self):
        labels = np.array(list('AAH'))
        np.testing.assert_array_equal(reconcile_gaps(
            labels, [False, False, True], [True]*3), labels)


if __name__ == '__main__':
    unittest.main()
