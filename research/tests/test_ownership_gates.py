import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ownership_gates import replacement_mask


class OwnershipGateTests(unittest.TestCase):
    def test_only_coherent_missing_live_changes_are_accepted(self):
        f = {'ball_present':np.zeros(100,bool),'period':np.ones(100,np.int8)}
        reference = np.full(100,'H')
        logits = np.full(100,-3.)
        f['ball_present'][:10] = True
        reference[40:50] = 'D'
        reference[80:] = 'X'
        mask = replacement_mask(f,reference,np.ones(100,bool),logits,'run90_mean90')
        expected = np.zeros(100,bool); expected[10:40] = True; expected[50:80] = True
        np.testing.assert_array_equal(mask,expected)

    def test_a_period_boundary_cannot_join_short_changes(self):
        f = {'ball_present':np.zeros(40,bool),'period':np.r_[np.ones(20),np.full(20,2)]}
        mask = replacement_mask(f,np.full(40,'A'),np.ones(40,bool),np.full(40,4.),'run90_mean90')
        self.assertFalse(mask.any())

    def test_mean_evidence_and_duration_are_both_required(self):
        f = {'ball_present':np.zeros(40,bool),'period':np.ones(40,np.int8)}
        logits = np.full(40,-.5); logits[20] = -5.
        mask = replacement_mask(f,np.full(40,'H'),np.ones(40,bool),logits,'run90_mean90')
        self.assertFalse(mask.any())


if __name__ == '__main__':
    unittest.main()
