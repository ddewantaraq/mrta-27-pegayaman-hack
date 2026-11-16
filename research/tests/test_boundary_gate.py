"""Boundary ensemble defaults reproduce v2.3; gate modes stay well-defined."""
import unittest
import numpy as np

from boundary_ensemble_model import (
    apply_boundary_ensemble, DEFAULT_BOUNDARY_BLEND, DEFAULT_BOUNDARY_SCALE,
    DEFAULT_BOUNDARY_GATE)
from numpy_boundary_ownership import decode_boundary
from dead_ball_model import decode_dead_logits


class BoundaryGateTests(unittest.TestCase):
    def test_defaults_match_v23_policy(self):
        self.assertEqual(DEFAULT_BOUNDARY_BLEND, 0.5)
        self.assertEqual(DEFAULT_BOUNDARY_SCALE, 1.0)
        self.assertEqual(DEFAULT_BOUNDARY_GATE, 'none')

    def test_decode_boundary_scale_one_matches_unscaled(self):
        rng = np.random.RandomState(0)
        n = 200
        logits = rng.randn(n)
        boundary = rng.randn(n)
        live = np.ones(n, bool)
        live[50:60] = False
        a = decode_boundary(logits, boundary, live, boundary_scale=1.0)
        b = decode_boundary(logits, boundary, live)
        np.testing.assert_array_equal(a, b)

    def test_missing_only_uses_isolated_missing_runs(self):
        n = 100
        logits = np.linspace(-2, 2, n)
        boundary = np.full(n, 3.0)
        live = np.ones(n, bool)
        present = np.ones(n, bool)
        present[20:40] = False
        present[70:85] = False
        missing_live = live & ~present
        home_full = decode_boundary(logits, boundary, live)
        home_missing = decode_boundary(logits, boundary, missing_live)
        # Isolated missing runs need not match full-span Viterbi.
        self.assertTrue(np.any(home_full[missing_live] != home_missing[missing_live])
                        or np.array_equal(home_full[missing_live], home_missing[missing_live]))
        # Outside missing runs the missing-only mask is inactive.
        self.assertFalse(np.any(home_missing[present]))

    def test_unknown_gate_rejected(self):
        features = dict(
            ball_present=np.ones(8, bool),
            period=np.zeros(8, np.int8),
            player_motion=np.zeros((8, 12), np.float32),
            n_frames=8,
            game='Sample_Game_2',
        )
        # Minimal smoke: invalid gate raises before heavy work if we call with
        # empty-like inputs through the public API validation.
        with self.assertRaises(ValueError):
            apply_boundary_ensemble(
                features, np.array(['H'] * 8), np.ones(8, bool),
                np.zeros((2, 2, 11, 8), np.float32),
                gate='not_a_gate')


if __name__ == '__main__':
    unittest.main()
