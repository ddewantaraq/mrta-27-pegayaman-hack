import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dead_ball_model import (design_features, fit_logistic, decode_dead,
                             protect_hidden_live, load_model)
from features import empty_feature_arrays


class DeadBallModelTests(unittest.TestCase):
    def features(self):
        f = empty_feature_arrays(100)
        f['n_frames'] = 100
        f['ball_present'][:] = True
        f['ball_x_norm'][:] = .5
        f['ball_y_norm'][:] = .5
        f['ball_speed'][:] = 0
        f['nearest_dist'][:] = 2
        return f

    def test_missing_observations_have_finite_features(self):
        f = empty_feature_arrays(100)
        self.assertTrue(np.isfinite(design_features(f)).all())

    def test_packaged_model_matches_feature_schema(self):
        model = load_model()
        self.assertEqual(len(model['mean']), design_features(self.features()).shape[1])

    def test_decoder_preserves_ungraded_frames(self):
        f = self.features()
        d = design_features(f).shape[1]
        model = dict(mean=np.zeros(d), scale=np.ones(d), beta=np.r_[10., np.zeros(d)])
        graded = np.ones(100, bool)
        graded[40:60] = False
        dead = decode_dead(f, model, graded)
        np.testing.assert_array_equal(dead, graded)

    def test_fitting_learns_direction(self):
        x = np.arange(-10, 11, dtype=float)[:, None]
        y = x[:, 0] > 0
        model = fit_logistic(x, y, ridge=.1)
        self.assertGreater(model['beta'][1], 0)
        self.assertTrue(np.isfinite(model['beta']).all())

    def test_protection_cannot_lose_a_correct_hidden_live_answer(self):
        baseline = np.array(list('HADDXX'))
        prediction = np.array(list('DDHAHA'))
        out = protect_hidden_live(prediction, baseline, np.zeros(6, bool))
        self.assertEqual(''.join(out), 'HAHAHA')
        np.testing.assert_array_equal(protect_hidden_live(
            prediction, baseline, np.ones(6, bool)), prediction)


if __name__ == '__main__':
    unittest.main()
