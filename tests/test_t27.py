import unittest
import numpy as np
from starter.pass_travel_t27 import detect_same_team_travel


def sample(end_team=0, end_x=12, end_frame=20):
    n = 60
    f = dict(n_frames=n, ball_present=np.zeros(n, dtype=bool),
             ball_x_m=np.full(n, np.nan), ball_y_m=np.full(n, np.nan),
             nearest_team=np.full(n, -1), nearest_dist=np.full(n, np.nan))
    for i, team, x in [(2, 0, 0), (end_frame, end_team, end_x)]:
        f['ball_present'][i] = True
        f['ball_x_m'][i] = x
        f['ball_y_m'][i] = 10
        f['nearest_team'][i] = team
        f['nearest_dist'][i] = 0.5
    return f


class T27Tests(unittest.TestCase):
    def test_same_team_missing_ball(self):
        c = detect_same_team_travel(sample())
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0].start_frame, c[0].end_frame, c[0].team), (2, 20, 0))

    def test_opposing_team_rejected(self):
        self.assertEqual(detect_same_team_travel(sample(end_team=1)), [])

    def test_impossible_speed_rejected(self):
        self.assertEqual(detect_same_team_travel(sample(end_x=70)), [])

    def test_tiny_displacement_rejected(self):
        self.assertEqual(detect_same_team_travel(sample(end_x=1)), [])

    def test_dead_and_ungraded_rejected(self):
        dead = np.zeros(60, dtype=bool)
        dead[10] = True
        self.assertEqual(detect_same_team_travel(sample(), dead_mask=dead), [])
        graded = np.ones(60, dtype=bool)
        graded[10] = False
        self.assertEqual(detect_same_team_travel(sample(), graded=graded), [])

    def test_gap_limit(self):
        self.assertEqual(detect_same_team_travel(sample(), max_gap_frames=10), [])

    def test_bad_shapes(self):
        f = sample()
        f['nearest_team'] = f['nearest_team'][:-1]
        with self.assertRaises(ValueError):
            detect_same_team_travel(f)

    def test_no_contact(self):
        f = sample()
        f['nearest_dist'][20] = 5
        self.assertEqual(detect_same_team_travel(f), [])

    def test_no_mutation(self):
        f = sample()
        original = f['ball_present'].copy()
        detect_same_team_travel(f)
        np.testing.assert_array_equal(original, f['ball_present'])


if __name__ == '__main__':
    unittest.main()
