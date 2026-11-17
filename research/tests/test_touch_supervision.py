import tempfile
import unittest
from pathlib import Path

import numpy as np

from touch_supervision import (
    accel_candidates,
    align_hits,
    frame25_from_frame10,
    parse_skillcorner_dynamic,
    parse_sportec_events_raw,
    sportec_frame_offset,
)


class TouchSupervisionTests(unittest.TestCase):
    def test_sportec_maps_event_time_to_frame_and_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'matchinformation_TEST.xml').write_text(
                '<Root><General HomeTeamId="home" GuestTeamId="away"/>'
                '<Environment PitchX="100" PitchY="60"/></Root>')
            (root / 'positions_TEST.xml').write_text('''<Root><Positions>
<FrameSet GameSection="firstHalf" TeamId="ball" PersonId="ball">
<Frame N="100" T="2023-05-27T13:30:12.600+00:00" X="0" Y="0" BallStatus="1"/>
<Frame N="101" T="2023-05-27T13:30:12.640+00:00" X="0" Y="0" BallStatus="1"/>
<Frame N="102" T="2023-05-27T13:30:12.680+00:00" X="0" Y="0" BallStatus="1"/>
</FrameSet></Positions></Root>''')
            (root / 'events_raw_TEST.xml').write_text('''<Root>
<Event EventId="1" EventTime="2023-05-27T13:30:12.642+00:00">
 <Play Player="h" Team="home"><Pass/></Play>
</Event>
<Event EventId="2" EventTime="2023-05-27T13:30:12.678+00:00">
 <TacklingGame WinnerTeam="away" PossessionChange="true"/>
</Event>
</Root>''')
            touches = parse_sportec_events_raw('TEST', root)
            kinds = {(t.frame_n, t.kind, t.team) for t in touches}
            self.assertIn((101, 'play', 0), kinds)
            self.assertIn((102, 'duel', 1), kinds)
            self.assertTrue(all(t.possession_change is False for t in touches
                                if t.kind == 'play'))
            self.assertTrue(any(t.possession_change for t in touches
                                if t.kind == 'duel'))
            self.assertEqual(sportec_frame_offset('TEST', root), 100)

    def test_skillcorner_maps_frames_and_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / '1874553'
            root.mkdir(parents=True)
            (root / '1874553_match.json').write_text(
                '{"home_team": {"id": 7}, "pitch_length": 105, "pitch_width": 68,'
                ' "players": [{"id": 99, "team_id": 7}, {"id": 100, "team_id": 7}]}')
            (root / '1874553_dynamic_events.csv').write_text(
                'event_type,event_subtype,frame_start,frame_end,period,team_id,'
                'player_in_possession_id,phase_index,team_possession_loss_in_phase\n'
                'player_possession,,36,40,1,7,,0,False\n'
                'on_ball_engagement,pressure,50,55,1,9,100,1,True\n'
                'passing_option,,60,60,1,7,,2,False\n')
            touches = parse_skillcorner_dynamic('SC_1874553', root.parent)
            kinds = {(t.frame10, t.frame25, t.kind, t.team) for t in touches}
            self.assertIn((36, 90, 'touch', 0), kinds)
            # duel team_id=9 is the pressing side; possessor id=100 is home
            self.assertIn((50, 125, 'duel', 0), kinds)
            # passing_option is off-ball and must not become a touch
            self.assertEqual(len(touches), 2)
            self.assertTrue(any(t.possession_loss for t in touches))

    def test_frame10_to_frame25_rounding(self):
        self.assertEqual(frame25_from_frame10(36), 90)
        self.assertEqual(frame25_from_frame10(0), 0)

    def test_accel_fires_on_kick_not_on_smooth_roll(self):
        n = 60
        bx = np.arange(n, dtype=float) * 0.2  # 5 m/s smooth
        bx[30:] += (np.arange(n - 30, dtype=float) ** 2) * 0.05  # kick at 30
        features = dict(ball_x_m=bx, ball_y_m=np.zeros(n),
                        ball_present=np.ones(n, bool), period=np.ones(n),
                        nearest_dist=np.full(n, 1.0))
        graded = np.ones(n, bool)
        idx = accel_candidates(features, graded)
        self.assertTrue(len(idx) > 0)
        self.assertTrue(all(abs(i - 30) <= 4 for i in idx))
        # smooth roll: no candidate
        features['ball_x_m'] = np.arange(n, dtype=float) * 0.2
        self.assertEqual(len(accel_candidates(features, graded)), 0)
        # missing ball around the kick: no candidate
        features['ball_x_m'] = bx
        features['ball_present'] = np.ones(n, bool)
        features['ball_present'][28:33] = False
        idx = accel_candidates(features, graded)
        self.assertTrue(all(not 26 <= i <= 34 for i in idx))

    def test_align_hits_counts_precision_recall(self):
        # candidates at 100/200/300, references at 102/205; tol 5 -> 2 hits
        precision, recall, hits = align_hits([100, 200, 300], [102, 205], tol=5)
        self.assertEqual(hits, 2)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 1.0)
        precision, recall, hits = align_hits([], [1], tol=5)
        self.assertEqual((precision, recall, hits), (0.0, 0.0, 0))


if __name__ == '__main__':
    unittest.main()
