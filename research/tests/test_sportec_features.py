import tempfile
import unittest
from pathlib import Path

import numpy as np

from sportec_features import extract_sportec


class SportecFeatureTests(unittest.TestCase):
    def fixture(self, root, status='1', possession='1'):
        (root/'matchinformation_TEST.xml').write_text('''<Root><General HomeTeamId="home"
GuestTeamId="away"/><Environment PitchX="100" PitchY="60"/></Root>''')
        (root/'positions_TEST.xml').write_text('''<Root>
<FrameSet GameSection="firstHalf" TeamId="home" PersonId="h">
<Frame N="100" X="0" Y="0"/><Frame N="101" X="0" Y="0"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="away" PersonId="a">
<Frame N="100" X="10" Y="0"/><Frame N="101" X="10" Y="0"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="ball" PersonId="ball">
<Frame N="100" X="0.1" Y="0" BallStatus="%s" BallPossession="%s"/>
<Frame N="101" X="1.1" Y="0" BallStatus="%s" BallPossession="%s"/>
</FrameSet></Root>''' % (status, possession, status, possession))

    def test_centered_units_frame_offset_and_team_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            f, gt, graded = extract_sportec('TEST', root)
            self.assertEqual(f['n_frames'], 2)
            np.testing.assert_allclose(f['ball_x_norm'], [.501, .511], atol=1e-7)
            np.testing.assert_array_equal(f['nearest_team'], [0, 0])
            self.assertAlmostEqual(f['ball_speed'][1], 25., places=4)
            self.assertEqual(''.join(gt), 'HH')
            self.assertTrue(graded.all())

    def test_annotation_values_do_not_enter_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            before, truth_before, _ = extract_sportec('TEST', root)
            self.fixture(root, status='0', possession='2')
            after, truth_after, _ = extract_sportec('TEST', root)
            self.assertEqual(''.join(truth_before), 'HH')
            self.assertEqual(''.join(truth_after), 'DD')
            for key in before:
                np.testing.assert_equal(before[key], after[key], err_msg=key)


if __name__ == '__main__':
    unittest.main()
