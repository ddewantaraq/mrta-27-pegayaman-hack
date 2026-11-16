import unittest

from metrica_reference_v2 import labels_with_explicit_stoppages


def event(kind, frame, end=None):
    return dict(Team='Home', Type=kind, Subtype='', Period=1,
                **{'Start Frame': frame, 'End Frame': end or frame})


class ExplicitStoppageReferenceTests(unittest.TestCase):
    def test_explicit_foul_followed_by_restart_adds_dead(self):
        events = [event('PASS', 1), event('FAULT RECEIVED', 10),
                  event('SET PIECE', 100), event('PASS', 101, 150)]
        labels = labels_with_explicit_stoppages(events, 150)
        self.assertTrue((labels[9:99] == 'D').all())
        self.assertEqual(labels[99], 'H')

    def test_play_continues_after_foul_without_inferred_stoppage(self):
        events = [event('PASS', 1), event('FAULT RECEIVED', 10),
                  event('PASS', 20), event('SET PIECE', 100), event('PASS', 101, 150)]
        labels = labels_with_explicit_stoppages(events, 150)
        self.assertTrue((labels == 'H').all())


if __name__ == '__main__':
    unittest.main()
