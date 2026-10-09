import unittest
import tempfile
from pathlib import Path
from xms.qa.contract import metric,EXIT_CODES,NeedsInput
from xms.report.html import report
from xms.observations.pose_mediapipe import pack_detection
from types import SimpleNamespace


class QA(unittest.TestCase):
    def test_unavailable_is_not_pass(self):
        m=metric(reason='No annotated stance',unit='m');self.assertEqual(m['status'],'unavailable');self.assertIsNone(m['value'])
        with tempfile.TemporaryDirectory() as d:
            report(d,{'status':'needs_review','metrics':{'foot_drift':m}})
            text=(Path(d)/'report.html').read_text();self.assertIn('class="unavailable"',text);self.assertNotIn('class="pass"',text)
    def test_ambiguous_subject_needs_input(self):
        with self.assertRaises(NeedsInput):pack_detection(SimpleNamespace(pose_landmarks=[[],[]]))
        self.assertEqual(EXIT_CODES['needs_input'],2)
