import unittest
import numpy as np
from xms.solve.windows import intervals
from xms.animation.io import read_bundle
from xms.qa.continuity import measure

class Windows(unittest.TestCase):
    def test_terminal_and_coverage(self):
        for n in (1,60,72,73,240):
            ranges=list(intervals(n,24));self.assertEqual(ranges[-1][1],n)
            covered=set(i for a,b in ranges for i in range(a,b));self.assertEqual(covered,set(range(n)))
            self.assertTrue(all(b-a<=72 for a,b in ranges))
    def test_boundary_fast_gesture_is_reported(self):
        b=read_bundle('tests/fixtures/synthetic_bundle');r=measure(b,[1]);self.assertEqual(len(r['boundaries']),1);self.assertFalse(r['quality_accepted'])
