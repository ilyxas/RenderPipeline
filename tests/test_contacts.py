import unittest
import numpy as np
from xms.solve.contacts import detect

class Contacts(unittest.TestCase):
    def test_stance_flight_visibility(self):
        t=np.arange(60)/24;p=np.zeros((60,3));p[20:40,1]=.3;c=np.ones(60)
        ranges=detect(p,t,c,0);self.assertTrue(any(a==0 and b<=20 for a,b in ranges));self.assertTrue(any(a>=40 for a,b in ranges));self.assertFalse(any(a<=25<b for a,b in ranges))
        self.assertEqual(detect(p,t,c*0,0),[])
