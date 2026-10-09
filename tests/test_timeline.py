import unittest
from fractions import Fraction
from xms.ingest.timeline import make_timeline


class Timeline(unittest.TestCase):
    def test_rates_and_offsets(self):
        for rate in ['24','30','30000/1001']:
            f=Fraction(rate);p={'fps':rate,'time_base':str(1/f),'pts':list(range(120,240)),'audio':{'start_time':str(float(120/f)+.125)},'rotation_degrees':0,'sample_aspect_ratio':'1:1'}
            t=make_timeline(p,0,2)
            self.assertEqual(t['frame_count'],60 if rate!='24' else 48)
            self.assertAlmostEqual(t['audio_offset_from_clip_s'],.125)
            self.assertEqual(t['source_time_map'][0],float(120/f))
            self.assertTrue(all(x<2 for x in t['output_times_s']))
    def test_no_detection(self):
        from types import SimpleNamespace
        from xms.observations.pose_mediapipe import pack_detection
        image,world,vis,pres,valid=pack_detection(SimpleNamespace(pose_landmarks=[]))
        self.assertFalse(valid);self.assertEqual(vis.sum()+pres.sum(),0)
