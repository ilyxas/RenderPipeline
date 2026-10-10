import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from xms.gui import Runner,command_for
from xms.render.camera import parse_resolution

class Launcher(unittest.TestCase):
    def test_options_reach_existing_cli_and_resolution_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'input with spaces.mp4';video.touch()
            settings={'input':str(video),'start':2,'duration':3,'resolution':'720x1280','quality':'final','solver':'temporal','audio':'speech','surface_qa':'all','body_only':True,'face_closeup':True,'fixed_camera':True,'no_refine':True,'output':str(Path(tmp)/'out.mp4')}
            cmd=command_for(settings,Path(tmp)/'runs')
            for option,value in [('--start','2.0'),('--end','5.0'),('--resolution','720x1280'),('--quality','final'),('--solver','temporal'),('--channels','body'),('--audio','speech'),('--surface-qa','all'),('--camera','fixed')]:self.assertEqual(cmd[cmd.index(option)+1],value)
            self.assertIn('--no-refine',cmd);self.assertNotIn('--face-closeup',cmd);self.assertIn(str(video.resolve()),cmd)
            self.assertEqual(parse_resolution('1080x1920'),[1080,1920])
            for value in ['1081x1920','0x960','720','huge']:
                with self.assertRaises(ValueError):parse_resolution(value)

    def test_whole_video_uses_pipeline_exact_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'input.mp4';video.touch()
            cmd=command_for({'input':str(video),'start':2,'duration':13.041667,'to_end':True},Path(tmp)/'runs')
            self.assertNotIn('--end',cmd)

    def test_stop_kills_descendants_even_after_cli_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'in.mp4';video.touch();pulse=root/'pulse';child=root/'child.py';parent=root/'parent.py'
            child.write_text("import signal,time\nfrom pathlib import Path\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\np=Path("+repr(str(pulse))+")\nwhile True:\n p.write_text(str(time.time_ns()));time.sleep(.03)\n")
            parent.write_text("import subprocess,time\nsubprocess.Popen(["+repr(sys.executable)+","+repr(str(child))+"])\nprint('XMS_PROGRESS pose 2 10',flush=True)\ntime.sleep(60)\n")
            runner=Runner(root/'runs',[sys.executable,'-u',str(parent)])
            try:
                runner.start({'input':str(video),'duration':1})
                deadline=time.monotonic()+5
                while not pulse.exists() and time.monotonic()<deadline:time.sleep(.03)
                self.assertTrue(pulse.exists());self.assertEqual(runner.snapshot()['completed'],2)
                runner.stop();self.assertIsNotNone(runner.process.poll());self.assertEqual(runner.snapshot()['status'],'cancelled')
                before=pulse.read_text();time.sleep(.15);self.assertEqual(pulse.read_text(),before)
            finally:runner.stop()
