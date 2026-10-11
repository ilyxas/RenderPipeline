import tempfile,unittest
from pathlib import Path
from xms.render.process import render


class RenderProcess(unittest.TestCase):
    def test_successful_blender_run_returns_its_record(self):
        # /bin/echo stands in for Blender: the success path must return, not just the error path.
        with tempfile.TemporaryDirectory() as tmp:
            config={'blender':'/bin/echo','character_asset':'a.blend','character_profile':'p','scene_asset':'s.blend','scene_profile':'s.json'}
            result=render(Path(tmp)/'bundle',Path(tmp)/'frames',config,'24',Path(tmp)/'log.txt')
            self.assertEqual(result['exit_status'],0);self.assertIn('/bin/echo',result['command'])
    def test_failed_blender_run_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={'blender':'/usr/bin/false','character_asset':'a','character_profile':'p','scene_asset':'s','scene_profile':'s'}
            with self.assertRaisesRegex(RuntimeError,'Blender exit'):render(Path(tmp)/'b',Path(tmp)/'f',config,'24',Path(tmp)/'log.txt')


if __name__=='__main__':unittest.main()
