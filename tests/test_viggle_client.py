import os,tempfile,unittest
from pathlib import Path
from unittest import mock
from xms.observations import viggle


class ViggleCache(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);self.video=root/'clip.mp4';self.video.write_bytes(b'not a real video')
        self.env=mock.patch.dict(os.environ,{'XMS_VIGGLE_CACHE':str(root/'cache')});self.env.start()
        os.environ.pop('XMS_VIGGLE_ALLOW_UPLOAD',None)
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def test_cache_hit_needs_no_key_or_network(self):
        folder=viggle.cache_dir(self.video,24);folder.mkdir(parents=True);(folder/'metahuman_body.glb').write_bytes(b'glb')
        with mock.patch.object(viggle,'request',side_effect=AssertionError('network used')):
            self.assertEqual(viggle.fetch(self.video,24,log=lambda m:None),folder/'metahuman_body.glb')
    def test_cache_key_depends_on_content_and_fps(self):
        a=viggle.cache_dir(self.video,24);self.assertNotEqual(a,viggle.cache_dir(self.video,25))
        self.video.write_bytes(b'other content');self.assertNotEqual(a,viggle.cache_dir(self.video,24))
    def test_request_fields_fit_viggle_columns(self):
        self.assertLessEqual(len(viggle.task_id(self.video,23.976)),50)
        body,_=viggle.multipart({'type':'glb'},{'motion_video':self.video});self.assertIn(b'filename="source.mp4"',body)
    def test_miss_refuses_paid_upload_without_permission(self):
        with mock.patch.object(viggle,'request',side_effect=AssertionError('network used')):
            with self.assertRaisesRegex(RuntimeError,'XMS_VIGGLE_ALLOW_UPLOAD'):viggle.fetch(self.video,24,log=lambda m:None)


if __name__=='__main__':unittest.main()
