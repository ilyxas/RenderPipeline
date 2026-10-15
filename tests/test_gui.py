import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from xms.gui import Runner,command_for,env_for
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

    def test_viggle_solver_and_explicit_upload_consent(self):
        with tempfile.TemporaryDirectory() as tmp:
            video=Path(tmp)/'in.mp4';video.touch()
            cmd=command_for({'input':str(video),'start':0,'duration':2,'solver':'viggle'},Path(tmp)/'runs')
            self.assertEqual(cmd[cmd.index('--solver')+1],'viggle');self.assertEqual(cmd[cmd.index('--channels')+1],'full')
            base={'XMS_VIGGLE_ALLOW_UPLOAD':'1','VIGGLE_API_KEY':'k'}
            self.assertNotIn('XMS_VIGGLE_ALLOW_UPLOAD',env_for({'solver':'viggle'},base))
            self.assertNotIn('XMS_VIGGLE_ALLOW_UPLOAD',env_for({'solver':'video','viggle_upload':True},base))
            self.assertEqual(env_for({'solver':'viggle','viggle_upload':True},base)['XMS_VIGGLE_ALLOW_UPLOAD'],'1')
            self.assertEqual(env_for({'solver':'viggle'},base)['VIGGLE_API_KEY'],'k')

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


class RemoteAccess(unittest.TestCase):
    def test_only_explicit_addresses_and_tailscale_range(self):
        from unittest import mock
        from xms import gui
        self.assertEqual(gui.resolve_hosts(),['127.0.0.1'])
        for host in ['0.0.0.0','::','my-mac.local']:
            with self.assertRaises(ValueError):gui.resolve_hosts([host])
        self.assertTrue(gui.is_tailscale_address('100.101.2.3'));self.assertFalse(gui.is_tailscale_address('192.168.1.5'));self.assertFalse(gui.is_tailscale_address('100.128.0.1'))
        ifconfig='lo0: inet 127.0.0.1 netmask 0xff000000\nen0: inet 192.168.1.20 netmask 0xffffff00\nutun4: inet 100.88.12.7 --> 100.88.12.7 netmask 0xffffffff\n'
        with mock.patch.object(gui,'_tailscale_cli',return_value=None),mock.patch.object(gui,'_run_text',side_effect=lambda cmd:ifconfig if cmd==['ifconfig'] else ''):
            self.assertEqual(gui.resolve_hosts(tailscale=True),['127.0.0.1','100.88.12.7'])
        with mock.patch.object(gui,'tailscale_addresses',return_value=[]):
            with self.assertRaises(ValueError):gui.resolve_hosts(tailscale=True)

    def test_upload_is_streamed_named_by_content_and_bounded(self):
        import io
        from xms.gui import receive_upload,safe_upload_name
        self.assertEqual(safe_upload_name('IMG_4321.MOV'),'IMG_4321.mov');self.assertEqual(safe_upload_name('..%2F..%2Fetc%2Fclip.mp4'),'clip.mp4')
        for name in ['run.sh','noext','']:
            with self.assertRaises(ValueError):safe_upload_name(name)
        with tempfile.TemporaryDirectory() as tmp:
            data=os.urandom(3*2**20+17)
            first=receive_upload(io.BytesIO(data),len(data),'IMG_1.MOV',tmp,2**30);second=receive_upload(io.BytesIO(data),len(data),'IMG_1.MOV',tmp,2**30)
            self.assertEqual(first,second);self.assertEqual(first.read_bytes(),data);self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),[first.name])
            with self.assertRaises(ValueError):receive_upload(io.BytesIO(data[:100]),len(data),'cut.mov',tmp,2**30)
            with self.assertRaises(ValueError):receive_upload(io.BytesIO(data),len(data),'big.mov',tmp,2**20)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),[first.name])
