import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from xms.render.engine import check,configure,estimate_seconds,exposure
from xms.render.process import render


class RenderEngine(unittest.TestCase):
    def test_exposure_falls_back_to_the_scene_value(self):
        self.assertEqual(exposure({'exposure':0.3},'cycles'),0.3)
        profile={'exposure':0.0,'engine_exposure':{'cycles':0.45}}
        self.assertEqual(exposure(profile,'cycles'),0.45);self.assertEqual(exposure(profile,'eevee'),0.0)

    def test_unknown_engine_is_rejected(self):
        with self.assertRaises(ValueError):check('workbench')

    def test_cycles_estimate_is_much_slower_than_eevee(self):
        eevee=estimate_seconds('eevee','final',[1080,1920],180);cycles=estimate_seconds('cycles','final',[1080,1920],180)
        self.assertGreater(cycles,8*eevee);self.assertTrue(30*60<cycles<60*60)

    def test_configure_cycles_without_gpu_falls_back_to_cpu(self):
        scene=SimpleNamespace(render=SimpleNamespace(engine='',use_motion_blur=False,motion_blur_shutter=0),cycles=SimpleNamespace(),eevee=SimpleNamespace(taa_render_samples=16))
        record=configure(scene,'cycles','final')
        self.assertEqual(scene.render.engine,'CYCLES');self.assertEqual(record['device'],'CPU');self.assertEqual(record['samples'],64)
        self.assertTrue(scene.render.use_motion_blur);self.assertTrue(scene.cycles.use_denoising)
        record=configure(scene,'eevee','final')
        self.assertEqual(scene.render.engine,'BLENDER_EEVEE_NEXT');self.assertFalse(scene.render.use_motion_blur);self.assertEqual(record,{'engine':'eevee','samples':16})

    def test_render_passes_the_engine_to_blender(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={'blender':'/bin/echo','character_asset':'a.blend','character_profile':'p','scene_asset':'s.blend','scene_profile':'s.json'}
            cmd=render(Path(tmp)/'bundle',Path(tmp)/'frames',config,'24',Path(tmp)/'log.txt',engine='cycles')['command']
            self.assertEqual(cmd[cmd.index('--engine')+1],'cycles')


if __name__=='__main__':unittest.main()
