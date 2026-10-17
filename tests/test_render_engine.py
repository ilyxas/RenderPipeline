import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from xms.render.engine import check,configure,estimate_seconds,exposure,samples_for
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
        self.assertTrue(scene.render.use_motion_blur);self.assertTrue(scene.cycles.use_denoising);self.assertTrue(scene.render.use_persistent_data)
        record=configure(scene,'eevee','final')
        self.assertEqual(scene.render.engine,'BLENDER_EEVEE_NEXT');self.assertFalse(scene.render.use_motion_blur);self.assertEqual(record,{'engine':'eevee','samples':32})

    def test_samples_override_quality_default(self):
        self.assertEqual(samples_for('cycles','preview'),32);self.assertEqual(samples_for('eevee','final'),32);self.assertEqual(samples_for('cycles','final',96),96)
        for bad in (0,5000,'64',2.5,True):
            with self.assertRaises(ValueError):samples_for('cycles','final',bad)
        scene=SimpleNamespace(render=SimpleNamespace(engine='',use_motion_blur=False,motion_blur_shutter=0),cycles=SimpleNamespace(),eevee=SimpleNamespace(taa_render_samples=16))
        self.assertEqual(configure(scene,'cycles','preview',samples=128)['samples'],128);self.assertEqual(scene.cycles.samples,128)
        self.assertEqual(configure(scene,'eevee','preview',samples=8)['samples'],8);self.assertEqual(scene.eevee.taa_render_samples,8)
        base=estimate_seconds('cycles','final',[1080,1920],100);self.assertEqual(estimate_seconds('cycles','final',[1080,1920],100,64),base)
        self.assertLess(estimate_seconds('cycles','final',[1080,1920],100,32),base);self.assertGreater(estimate_seconds('cycles','final',[1080,1920],100,128),base)

    def test_render_passes_the_engine_to_blender(self):
        with tempfile.TemporaryDirectory() as tmp:
            config={'blender':'/bin/echo','character_asset':'a.blend','character_profile':'p','scene_asset':'s.blend','scene_profile':'s.json'}
            cmd=render(Path(tmp)/'bundle',Path(tmp)/'frames',config,'24',Path(tmp)/'log.txt',engine='cycles')['command']
            self.assertEqual(cmd[cmd.index('--engine')+1],'cycles');self.assertNotIn('--samples',cmd)
            cmd=render(Path(tmp)/'bundle',Path(tmp)/'frames',config,'24',Path(tmp)/'log.txt',engine='cycles',samples=48)['command']
            self.assertEqual(cmd[cmd.index('--samples')+1],'48')


if __name__=='__main__':unittest.main()
