import unittest
import numpy as np
from xms.profiles.character import load_character
from xms.profiles.scene import load_scene
from xms.animation.bundle import neutral
from xms.render.camera import presentation
from xms.qa.framing import measure
from xms.ingest.timeline import nearest_indices

class Presentation(unittest.TestCase):
    def test_full_clip_movement_fits_preview_and_final(self):
        p,r,_=load_character('profiles/characters/xandra/v3');scene=load_scene('profiles/scenes/bedroom/v3/scene.json');b=neutral(p,r,[0.,.04,.08]);b.arrays['root_translation'][:,0]=[-2,0,2]
        for quality in ('preview','final'):
            plan=presentation(b,scene,quality);self.assertEqual(measure(plan)['framed_fraction'],1);self.assertEqual(plan['samples'],16 if quality=='preview' else 64)
    def test_linear_memory_nearest_preserves_ties(self):
        source=np.array([0.,.1,.4]);targets=np.array([-.1,.05,.11,.3,1.]);np.testing.assert_array_equal(nearest_indices(source,targets),np.abs(source[:,None]-targets).argmin(axis=0))
