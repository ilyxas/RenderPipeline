import unittest
from pathlib import Path
import numpy as np
from xms.profiles.character import load_character
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate
from xms.animation.validate import validate
from xms.solve.grounding import ground,sole_proxies,sole_positions
from xms.solve.contacts import constraints
from xms.solve.video_body import set_direction
from xms.solve.head import local_delta_for_world,rotation

class Grounding(unittest.TestCase):
    def setup_bundle(self):
        p,r,_=load_character(Path(__file__).resolve().parents[1]/'profiles/characters/xandra/v3')
        b=neutral(p,r,np.arange(40)/24);b.arrays['root_translation'][:,1]=.2
        image=np.zeros((40,33,3));image[:,[27,28,29,30,31,32],1]=.9;image[:,[23,24],1]=.5
        return b,image,np.ones((40,33),bool)

    def heights(self,b):
        import copy
        rest=copy.deepcopy(b);rest.arrays['root_translation'][:]=0;rest.arrays['local_rotation_delta'][:]=[0,0,0,1]
        points=sole_proxies(b,evaluate(rest,0));return np.array([sole_positions(evaluate(b,i),points)[:,:,1] for i in range(40)])

    def test_static_offset_is_removed_and_contract_retained(self):
        b,image,valid=self.setup_bundle();ground(b,image,valid,1.7);validate(b)
        self.assertLess(np.abs(self.heights(b).min(axis=(1,2))).max(),.003)
        self.assertGreater(len(b.metadata['grounding_diagnostics']['contact_segments']),0)

    def test_switching_support_keeps_free_leg_lifted(self):
        b,image,valid=self.setup_bundle();names=b.metadata['joint_names']
        for side,a,z,toe in [('l',0,20,31),('r',20,40,32)]:
            j=names.index('calf_'+side);f=names.index('foot_'+side)
            desired=rotation(evaluate(b,0)[f])
            for i in range(a,z):
                set_direction(b,i,j,f,np.array([0,.2,.8]))
                fk=evaluate(b,i);b.arrays['local_rotation_delta'][i,f]=local_delta_for_world(desired,fk[j],b.arrays['rest_rotation'][f])
            image[a:z,toe,1]=.65
        ground(b,image,valid,1.7);h=self.heights(b)
        self.assertLess(np.abs(h.min(axis=(1,2))).max(),.003)
        self.assertGreater(h[10,0].min(),.05);self.assertGreater(h[30,1].min(),.05)
        self.assertEqual(b.metadata['grounding_diagnostics']['flight_sample_indices'],[])

    def test_source_jump_survives_and_events_use_source_evidence(self):
        b,image,valid=self.setup_bundle();b.arrays['root_translation'][12:25,1]+=.25
        image[12:25,[31,32],1]=.7;image[12:25,[23,24],1]=.3
        ground(b,image,valid,1.7,verified_static_camera=True);constraints(b)
        self.assertGreater(self.heights(b)[18].min(),.20);self.assertTrue(b.metadata['events'])
        self.assertLess(abs(self.heights(b)[3].min()),.003)

    def test_missing_feet_do_not_invent_support(self):
        b,image,valid=self.setup_bundle();valid[:,[27,28,31,32]]=False;before=b.arrays['root_translation'].copy();ground(b,image,valid,1.7)
        self.assertTrue(np.array_equal(before,b.arrays['root_translation']));self.assertEqual(b.metadata['grounding_diagnostics']['unsupported_samples'],40)

    def test_shared_image_translation_is_not_flight_evidence(self):
        from xms.solve.grounding import source_support
        b,image,valid=self.setup_bundle()
        image[12:25,:,1]-=.2
        reliable,flight=source_support(image,valid,b.arrays['sample_times_s'],1.7)
        self.assertTrue(reliable.all())
        self.assertFalse(flight.any())
