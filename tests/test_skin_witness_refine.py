import json,unittest
from pathlib import Path
import numpy as np
from xms.profiles.character import load_character
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate
from xms.solve.surface_refine import refine

ROOT=Path(__file__).resolve().parents[1]


class SkinWitnessRefine(unittest.TestCase):
    def setUp(self):
        p,rig,_=load_character(ROOT/'profiles/characters/xandra/v4');self.b=neutral(p,rig,np.array([0.]),np.array([0.]))
        a=self.b.arrays;n=p['joint_names']
        for x in ('upperarm','lowerarm','hand'):
            for s in 'lr':a['rotation_validity'][:,n.index(f'{x}_{s}')]=True;a['rotation_confidence'][:,n.index(f'{x}_{s}')]=.6;a['rotation_provenance'][:,n.index(f'{x}_{s}')]=1
        self.names=n
    def evidence(self,side):
        # A witness point at the LEFT hand that names the side to correct explicitly.
        w=evaluate(self.b,0)[self.names.index('hand_l'),:3,3];point=[w[0],-w[2],w[1]]
        return {'records':[{'sample':0,'signed_interior_available':True,'worst_evidence':{'depth_m':.02,'point_blender_m':point,'normal_blender':[0,-1,0],'side':side}}]}
    def test_witness_side_overrides_nearest_hand(self):
        log=refine(self.b,self.evidence('r')).metadata['surface_refinement']['samples']
        self.assertEqual([x['side'] for x in log],['r'])
    def test_without_side_the_nearest_hand_is_used(self):
        report=self.evidence('r');del report['records'][0]['worst_evidence']['side']
        log=refine(self.b,report).metadata['surface_refinement']['samples']
        self.assertEqual([x['side'] for x in log],['l'])
    def test_spec_is_bound_to_the_registered_skeleton(self):
        spec=json.loads((ROOT/'profiles/characters/xandra/skin_witnesses.v1.json').read_text());p,_,_=load_character(ROOT/'profiles/characters/xandra/v4')
        self.assertEqual(spec['rig_sha256'],p['files']['rig.npz'])


if __name__=='__main__':unittest.main()
