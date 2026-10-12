import copy,unittest
from pathlib import Path
import numpy as np
from xms.profiles.character import load_character
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate
from xms.solve import collision_refine as cr

ROOT=Path(__file__).resolve().parents[1]


def to_blender(v):return [v[0],-v[2],v[1]]


class CollisionRefine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p,rig,_=load_character(ROOT/'profiles/characters/xandra/v4');n=9;t=np.arange(n)/24.
        b=neutral(p,rig,t,t);a=b.arrays;cls.names=p['joint_names']
        for c in cr.CHAIN:
            for s in cr.SIDES:j=cls.names.index(f'{c}_{s}');a['rotation_validity'][:,j]=True;a['rotation_confidence'][:,j]=.6;a['rotation_provenance'][:,j]=1
        cls.b=b
    def witness(self,frame,depth=.012):
        w=evaluate(self.b,frame);l=w[self.names.index('lowerarm_l'),:3,3];h=w[self.names.index('hand_l'),:3,3];p=(l+h)/2
        # Fixed body surface 1.2 cm outward along -Z (toward the back): the forearm point must move back.
        q=p+np.array([0,0,-1.])*depth
        return {'class':'body','side':'l','depth_m':depth,'point_blender_m':to_blender(p),'target_blender_m':to_blender(q),'attach':'lowerarm_l','target_attach':'body'}
    def point_after(self,bundle,frame,w):
        arms=cr.Arms(self.b,np.arange(9));term=cr.build_terms(arms,[{**w,'frame':frame}])[0]
        world=evaluate(bundle,frame)[self.names.index('lowerarm_l')];return (world@term['p_local'])[:3],term
    def test_penetrating_point_exits_and_source_is_untouched(self):
        before=copy.deepcopy(self.b.arrays['local_rotation_delta']);w=self.witness(4,.004)
        cand=cr.refine(self.b,{'records':[{'sample':4,'witnesses':[w]}]})
        np.testing.assert_array_equal(self.b.arrays['local_rotation_delta'],before)
        p,term=self.point_after(cand,4,w);moved=(p-term['p'])@term['u']
        self.assertGreater(moved,.004);info=cand.metadata['collision_refinement']
        self.assertTrue(all(v<=5+1e-6 for v in info['increment_max_deg'].values()))
    def test_correction_ramps_in_and_out_without_a_jump(self):
        cand=cr.refine(self.b,{'records':[{'sample':4,'witnesses':[self.witness(4,.004)]}]})
        j=self.names.index('upperarm_l');q=cand.arrays['local_rotation_delta'][:,j]
        angle=np.degrees(2*np.arccos(np.clip(np.abs(q[:,3]),0,1)))
        self.assertGreater(angle[3],0.);self.assertLess(angle[3],angle[4]+1e-9);self.assertLess(np.max(np.abs(np.diff(angle))),angle[4]*.8+1e-6)
    def test_unchanged_penetration_is_rejected(self):
        report={'records':[{'sample':4,'witnesses':[self.witness(4)]}]}
        verdict=cr.accept(report,report,self.b,copy.deepcopy(self.b))
        self.assertFalse(verdict['accepted']);self.assertFalse(verdict['gates']['body_meaningful_reduction'])


if __name__=='__main__':unittest.main()
