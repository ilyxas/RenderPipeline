import json,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from xms.profiles.character import load_character
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate
from xms.solve.viggle import retarget,SPEC

ROOT=Path(__file__).resolve().parents[1];SAMPLE=Path.home()/'.cache/xms/viggle/text-probe/metahuman_body.glb'


def unit(v):return v/np.linalg.norm(v)
def angle(a,b):return float(np.degrees(np.arccos(np.clip(unit(a)@unit(b),-1,1))))


class ViggleRetarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile,cls.rig,_=load_character(ROOT/'profiles/characters/xandra/v3');cls.spec=json.loads(SPEC.read_text())
        cls.names=cls.profile['joint_names'];cls.rest=evaluate(neutral(cls.profile,cls.rig,[0.]),0)
        cls.offset=Rotation.from_quat(cls.spec['axis_offset']['xyzw']).as_matrix()
    def frame(self,name):m=self.rest[self.names.index(name),:3,:3];return m/np.linalg.norm(m,axis=0)
    def posed_in_xandra_rest(self,n):
        """A Viggle skeleton whose bones sit exactly in Xandra's rest pose."""
        world={x:np.tile(self.frame(x)@self.offset.T,(n,1,1)) for x in self.spec['body_joints']}
        return world,np.tile(self.rest[self.profile['root_carrier_index'],:3,3],(n,1))
    def run_retarget(self,world,pelvis,rest_pelvis):
        t=np.arange(len(pelvis))/24.;return retarget(world,pelvis,rest_pelvis,t,t,self.profile,self.rig,self.spec)
    def test_xandra_rest_pose_gives_zero_deltas(self):
        world,pelvis=self.posed_in_xandra_rest(3);b,root=self.run_retarget(world,pelvis,pelvis[0])
        degrees=np.degrees(Rotation.from_quat(b.arrays['local_rotation_delta'].reshape(-1,4)).magnitude())
        self.assertLess(degrees.max(),1e-5);self.assertLess(np.abs(b.arrays['root_translation']).max(),.005)
    def test_right_shoulder_forward_45_degrees(self):
        world,pelvis=self.posed_in_xandra_rest(1)
        # The rigid right arm swings 45 deg toward +Z (forward), about the axis normal to the arm and forward.
        arm=self.rest[self.names.index('lowerarm_r'),:3,3]-self.rest[self.names.index('upperarm_r'),:3,3]
        flex=Rotation.from_rotvec(unit(np.cross(arm,[0,0,1]))*np.radians(45)).as_matrix()
        for x in ('upperarm_r','lowerarm_r','hand_r'):world[x]=flex@world[x]
        b,_=self.run_retarget(world,pelvis,pelvis[0]);pose=evaluate(b,0);P=lambda x,m:m[self.names.index(x),:3,3]
        before=P('lowerarm_r',self.rest)-P('upperarm_r',self.rest);after=P('lowerarm_r',pose)-P('upperarm_r',pose)
        self.assertAlmostEqual(angle(before,after),45,delta=.01);self.assertAlmostEqual(angle(flex@before,after),0,delta=.01)
        self.assertGreater(P('hand_r',pose)[2]-P('upperarm_r',pose)[2],P('hand_r',self.rest)[2]-P('upperarm_r',self.rest)[2]+.2)
        # Nothing outside the right arm subtree moved.
        parents=self.rig['parent_indices'];arm=set()
        for i in range(len(self.names)):
            if self.names[i]=='upperarm_r' or parents[i] in arm:arm.add(i)
        others=[i for i in range(len(self.names)) if i not in arm]
        self.assertLess(np.abs(pose[others,:3,3]-self.rest[others,:3,3]).max(),1e-6)
    @unittest.skipUnless(SAMPLE.exists(),'real Viggle sample not cached')
    def test_real_viggle_rest_is_a_t_posed_xandra(self):
        from xms.animation.gltf import Skeleton
        s=Skeleton(SAMPLE);rotations,positions=s.world();idx={x:s.names.index(x) for x in self.spec['body_joints']}
        world={x:rotations[:,i] for x,i in idx.items()};b,_=self.run_retarget(world,positions[:,idx['pelvis']],positions[0,idx['pelvis']]);pose=evaluate(b,0);P=lambda x:pose[self.names.index(x),:3,3]
        for side,sign in (('l',1),('r',-1)):
            self.assertLess(angle(P('lowerarm_'+side)-P('upperarm_'+side),[sign,0,0]),3);self.assertLess(angle(P('hand_'+side)-P('lowerarm_'+side),[sign,0,0]),3)
            self.assertLess(angle(P('calf_'+side)-P('thigh_'+side),self.rest[self.names.index('calf_'+side),:3,3]-self.rest[self.names.index('thigh_'+side),:3,3]),3)


if __name__=='__main__':unittest.main()
