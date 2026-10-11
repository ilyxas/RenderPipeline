import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from xms.solve.image_hands import axial_roll

class PalmPlane(unittest.TestCase):
    def test_moving_reference_does_not_clip_the_visible_palm(self):
        angle=np.deg2rad(np.arange(0.,361.,2.))
        axis=np.tile([0.,1,0],(len(angle),1))
        base=Rotation.from_rotvec(axis*angle[:,None]).apply(np.tile([1.,0,0],(len(angle),1)))
        target=np.tile([1.,0,0],(len(angle),1))
        relative=Rotation.from_rotvec(-axis*angle[:,None]).as_quat()
        relative[::3]*=-1
        roll=axial_roll(relative,axis)
        aligned=Rotation.from_rotvec(axis*roll[:,None]).apply(base)
        np.testing.assert_allclose(aligned,target,atol=1e-8)
        self.assertLess(np.abs(np.diff(roll)).max(),np.deg2rad(3))

    def test_arm_plane_changes_preserve_elbow_and_wrist_positions(self):
        from pathlib import Path
        from xms.profiles.character import load_character
        from xms.animation.bundle import neutral
        from xms.animation.fk import evaluate
        from xms.solve.image_hands import solve
        p,r,_=load_character(Path(__file__).resolve().parents[1]/'profiles/characters/xandra/v3')
        times=np.arange(8)/24;body=neutral(p,r,times);body.metadata['image_retarget_diagnostics']={'neutral_sample_indices':[0]}
        ids={s:i for i,s in enumerate(p['joint_names'])};rest=evaluate(body,0);lm=np.zeros((8,2,21,3))
        for si,side in enumerate(('l','r')):
            wrist=rest[ids['hand_'+side],:3,3];elbow=rest[ids['lowerarm_'+side],:3,3];axis=(wrist-elbow)/np.linalg.norm(wrist-elbow)
            points=[wrist]
            for finger in ('thumb','index','middle','ring','pinky'):
                bones=[rest[ids[f'{finger}_{k:02d}_{side}'],:3,3] for k in (1,2,3)]
                points.extend(bones+[bones[-1]+(bones[-1]-bones[-2])])
            for i in range(8):lm[i,si]=(Rotation.from_rotvec(axis*np.deg2rad(i*10)).apply(np.array(points)-wrist)+wrist)*[1,-1,-1]
        observations=({}, {'world_landmarks_raw':lm,'validity':np.ones((8,2),bool),'confidence':np.ones((8,2))})
        tracked=[ids[f'{bone}_{side}'] for side in ('l','r') for bone in ('upperarm','lowerarm','hand')]
        before=np.array([evaluate(body,i)[tracked,:3,3] for i in range(8)])
        solve(body,observations,{'output_observation_indices':list(range(8))})
        after=np.array([evaluate(body,i)[tracked,:3,3] for i in range(8)])
        np.testing.assert_allclose(after,before,atol=5e-6)

    def test_distal_fingers_cannot_splay_and_knuckle_splay_is_bounded(self):
        from xms.solve.image_hands import finger_rotation
        current=np.array([[0.,1,0]]);target=np.array([[1.,1,0]])/np.sqrt(2);hinge=np.array([[1.,0,0]])
        distal,_=finger_rotation(current,target,hinge)
        np.testing.assert_allclose(distal[0],np.eye(3),atol=1e-8)
        knuckle,limited=finger_rotation(current,target,hinge,mcp=True)
        self.assertTrue(limited[0]);self.assertAlmostEqual(Rotation.from_matrix(knuckle).magnitude()[0],np.deg2rad(15))
