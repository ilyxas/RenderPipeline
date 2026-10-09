import unittest,copy
import numpy as np
from test_bundle_contract import fixture
from xms.solve.baseline_face import solve_face,smooth_valid
from xms.solve.head import local_delta_for_world
from xms.observations.hands_mediapipe import assign_sides
from xms.animation.fk import quat_matrix
from xms.animation.compose import compose


class Details(unittest.TestCase):
    def test_face_named_neutral_alias_and_gaps(self):
        p={'face_channel_names':['jawOpen','eyeBlinkLeft'],'face_ranges':[[0,1],[0,1]]};m={'blendshape_names':['_neutral','eyeBlinkLeft','jawOpen']};a={'validity':np.array([True,False,True]),'confidence':np.array([.8,0,.8]),'blendshapes_raw':np.array([[1.,.2,.7],[1,.9,.9],[1,.4,.6]])};mapping={'channels':{'jawOpen':{'object':'head','key':'jawOpen'},'eyeBlinkLeft':{'object':'head','key':'eyeBlinkLeft'}}};t={'output_observation_indices':[0,1,2]}
        f=solve_face((m,a),t,p,mapping);np.testing.assert_allclose(f['weights'],[[.7,.2],[0,0],[.6,.4]]);self.assertIn('_neutral',f['ignored_observation_channels'])
        mapping['channels']['eyeBlinkLeft']['key']='jawOpen'
        with self.assertRaises(ValueError):solve_face((m,a),t,p,mapping)
    def test_mirror_preserves_anatomical_side(self):
        body=np.array([[.8,.5],[.2,.5]]);hands=np.array([[.22,.5],[.78,.5]])
        self.assertEqual(assign_sides(hands,body,[True,True]),[(1,True),(0,True)])
        body[:,0]=1-body[:,0];hands[:,0]=1-hands[:,0]
        self.assertEqual(assign_sides(hands,body,[True,True]),[(1,True),(0,True)])
    def test_forearm_world_rotation_not_double_applied(self):
        q=np.array([0,0,np.sqrt(.5),np.sqrt(.5)]);parent=np.eye(4);parent[:3,:3]=quat_matrix(q)
        delta=local_delta_for_world(np.eye(3),parent,[0,0,0,1]);np.testing.assert_allclose(parent[:3,:3]@quat_matrix(delta),np.eye(3),atol=1e-10)
    def test_head_replaces_body_and_jaw_stays_morph(self):
        b=fixture();b.metadata['ownership']['head']['owner']='face_with_body_fallback';b.arrays['local_rotation_delta'][:,1]=[0,0,.5,np.sqrt(.75)]
        face={'weights':b.arrays['face_weights'],'validity':b.arrays['face_validity'],'confidence':b.arrays['face_confidence'],'provenance':b.arrays['face_provenance']};head={'index':1,'face_validity':np.array([True,False]),'rotations':np.tile([0,0,0,1.],(2,1)),'confidence':np.array([.8,0])};hands={'indices':[]}
        result=compose(b,face,head,hands);np.testing.assert_allclose(result.arrays['local_rotation_delta'][0,1],[0,0,0,1]);np.testing.assert_allclose(result.arrays['local_rotation_delta'][1,1],b.arrays['local_rotation_delta'][1,1]);self.assertEqual(result.metadata['head_source_by_sample'],['face','body_prior']);self.assertEqual(result.metadata['ownership']['jaw']['joints'],[])
        hands['indices']=[1]
        with self.assertRaises(ValueError):compose(b,face,head,hands)
