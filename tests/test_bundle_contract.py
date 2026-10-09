import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from xms.animation.bundle import neutral
from xms.animation.io import read_bundle,write_bundle
from xms.animation.validate import validate


def fixture():
    p={'profile_hash':'a'*64,'joint_names':['root','head'],'face_channel_names':['jawOpen'],'face_ranges':[[0,1]],'root_carrier_index':0,'ownership':{'head':{'owner':'body','joints':['head'],'faces':[]},'eyes':{'owner':'neutral','joints':[],'faces':[]},'jaw':{'owner':'morph','joints':[],'faces':['jawOpen']}}}
    r={'parent_indices':np.array([-1,0]),'rest_translation':np.array([[0.,0,0],[1.,0,0]]),'rest_rotation':np.array([[0.,0,0,1],[0.,0,0,1]]),'rest_scale':np.ones((2,3)),'armature_transform':np.eye(4)}
    b=neutral(p,r,[0,1]); b.arrays['local_rotation_delta'][1,0]=[0,0,np.sqrt(.5),np.sqrt(.5)]
    b.arrays['root_translation'][1]=[0,2,0]; b.arrays['face_weights'][1,0]=.5
    for g in ('rotation','root','face'):
        b.arrays[g+'_validity'][:]=True;b.arrays[g+'_confidence'][:]=1;b.arrays[g+'_provenance'][:]=3
    return b


class Contract(unittest.TestCase):
    def test_roundtrip_and_immutable(self):
        b=fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bundle';write_bundle(b,p);c=read_bundle(p,'a'*64)
            self.assertEqual(b.metadata,c.metadata)
            for k in b.arrays: np.testing.assert_array_equal(b.arrays[k],c.arrays[k])
            with self.assertRaises(FileExistsError):write_bundle(b,p)
    def test_rejections(self):
        mutations=[lambda b:b.arrays['parent_indices'].__setitem__(0,1),lambda b:b.arrays['root_translation'].__setitem__((0,0),np.nan),lambda b:b.arrays['sample_times_s'].__setitem__(1,0),lambda b:b.metadata['ownership']['jaw']['joints'].append('head'),lambda b:b.metadata.__setitem__('schema_version','future'),lambda b:b.arrays['local_rotation_delta'].__setitem__((0,0),[0,0,0,2]),lambda b:b.arrays['face_weights'].__setitem__((1,0),2)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                b=fixture();mutate(b)
                with self.assertRaises(ValueError):validate(b)
        with self.assertRaises(ValueError):validate(fixture(),'b'*64)
