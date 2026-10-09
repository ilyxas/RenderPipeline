import unittest
import numpy as np
from test_bundle_contract import fixture
from xms.animation.fk import evaluate,trs,matrix_quat,quat_matrix
from xms.animation.sample import sample,continuous


class FK(unittest.TestCase):
    def test_nonuniform_rest_scale_order(self):
        b=fixture();b.arrays['rest_scale'][0]=[2,3,4]
        np.testing.assert_allclose(evaluate(b,1)[1,:3,3],[0,4,0],atol=1e-10)
    def test_analytical(self):
        b=fixture();np.testing.assert_allclose(evaluate(b,1)[1,:3,3],[0,3,0],atol=1e-10)
        np.testing.assert_allclose(evaluate(sample(b,[.5]),0)[1,:3,3],[np.sqrt(.5),1+np.sqrt(.5),0],atol=1e-10)
    def test_armature_scale_rotation_translation(self):
        b=fixture(); b.arrays['armature_transform']=trs([3,4,5],[0,0,np.sqrt(.5),np.sqrt(.5)],[2,3,4])
        np.testing.assert_allclose(evaluate(b,1)[1,:3,3],[0,6,5],atol=1e-10)
    def test_sign_and_halfturn(self):
        b=fixture();b.arrays['local_rotation_delta'][1]*=-1
        q=sample(b,[.5]).arrays['local_rotation_delta'];self.assertLess(np.max(abs(np.linalg.norm(q,axis=-1)-1)),1e-4)
        np.testing.assert_allclose(quat_matrix(matrix_quat(np.diag([1,-1,-1]))),np.diag([1,-1,-1]),atol=1e-10)
        self.assertGreater(np.sum(continuous(np.array([[0.,0,0,1],[0,0,0,-1]]))[1]),0)
