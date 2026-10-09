import unittest
import numpy as np
from scipy.optimize import least_squares
from xms.solve.residuals import residual,sparsity,forward,temporal_weights

class TemporalRecovery(unittest.TestCase):
    def test_noisy_fast_motion_outlier_and_missing(self):
        # Known fixed-length arm, actual 2D projection + weak 3D observations.
        rng=np.random.default_rng(8);n=60;times=np.arange(n)/24
        true=np.zeros((n,1,3));true[:,0,2]=.35*np.sin(times*2);true[20:29,0,2]+=.65*np.sin(np.linspace(0,np.pi,9))
        noisy=true+rng.normal(0,.08,true.shape);noisy[12,0,2]+=.65
        valid=np.ones(n,bool);valid[34:38]=False;confidence=valid.astype(float);confidence[12]=.05
        identity=np.eye(4);child=identity.copy();child[0,3]=1
        camera={'world_anchor_m':[0,0,0],'world_to_camera_rotation':np.eye(3).tolist(),'metres_per_normalized_image_height':2,'display_aspect':1,'image_anchor':[.5,.5]}
        c=dict(n=n,k=1,width=5,initial=noisy,closure=[0,1],rest=[identity,child],rest_rotation=[np.eye(3)]*2,scale=np.ones((2,3)),parents=np.array([-1,0]),armature=identity,fixed_rotation=np.tile(np.eye(3),(n,2,1,1)),variable={0:0},root_initial=np.zeros((n,3)),camera=camera,times=times,confidence=confidence[:,None],weights=temporal_weights(noisy,confidence[:,None]),caps=np.array([2.]))
        truth=c.copy();truth['initial']=true;position=forward(np.zeros(n*5),truth)[0][1]
        from xms.qa.pose import project
        target=project(position,camera)+rng.normal(0,.02,(n,2));target[12]+=.5
        direction=position/np.linalg.norm(position,axis=1,keepdims=True)
        c.update(points=[(1,target,valid,confidence)],directions=[(0,1,direction,valid,confidence)])
        x0=np.zeros(n*5);result=least_squares(lambda x:residual(x,c),x0,jac_sparsity=sparsity(c),loss='soft_l1',f_scale=.015,max_nfev=80,ftol=1e-3,xtol=1e-3,gtol=1e-4)
        solved=forward(result.x,c)[1];self.assertTrue(result.success)
        self.assertLess(np.mean((solved-true)**2),np.mean((noisy-true)**2))
        self.assertLess(np.mean(np.diff(solved,n=2,axis=0)**2),np.mean(np.diff(noisy,n=2,axis=0)**2))
        self.assertLess(abs(solved[24,0,2]-true[24,0,2]),np.deg2rad(5))
        self.assertLess(np.mean((solved[34:38]-true[34:38])**2),np.mean((noisy[34:38]-true[34:38])**2))
        # Missing targets cannot influence the solution, even with large finite values.
        before=residual(result.x,c);target[~valid]=1000;np.testing.assert_array_equal(before,residual(result.x,c))
        # Every numerical residual dependency must be represented in the sparse Jacobian.
        pattern=sparsity(c).toarray();base=residual(x0,c)
        for column in (0,2,42,122,291):
            x=x0.copy();x[column]=1e-5;affected=abs(residual(x,c)-base)>1e-10
            self.assertTrue(np.all(pattern[affected,column]))

if __name__=='__main__':unittest.main()
