import unittest
import numpy as np
from xms.solve.residuals import residual,sparsity,temporal_weights

class ConstraintDependencies(unittest.TestCase):
    def test_boundary_contact_collision_rows_match_objective(self):
        n=5;initial=np.zeros((n,1,3));initial[:,:,2]=.15
        rest=np.tile(np.eye(4),(5,1,1));rest[1,0,3]=1;rest[2,0,3]=1;rest[3,:3,3]=[.8,-.3,0];rest[4,1,3]=.6
        valid=np.ones(n,bool);conf=np.ones(n)
        c=dict(n=n,k=1,width=5,initial=initial,closure=list(range(5)),rest=rest,rest_rotation=np.tile(np.eye(3),(5,1,1)),scale=np.ones((5,3)),parents=np.array([-1,0,1,-1,3]),armature=np.eye(4),fixed_rotation=np.tile(np.eye(3),(n,5,1,1)),variable={0:0},root_initial=np.zeros((n,3)),camera={'world_anchor_m':[0,0,0],'world_to_camera_rotation':np.eye(3).tolist(),'metres_per_normalized_image_height':2,'display_aspect':1,'image_anchor':[.5,.5]},times=np.arange(n)/24,confidence=conf[:,None],weights=temporal_weights(initial,conf[:,None]),caps=np.array([2.]),points=[(2,np.zeros((n,2)),valid,conf)],directions=[(0,1,np.tile([1,0,0.],(n,1)),valid,conf)],boundary=(2,np.zeros((2,5))),contacts=[(2,np.zeros(3),valid,conf)],joint_lookup={'arm':1,'hand':2,'hip':3,'neck':4},collision_proxies={'margin_m':.005,'capsules':{'forearm':{'start':'arm','end':'hand','radius_m':.1},'torso':{'start':'hip','end':'neck','radius_m':.1}},'pairs':[['forearm','torso']]})
        x=np.zeros(n*5);before=residual(x,c);pattern=sparsity(c).toarray();self.assertEqual(pattern.shape,(len(before),len(x)))
        for column in range(len(x)):
            changed=x.copy();changed[column]+=1e-5;affected=abs(residual(changed,c)-before)>1e-10
            self.assertTrue(np.all(pattern[affected,column]),f'Missing dependency in column {column}')
