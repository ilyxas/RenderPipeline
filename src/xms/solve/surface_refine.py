"""At most one bounded correction pass from signed, evaluated surface evidence."""
import copy
import numpy as np
from xms.animation.fk import evaluate


def refine(bundle,report):
    from scipy.optimize import least_squares
    b=copy.deepcopy(bundle);lookup={name:i for i,name in enumerate(b.metadata['joint_names'])};logs=[];targets={};skipped=[]
    for row in report['records']:
        evidence=row.get('worst_evidence')
        if not row.get('signed_interior_available') or not evidence or evidence['depth_m']<=.005:continue
        i=row['sample'];world=evaluate(b,i);point=np.array(evidence['point_blender_m'])[[0,2,1]]*[1,1,-1];normal=np.array(evidence['normal_blender'])[[0,2,1]]*[1,1,-1]
        side=min(('l','r'),key=lambda s:np.linalg.norm(world[lookup['hand_'+s],:3,3]-point));key=(i,side)
        if key not in targets or targets[key][1]<evidence['depth_m']:targets[key]=(normal,evidence['depth_m'])
    for (i,side),(normal,depth) in targets.items():
        ids=[lookup[x+'_'+side] for x in ('upperarm','lowerarm','hand')]
        if not b.arrays['rotation_validity'][i,ids].all():
            skipped.append({'sample':i,'side':side,'reason':'Unobserved arm/wrist channels; bounded correction unavailable'});continue
        initial=b.arrays['local_rotation_delta'][i,ids].copy();wrist=lookup['hand_'+side];target=evaluate(b,i)[wrist,:3,3]+normal*min(depth+.002,.03)
        def cost(x):
            from scipy.spatial.transform import Rotation
            b.arrays['local_rotation_delta'][i,ids]=(Rotation.from_quat(initial)*Rotation.from_rotvec(x.reshape(3,3))).as_quat();return np.r_[(evaluate(b,i)[wrist,:3,3]-target)*10,x*.1]
        limit=np.deg2rad(5)/np.sqrt(3);result=least_squares(cost,np.zeros(9),bounds=(-limit,limit),max_nfev=12);cost(result.x)
        logs.append({'sample':i,'side':side,'surface_depth_before_m':depth,'max_correction_degrees':float(np.degrees(np.linalg.norm(result.x.reshape(3,3),axis=1).max()))})
    b.metadata['surface_refinement']={'passes':int(bool(logs)),'max_passes':1,'angular_budget_degrees':5,'translation_target_budget_m':.03,'samples':logs,'skipped_samples':skipped,'quality_accepted':False}
    return b
