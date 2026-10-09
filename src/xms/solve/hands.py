"""Rest-length hand solve with bounded short gaps; fast observed gestures untouched."""
import numpy as np
from .baseline_hands import solve_hands as baseline
from .contacts import spans
from xms.animation.sample import slerp,continuous


def solve_hands(body,observations,timeline):
    result=baseline(body,observations,timeline);times=body.arrays['sample_times_s'];inferred=np.zeros_like(result['validity'])
    for j in result['indices']:
        for start,end in spans(~result['validity'][:,j],1):
            if start==0 or end==len(times) or times[end]-times[start-1]>.15:continue
            for i in range(start,end):result['rotations'][i,j]=slerp(result['rotations'][start-1,j],result['rotations'][end,j],(times[i]-times[start-1])/(times[end]-times[start-1]))
            result['validity'][start:end,j]=True;result['confidence'][start:end,j]=min(result['confidence'][start-1,j],result['confidence'][end,j])*.5;inferred[start:end,j]=True
    result['rotations']=continuous(result['rotations']);result['provenance']=np.where(inferred,2,result['validity'].astype(np.uint8)).astype(np.uint8)
    result['diagnostics']={'inferred_joint_samples':int(inferred.sum()),'policy':'No lowpass on valid gestures; gaps ≤150ms SLERP; longer gaps neutral. Wrist cap 95°, fingers 100°; fixed rest lengths.','metacarpals':'rest/unobserved; tracker does not identify their twist reliably'}
    return result


def palm_correction(bundle,max_degrees=5):
    """Joint bilateral bounded solve requires close 3D palms AND opposing normals."""
    from scipy.optimize import least_squares
    from .parameterization import quaternions_to_rotvec,rotvec_to_quaternions
    from xms.animation.fk import evaluate
    from .baseline_hands import hand_frame
    names=bundle.metadata['joint_names'];lookup={name:i for i,name in enumerate(names)};a=bundle.arrays;logs=[]
    selected=[lookup[x+'_'+s] for s in ('l','r') for x in ('upperarm','lowerarm','hand')];limit=np.deg2rad(max_degrees)/np.sqrt(3)
    def palms(world):
        centers=[];normals=[]
        for side in ('l','r'):
            w=world[lookup['hand_'+side],:3,3];middle=world[lookup['middle_01_'+side],:3,3];index=world[lookup['index_01_'+side],:3,3];pinky=world[lookup['pinky_01_'+side],:3,3]
            centers.append((w+middle+index+pinky)/4);normals.append(hand_frame(w,middle,index,pinky)[:,2])
        return np.array(centers),np.array(normals)
    for i in range(len(a['sample_times_s'])):
        if min(a['rotation_confidence'][i,[lookup['hand_l'],lookup['hand_r']]])<.5:continue
        world=evaluate(bundle,i);centers,normals=palms(world);distance=np.linalg.norm(centers[0]-centers[1])
        if distance>.09 or normals[0]@normals[1]>-.5:continue
        initial=a['local_rotation_delta'][i,selected].copy();v=quaternions_to_rotvec(initial);target=centers.mean(axis=0)
        def cost(x):
            a['local_rotation_delta'][i,selected]=rotvec_to_quaternions(v+x.reshape(-1,3));c,n=palms(evaluate(bundle,i))
            return np.r_[(c-target).ravel()*8,(n[0]+n[1])*.03,x*.1]
        before=float(np.linalg.norm(cost(np.zeros(18))));r=least_squares(cost,np.zeros(18),bounds=(-limit,limit),max_nfev=16,ftol=.01)
        after=float(np.linalg.norm(cost(r.x)))
        if after>=before:a['local_rotation_delta'][i,selected]=initial
        logs.append({'sample':i,'before':before,'after':min(before,after),'max_correction_degrees':float(np.degrees(np.linalg.norm(r.x.reshape(-1,3),axis=1).max())),'accepted':after<before})
    bundle.metadata['palm_diagnostics']={'samples':logs,'angular_budget_degrees':max_degrees,'quality_accepted':False}
    return bundle
