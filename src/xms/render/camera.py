"""Presentation framing only; reconstruction/reference camera stays fixed."""
import math
import numpy as np
from xms.animation.fk import evaluate

PRESETS={'preview':{'resolution':[540,960],'samples':16},'final':{'resolution':[1080,1920],'samples':64}}


def presentation(bundle,scene,quality='preview',mode='fit'):
    preset=PRESETS[quality];camera=dict(scene['camera']);names=bundle.metadata['joint_names'];ids=[names.index(n) for n in ('head','foot_l','foot_r','hand_l','hand_r','lowerarm_l','lowerarm_r','pelvis')]
    bounds=[]
    for i in range(len(bundle.arrays['sample_times_s'])):
        pos=evaluate(bundle,i)[ids,:3,3];bounds.append(np.r_[pos.min(axis=0)-.12,pos.max(axis=0)+.12])
    bounds=np.array(bounds);low=bounds[:,:3].min(axis=0);high=bounds[:,3:].max(axis=0);center=(low+high)/2;half=(high-low)/2
    angle=math.radians(camera['vertical_fov_degrees'])/2;aspect=preset['resolution'][0]/preset['resolution'][1]
    if mode=='fit':
        distance=max(abs(camera['position_blender_m'][1]),half[1]/math.tan(angle),half[0]/(math.tan(angle)*aspect))+half[2]+.1
        camera['position_blender_m']=[float(center[0]),float(-center[2]-distance),float(center[1])]
    return {'camera':camera,**preset,'mode':mode,'canonical_bounds_by_sample_m':bounds.tolist(),'policy':'Static full-clip skeletal bounds plus 12cm surface margin, stable FOV; independent of reference calibration.'}
