import math
import numpy as np


def measure(plan):
    camera=plan['camera'];x,by,y=camera['position_blender_m'];angle=math.radians(camera['vertical_fov_degrees'])/2;aspect=plan['resolution'][0]/plan['resolution'][1];ok=[]
    for row in plan['canonical_bounds_by_sample_m']:
        low,high=np.array(row[:3]),np.array(row[3:]);nearest=-high[2]-by;vertical=nearest*math.tan(angle);horizontal=vertical*aspect
        ok.append(bool(nearest>0 and low[0]>=x-horizontal and high[0]<=x+horizontal and low[1]>=y-vertical and high[1]<=y+vertical))
    return {'status':'passed' if all(ok) else 'needs_review','planned_frames':len(ok),'framed_fraction':sum(ok)/len(ok),'offending_samples':[i for i,v in enumerate(ok) if not v],'geometry':'skeletal bounds plus margin, not pixel silhouette','intentional_crop':False}
