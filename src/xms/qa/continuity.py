import numpy as np


def measure(bundle,boundaries=()):
    a=bundle.arrays;dt=np.diff(a['sample_times_s']);q=a['local_rotation_delta']
    angles=2*np.arccos(np.clip(abs(np.sum(q[1:]*q[:-1],axis=-1)),0,1))
    speed=angles/dt[:,None] if len(dt) else np.empty((0,q.shape[1]))
    acceleration=np.diff(speed,axis=0)/dt[1:,None] if len(dt)>1 else np.empty((0,q.shape[1]))
    return {'max_rotation_step_deg':float(np.degrees(angles.max())) if angles.size else 0,'max_angular_acceleration_deg_s2':float(np.degrees(abs(acceleration).max())) if acceleration.size else 0,'boundaries':[{'sample':int(i),'root_step_m':float(np.linalg.norm(a['root_translation'][i]-a['root_translation'][i-1])),'rotation_step_deg':float(np.degrees(angles[i-1].max()))} for i in boundaries if 0<i<len(q)],'quality_accepted':False}
