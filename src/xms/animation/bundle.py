from dataclasses import dataclass
import copy
import numpy as np

VERSION = 'xms.animation_bundle.v1'
PROVENANCE = {'unobserved': 0, 'observed': 1, 'body_prior': 2, 'synthetic': 3}


@dataclass
class AnimationBundle:
    metadata: dict
    arrays: dict


def neutral(profile, rig, times, source_times=None):
    times = np.asarray(times, dtype=np.float64)
    n, j, f = len(times), len(profile['joint_names']), len(profile['face_channel_names'])
    m = {k: copy.deepcopy(profile[k]) for k in ('joint_names','face_channel_names','face_ranges','root_carrier_index','ownership')}
    m.update(schema_version=VERSION, character_profile_hash=profile['profile_hash'],
             units='metres', coordinate_system='RH_Y_UP_Z_FORWARD', quaternion_order='XYZW',
             rotation_semantics='rest_local_times_delta', provenance_codes=PROVENANCE,
             contacts=[], events=[], quality='baseline_preview')
    a = {k: np.array(v, copy=True) for k,v in rig.items()}
    a.update(sample_times_s=times, source_time_map=times.copy() if source_times is None else np.asarray(source_times,dtype=np.float64),
             local_rotation_delta=np.tile([0.,0.,0.,1.],(n,j,1)), root_translation=np.zeros((n,3)), face_weights=np.zeros((n,f)))
    for group, shape in [('rotation',(n,j)), ('root',(n,)), ('face',(n,f))]:
        a[group+'_validity']=np.zeros(shape,dtype=bool)
        a[group+'_confidence']=np.zeros(shape,dtype=np.float32)
        a[group+'_provenance']=np.zeros(shape,dtype=np.uint8)
    return AnimationBundle(m,a)
