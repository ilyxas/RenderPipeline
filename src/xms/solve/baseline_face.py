import numpy as np


def smooth_valid(values,valid,exclude=()):
    result=values.copy()
    for i in range(1,len(values)-1):
        if valid[i-1:i+2].all():result[i]=.25*values[i-1]+.5*values[i]+.25*values[i+1]
    if exclude:result[:,list(exclude)]=values[:,list(exclude)]
    return result


def solve_face(observations,timeline,profile,face_map):
    m,a=observations;indices=np.array(timeline['output_observation_indices']);lookup={name:i for i,name in enumerate(m['blendshape_names'])};names=profile['face_channel_names'];n=len(indices)
    targets=[(v['object'],v['key']) for v in face_map['channels'].values()]
    if len(targets)!=len(set(targets)):raise ValueError('Duplicate face aliases claim the same target')
    weights=np.zeros((n,len(names)));valid=np.zeros_like(weights,dtype=bool);confidence=np.zeros_like(weights)
    frame_valid=a['validity'][indices]&(a['confidence'][indices]>=.2)
    for j,name in enumerate(names):
        if name=='_neutral' or name not in lookup:continue
        weights[:,j]=np.clip(a['blendshapes_raw'][indices,lookup[name]],*profile['face_ranges'][j]);valid[:,j]=frame_valid;confidence[:,j]=np.where(frame_valid,a['confidence'][indices],0)
    # Preserve blink/jaw timing; smooth only supported contiguous expression samples.
    weights=smooth_valid(weights,frame_valid,[i for i,n in enumerate(names) if n.startswith('eyeBlink') or n=='jawOpen']);weights[~valid]=0
    return {'weights':weights,'validity':valid,'confidence':confidence,'provenance':valid.astype(np.uint8),'ignored_observation_channels':[n for n in lookup if n not in names or n=='_neutral']}
