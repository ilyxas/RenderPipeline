"""Bounded overlapping solves with one global calibration and explicit fallbacks."""
import copy
import numpy as np
from xms.animation.sample import continuous,slerp
from xms.animation.validate import validate
from .baseline_body import solve as baseline
from .calibration import calibrate

SAMPLED=('sample_times_s','source_time_map','local_rotation_delta','root_translation','face_weights',
         'rotation_validity','rotation_confidence','rotation_provenance','root_validity','root_confidence','root_provenance','face_validity','face_confidence','face_provenance')


def intervals(n,fps,seconds=3,overlap=.75):
    size=max(3,round(seconds*fps));shared=max(2,round(overlap*fps))
    if shared>=size:raise ValueError('Overlap must be shorter than window')
    start=0
    while start<n:
        end=min(n,start+size);yield start,end
        if end==n:break
        start=end-shared


def solve(observations,timeline,profile,rig,calibration=None,max_nfev=80):
    from .temporal_body import solve as window_solve,context
    from .parameterization import quaternions_to_rotvec
    calibration=calibrate(observations,timeline,profile,rig) if calibration is None else calibration
    base=baseline(observations,timeline,profile,rig,calibration);result=copy.deepcopy(base)
    times=base.arrays['sample_times_s'];fps=1/np.median(np.diff(times)) if len(times)>1 else 24
    logs=[];seams=[];previous_end=0
    for start,end in intervals(len(times),fps):
        part=copy.deepcopy(base)
        for key in SAMPLED:part.arrays[key]=base.arrays[key][start:end].copy()
        local=copy.deepcopy(timeline)
        for key in ('output_times_s','source_time_map','output_observation_indices'):local[key]=timeline[key][start:end]
        boundary=None
        if start<previous_end:
            count=previous_end-start;c=context(part,observations,local,calibration)
            target=np.zeros((count,c['width']));target[:,:2]=result.arrays['root_translation'][start:previous_end,:2]-part.arrays['root_translation'][:count,:2]
            target[:,2:]=np.stack([quaternions_to_rotvec(result.arrays['local_rotation_delta'][start:previous_end,j]) for j in c['selected']],axis=1).reshape(count,-1)-c['initial'][:count].reshape(count,-1)
            boundary=(count,target)
        try:
            solved=window_solve(observations,local,profile,rig,calibration,max_nfev,part,boundary)
            diag=solved.metadata.get('temporal_diagnostics',{'success':True,'termination':'short_baseline'})
            if not diag['success']:raise RuntimeError(diag['message'])
        except (RuntimeError,TimeoutError) as error:
            solved=part;diag={'success':False,'fallback':'baseline','error':str(error)}
        for key in SAMPLED:
            if key not in ('sample_times_s','source_time_map'):result.arrays[key][previous_end:end]=solved.arrays[key][previous_end-start:]
        # Boundary constraints preserve the already-published overlap; no crossfade
        # of fast gestures, no duplicated times, and no terminal padding.
        if start:seams.append(previous_end)
        previous_end=end;logs.append({'start':start,'end':end,**diag});print(f'SOLVE window {start}:{end} {diag["success"]}',flush=True)
    result.arrays['local_rotation_delta']=continuous(result.arrays['local_rotation_delta'])
    from xms.qa.continuity import measure
    result.metadata.update(quality='development_candidate',temporal_diagnostics={'success':True,'windows':logs,'fallback_windows':sum(not x['success'] for x in logs),'continuity':measure(result,seams)},limitations=['Experimental windowed temporal solver; motion-quality acceptance pending.','Exhausted windows use baseline and are recorded; monocular depth remains ambiguous.'])
    result.metadata['solver'].update(backend='temporal_windows',version='2',window_seconds=3,overlap_seconds=.75)
    return validate(result)
