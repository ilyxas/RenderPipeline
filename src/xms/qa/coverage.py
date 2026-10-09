import numpy as np
from .contract import metric


def missing_intervals(valid,times):
    times=np.asarray(times);step=float(np.median(np.diff(times))) if len(times)>1 else 0
    intervals=[];start=None
    for i,ok in enumerate(valid):
        if not ok and start is None:start=i
        if start is not None and (ok or i==len(times)-1):
            end=i if ok else i+1
            intervals.append({'start_source_s':float(times[start]),'end_source_s':float(times[end]) if end<len(times) else float(times[-1]+step)})
            start=None
    return intervals


def coverage_metrics(bundle,observations):
    meta,a=observations;b=bundle.arrays
    body=float(np.mean(a['frame_validity']))
    result={'pose_detection_coverage':metric(body,'fraction',passed=body==1),
            'body_rotation_observed_fraction':metric(float(np.mean(np.any(b['rotation_validity'],axis=1))),'fraction'),
            'face_tracking':metric(reason='Face observations unavailable in body-only baseline'),
            'hand_tracking':metric(reason='Hand observations unavailable in body-only baseline'),
            'foot_drift':metric(reason='No independently annotated high-confidence stance intervals',unit='m'),
            'surface_penetration':metric(reason='No verified surface/contact geometry in this stage',unit='m'),
            'lip_sync':metric(reason='No independently annotated mouth events',unit='s')}
    return result
