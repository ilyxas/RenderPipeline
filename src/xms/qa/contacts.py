import numpy as np
from xms.animation.fk import evaluate


def measure(bundle):
    drifts=[]
    for record in bundle.metadata.get('contact_anchors',[]):
        j=bundle.metadata['joint_names'].index(record['joint']);anchor=np.array(record['anchor_world_m'])
        drifts.append(max(np.linalg.norm(evaluate(bundle,i)[j,:3,3]-anchor) for i in record['sample_indices']))
    return {'status':'unavailable' if not drifts else ('passed' if np.median(drifts)<=.02 and np.percentile(drifts,95)<=.04 else 'needs_review'),'segments':len(drifts),'median_segment_max_drift_m':float(np.median(drifts)) if drifts else None,'p95_segment_max_drift_m':float(np.percentile(drifts,95)) if drifts else None,'geometry':'skeletal foot origin proxy'}
