import numpy as np
from xms.animation.fk import evaluate
from xms.geometry.proxies import terms


def measure(bundle,definitions):
    world=np.array([evaluate(bundle,i) for i in range(len(bundle.arrays['sample_times_s']))]);lookup={name:i for i,name in enumerate(bundle.metadata['joint_names'])};positions={i:world[:,i,:3,3] for i in range(len(lookup))};depths=terms(positions,lookup,definitions)
    return {'status':'needs_review' if depths.size and depths.max()>.005 else 'measured','max_proxy_depth_m':float(depths.max()) if depths.size else None,'p95_proxy_depth_m':float(np.percentile(depths,95)) if depths.size else None,'offending_samples':np.flatnonzero((depths>.005).any(axis=1)).tolist(),'geometry':'experimental skeletal capsule proxies; not surface acceptance','quality_accepted':False}
