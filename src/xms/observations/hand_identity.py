import itertools
import numpy as np


def assign(wrists,body_wrists,body_valid,previous=None,mirrored=False):
    """Anatomical body labels are authoritative; temporal costs break ambiguities."""
    wrists=np.asarray(wrists,dtype=float).reshape(-1,2);body_wrists=np.array(body_wrists,dtype=float,copy=True)
    if mirrored:wrists[:,0]=1-wrists[:,0];body_wrists[:,0]=1-body_wrists[:,0]
    if len(wrists)>2:raise ValueError('Single-person hands only')
    if not len(wrists):return []
    best=None
    for sides in itertools.permutations(range(2),len(wrists)):
        costs=[];matches=[]
        for wrist,side in zip(wrists,sides):
            distance=np.linalg.norm(wrist-body_wrists[side]) if body_valid[side] else np.inf
            temporal=np.linalg.norm(wrist-previous[side]) if previous is not None and np.isfinite(previous[side]).all() else np.inf
            costs.append(distance+.15*min(temporal,.3) if np.isfinite(distance) else temporal+.15)
            matches.append(bool(distance<=.2 or (not body_valid[side] and temporal<=.1)))
        candidate=(sum(costs),sides,matches)
        if best is None or candidate[0]<best[0]:best=candidate
    return list(zip(best[1],best[2]))
