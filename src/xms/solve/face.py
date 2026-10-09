"""Video-only channel policies with bounded interpolation and explicit uncertainty."""
import numpy as np
from .baseline_face import solve_face as baseline
from .contacts import spans


def fill_short(values,valid,confidence,times,max_gap=.15):
    values=values.copy();valid=valid.copy();confidence=confidence.copy();estimated=np.zeros_like(valid)
    for j in range(values.shape[1]):
        for start,end in spans(~valid[:,j],1):
            if start==0 or end==len(times) or times[end]-times[start-1]>max_gap:continue
            alpha=(times[start:end]-times[start-1])/(times[end]-times[start-1])
            values[start:end,j]=values[start-1,j]*(1-alpha)+values[end,j]*alpha
            confidence[start:end,j]=min(confidence[start-1,j],confidence[end,j])*.5;valid[start:end,j]=True;estimated[start:end,j]=True
    return values,valid,confidence,estimated


def solve_face(observations,timeline,profile,face_map):
    result=baseline(observations,timeline,profile,face_map);m,a=observations;idx=np.array(timeline['output_observation_indices']);times=np.array(timeline['output_times_s']);names=profile['face_channel_names'];offsets={}
    for j,name in enumerate(names):
        if name.startswith(('brow','cheek','nose')):
            reliable=result['validity'][:,j]&(result['confidence'][:,j]>=.5)
            if reliable.sum()>=8:
                offset=min(.08,float(np.percentile(result['weights'][reliable,j],10)));result['weights'][:,j]=np.maximum(0,result['weights'][:,j]-offset);offsets[name]=offset
        # Blink and mouth closure/opening retain event sample timing.
        if name.startswith(('brow','cheek','nose','mouthSmile','mouthFrown')):
            for i in range(1,len(times)):
                if result['validity'][i-1:i+1,j].all():
                    alpha=1-np.exp(-(times[i]-times[i-1])/.04);result['weights'][i,j]=alpha*result['weights'][i,j]+(1-alpha)*result['weights'][i-1,j]
    w,v,c,estimated=fill_short(result['weights'],result['validity'],result['confidence'],times)
    result.update(weights=w,validity=v,confidence=c,provenance=np.where(estimated,2,v.astype(np.uint8)).astype(np.uint8),diagnostics={'neutral_offsets':offsets,'gap_estimated_samples':int(estimated.sum()),'observed_frame_fraction':float(a['validity'][idx].mean()),'limitations':['Neutral offsets are bounded lower-envelope estimates, not known neutral expression.','Long gaps stay neutral; gaze uses head-local blendshape channels.']})
    return result
