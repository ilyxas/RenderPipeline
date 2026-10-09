import numpy as np


def extract(samples,rate,times):
    samples=np.asarray(samples,dtype=float);rms=[]
    for t in times:
        start=max(0,round((t-.015)*rate));end=min(len(samples),round((t+.015)*rate));chunk=samples[start:end]
        rms.append(float(np.sqrt(np.mean(chunk*chunk))) if len(chunk) else 0)
    rms=np.array(rms);scale=max(float(np.percentile(rms,95)),.005)
    return {'envelope':np.clip(rms/scale,0,1),'voicing':rms>.003,'confidence':np.where(rms>.003,.15,0),'policy':'Low-confidence mix energy proxy; not phonemes or singing accuracy.'}
