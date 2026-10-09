import copy
import numpy as np


def fuse(face,audio,names):
    result=copy.deepcopy(face);n=len(result['weights']);targets={'jawOpen':audio['envelope']*.55};conf=audio['confidence'].copy()
    for cue in audio.get('cues',[]):
        mask=(audio['times']>=cue['start'])&(audio['times']<cue['end']);shape=cue['value'];conf[mask]=.6
        for name in ('jawOpen','mouthClose','mouthPucker','mouthFunnel'):
            targets.setdefault(name,np.zeros(n))
            value={'jawOpen':{'C':.5,'D':.7,'H':.3},'mouthClose':{'A':1,'X':.8},'mouthPucker':{'F':.8},'mouthFunnel':{'E':.7}}[name].get(shape,0)
            targets[name][mask]=value
    for name,target in targets.items():
        if name not in names:continue
        j=names.index(name);vc=result['confidence'][:,j];ac=conf*(1-vc)**2
        denom=vc+ac;mask=ac>0;result['weights'][mask,j]=(result['weights'][mask,j]*vc[mask]+target[mask]*ac[mask])/np.maximum(denom[mask],1e-9)
        inferred=mask&~result['validity'][:,j];result['validity'][inferred,j]=True;result['confidence'][inferred,j]=ac[inferred];result['provenance'][inferred,j]=2
    w=result['weights']
    if 'mouthClose' in names and 'jawOpen' in names:w[:,names.index('jawOpen')]*=1-w[:,names.index('mouthClose')]
    for group in (('mouthPucker','mouthStretchLeft','mouthStretchRight'),('jawLeft','jawRight')):
        ids=[names.index(x) for x in group if x in names]
        if ids:w[:,ids]/=np.maximum(1,w[:,ids].sum(axis=1))[:,None]
    result['lip_diagnostics']={'policy':'Good video dominates; mixed audio energy is low-confidence only.','audio_warnings':audio.get('metadata',{}).get('warnings',[]),'quality_accepted':False,'alignment_correction_s':0}
    return result
