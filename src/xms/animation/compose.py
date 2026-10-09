"""Resolve exclusive ownership before publication; render only consumes the result."""
import copy
import numpy as np
from .validate import validate


def compose(body,face,head,hands):
    b=copy.deepcopy(body);a=b.arrays
    if b.metadata['ownership']['head']['owner']!='face_with_body_fallback':raise ValueError('Profile does not authorize face/body head compositor')
    a['face_weights']=face['weights'];a['face_validity']=face['validity'];a['face_confidence']=face['confidence'];a['face_provenance']=face['provenance']
    j=head['index'];sources=[]
    if j in hands['indices']:raise ValueError('Conflicting head/hand ownership')
    for i,valid in enumerate(head['face_validity']):
        if valid:
            # Replace the body's local head delta, never multiply both branches.
            a['local_rotation_delta'][i,j]=head['rotations'][i];a['rotation_validity'][i,j]=True;a['rotation_confidence'][i,j]=head['confidence'][i];a['rotation_provenance'][i,j]=1;sources.append('face')
        else:sources.append('body_prior' if a['rotation_validity'][i,j] else 'unobserved')
    for j in hands['indices']:
        mask=hands['validity'][:,j];a['local_rotation_delta'][:,j]=np.where(mask[:,None],hands['rotations'][:,j],[0,0,0,1]);a['rotation_validity'][:,j]=mask;a['rotation_confidence'][:,j]=hands['confidence'][:,j];a['rotation_provenance'][:,j]=mask.astype(np.uint8)
    b.metadata['head_source_by_sample']=sources;b.metadata['compositor']={'version':'1','head':'exclusive face replacement with body fallback','jaw':'face morph only','eyes':'face morph only','hands':'local wrist and primary fingers only','gaps':'No gap interpolation; long gaps are neutral; head may use body prior'}
    return validate(b)
