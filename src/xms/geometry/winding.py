"""Generalized winding numbers (Jacobson et al. 2013) for inside tests on open skin.

A body-skin region cut out of a closed character mesh has holes; ray parity is
undefined there, the winding number degrades smoothly (about 0.5 at a hole). Points
are inside where |w| > 0.5, independent of triangle orientation.
"""
import numpy as np


def winding_numbers(points,vertices,triangles,chunk=512):
    points=np.atleast_2d(np.asarray(points,float));tri=np.asarray(vertices,float)[np.asarray(triangles,int)];out=np.empty(len(points))
    for start in range(0,len(points),chunk):
        p=points[start:start+chunk,None,None,:];a,b,c=(tri[None,:,k,:]-p[:,:,0,:] for k in range(3))
        la,lb,lc=(np.linalg.norm(x,axis=-1) for x in (a,b,c))
        numerator=np.einsum('...i,...i->...',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('...i,...i->...',a,b)*lc+np.einsum('...i,...i->...',b,c)*la+np.einsum('...i,...i->...',c,a)*lb
        out[start:start+chunk]=np.sum(2*np.arctan2(numerator,denominator),axis=1)/(4*np.pi)
    return out


def inside(points,vertices,triangles):
    return np.abs(winding_numbers(points,vertices,triangles))>.5
