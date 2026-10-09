"""Small analytical mesh smoke helpers; large meshes use Blender's BVH."""
import numpy as np


def watertight(triangles):
    triangles=np.asarray(triangles,int);edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True);return bool(len(counts) and np.all(counts==2))


def ray_intersections(origin,direction,vertices,triangles):
    tri=np.asarray(vertices)[np.asarray(triangles)];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0];h=np.cross(direction,e2);det=np.sum(e1*h,axis=1);good=abs(det)>1e-10;inv=np.divide(1,det,out=np.zeros_like(det),where=good);s=origin-tri[:,0];u=np.sum(s*h,axis=1)*inv;q=np.cross(s,e1);v=q@direction*inv;t=np.sum(e2*q,axis=1)*inv
    return np.unique(np.round(t[good&(u>=0)&(v>=0)&(u+v<=1)&(t>1e-8)],9))


def inside(point,vertices,triangles):
    if not watertight(triangles):raise ValueError('Signed interior unavailable for open/nonmanifold mesh')
    direction=np.array([1.,.371,.529]);direction/=np.linalg.norm(direction)
    return len(ray_intersections(np.asarray(point),direction,vertices,triangles))%2==1
