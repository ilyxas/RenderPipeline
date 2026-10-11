"""Minimal GLB reading: JSON/binary chunks, accessors, node hierarchy and sampled animation."""
import json,struct
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

TYPES={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16};COMPONENTS={5126:np.float32,5125:np.uint32,5123:np.uint16,5122:np.int16,5121:np.uint8,5120:np.int8}
# glTF normalized integers (quantized animation): signed types clamp at -1.
NORMALIZED={5123:65535.,5122:32767.,5121:255.,5120:127.}


def load(path):
    raw=Path(path).read_bytes();size=struct.unpack('<I',raw[12:16])[0];gltf=json.loads(raw[20:20+size]);offset=20+size
    length=struct.unpack('<I',raw[offset:offset+4])[0];return gltf,raw[offset+8:offset+8+length]


def accessor(gltf,binary,index):
    a=gltf['accessors'][index];view=gltf['bufferViews'][a['bufferView']];width=TYPES[a['type']];dtype=np.dtype(COMPONENTS[a['componentType']])
    start=view.get('byteOffset',0)+a.get('byteOffset',0);stride=view.get('byteStride',dtype.itemsize*width)
    values=np.ndarray((a['count'],width),dtype=dtype,buffer=binary,offset=start,strides=(stride,dtype.itemsize)).astype(float)
    if a.get('normalized'):values=np.maximum(values/NORMALIZED[a['componentType']],-1.)
    return values


class Skeleton:
    """Node rest TRS, parents and one animation resampled onto arbitrary times (held at the ends)."""
    def __init__(self,path,animation=0):
        self.gltf,binary=load(path);nodes=self.gltf['nodes'];self.names=[x.get('name','') for x in nodes];self.parent=[-1]*len(nodes)
        for i,x in enumerate(nodes):
            for c in x.get('children',[]):self.parent[c]=i
        self.translation=np.array([x.get('translation',[0,0,0]) for x in nodes],float);self.rotation=np.array([x.get('rotation',[0,0,0,1]) for x in nodes],float)
        if any('scale' in x and not np.allclose(x['scale'],1) for x in nodes):raise ValueError('Scaled skeleton nodes are not supported')
        self.channels={};self.duration=0.
        if self.gltf.get('animations'):
            anim=self.gltf['animations'][animation]
            for c in anim['channels']:
                s=anim['samplers'][c['sampler']];t=accessor(self.gltf,binary,s['input'])[:,0];v=accessor(self.gltf,binary,s['output'])
                if s.get('interpolation','LINEAR')!='LINEAR':raise ValueError('Only LINEAR glTF animation is supported')
                self.channels[(c['target']['node'],c['target']['path'])]=(t,v);self.duration=max(self.duration,float(t[-1]))
        order=[];stack=[i for i,p in enumerate(self.parent) if p<0]
        while stack:i=stack.pop(0);order.append(i);stack+=nodes[i].get('children',[])
        self.order=order

    def local(self,times=None):
        """Local rotations (n,N,4) and translations (n,N,3); times=None gives the rest (node) pose."""
        n=1 if times is None else len(times);q=np.tile(self.rotation,(n,1,1));p=np.tile(self.translation,(n,1,1))
        if times is None:return q,p
        clipped=np.clip(times,0,self.duration)
        for (i,path),(t,v) in self.channels.items():
            if path=='rotation':
                q[:,i]=v[0] if len(t)==1 else Slerp(t,Rotation.from_quat(v))(np.clip(clipped,t[0],t[-1])).as_quat()
            elif path=='translation':
                p[:,i]=v[0] if len(t)==1 else np.stack([np.interp(clipped,t,v[:,k]) for k in range(3)],axis=1)
        return q,p

    def world(self,times=None):
        """World rotation matrices (n,N,3,3) and positions (n,N,3)."""
        q,p=self.local(times);n=len(q);rotations=np.zeros((n,len(self.names),3,3));positions=np.zeros((n,len(self.names),3))
        for i in self.order:
            r=Rotation.from_quat(q[:,i]).as_matrix();parent=self.parent[i]
            if parent<0:rotations[:,i]=r;positions[:,i]=p[:,i]
            else:rotations[:,i]=rotations[:,parent]@r;positions[:,i]=positions[:,parent]+np.einsum('nij,nj->ni',rotations[:,parent],p[:,i])
        return rotations,positions
