"""Derived animated GLB from a published bundle, and its numerical verification."""
import subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from xms.animation.fk import evaluate
from xms.animation.gltf import load,accessor
from xms.animation.io import read_bundle


def export(bundle,config,fps,out,log,name='XMS_Bundle'):
    script=Path(__file__).resolve().parents[3]/'blender/export_glb.py'
    cmd=[config['blender'],'-b',config['character_asset'],'--python-exit-code','1','--python',str(script),'--','--bundle',str(Path(bundle).resolve()),'--profile',config['character_profile'],'--out',str(Path(out).resolve()),'--fps',str(fps),'--name',name]
    start=time.monotonic()
    with open(log,'w') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'Blender GLB export exit {r.returncode}; see {log}')
    return {'exit_status':0,'seconds':time.monotonic()-start,'command':cmd}


def verify(path,bundle_path,face_map=None,name='XMS_Bundle'):
    """Joint world positions and face weights of the GLB against bundle FK at every sample.

    face_map resolves channel aliases to shape-key (morph target) names."""
    gltf,binary=load(path);bundle=read_bundle(bundle_path);times=bundle.arrays['sample_times_s'];n=len(times)
    animation=[x for x in gltf.get('animations',[]) if x['name']==name]
    if not animation:raise ValueError(f'GLB has no animation {name}')
    nodes=gltf['nodes'];names=[x.get('name') for x in nodes];parent=[-1]*len(nodes)
    for i,x in enumerate(nodes):
        for c in x.get('children',[]):parent[c]=i
    local={}
    for i,x in enumerate(nodes):
        local[i]={'translation':np.tile(x.get('translation',[0,0,0]),(n,1)),'rotation':np.tile(x.get('rotation',[0,0,0,1]),(n,1)),'scale':np.tile(x.get('scale',[1,1,1]),(n,1))}
    morph=None;keys=0
    for channel in animation[0]['channels']:
        sampler=animation[0]['samplers'][channel['sampler']];t=accessor(gltf,binary,sampler['input'])[:,0];v=accessor(gltf,binary,sampler['output']);target=channel['target']
        keys=max(keys,len(t))
        if target['path']=='weights':
            width=len(v)//len(t);morph=(target['node'],np.stack([np.interp(times,t,c) for c in v.reshape(len(t),width).T],axis=1));continue
        if target['path']=='rotation':
            for k in range(1,len(v)):
                if v[k]@v[k-1]<0:v[k]=-v[k]
        resampled=np.stack([np.interp(times,t,c) for c in v.T],axis=1)
        if target['path']=='rotation':resampled/=np.linalg.norm(resampled,axis=1,keepdims=True)
        local[target['node']][target['path']]=resampled
    world={}
    def solve(i):
        if i in world:return world[i]
        m=np.tile(np.eye(4),(n,1,1));m[:,:3,:3]=Rotation.from_quat(local[i]['rotation']).as_matrix()*local[i]['scale'][:,None,:];m[:,:3,3]=local[i]['translation']
        world[i]=m if parent[i]<0 else solve(parent[i])@m;return world[i]
    joints=bundle.metadata['joint_names'];missing=[j for j in joints if j not in names]
    if missing:raise ValueError('GLB lacks bundle joints: '+', '.join(missing[:5]))
    glb=np.stack([solve(names.index(j))[:,:3,3] for j in joints],axis=1);fk=np.stack([evaluate(bundle,i)[:,:3,3] for i in range(n)])
    error=np.linalg.norm(glb-fk,axis=2);result={'animation':name,'samples':n,'glb_keyframes':keys,'duration_s':float(times[-1]-times[0]),'max_joint_position_error_m':float(error.max()),'worst_joint':joints[int(np.unravel_index(error.argmax(),error.shape)[1])]}
    if morph is not None:
        mesh=gltf['meshes'][nodes[morph[0]]['mesh']];targets=mesh.get('extras',{}).get('targetNames',[])
        channels=bundle.metadata['face_channel_names'];key={c:(face_map['channels'][c]['key'] if face_map else c) for c in channels};shared=[c for c in channels if key[c] in targets]
        result['face_channels_matched']=len(shared)
        if shared:result['max_face_weight_error']=float(max(np.abs(morph[1][:,targets.index(key[c])]-bundle.arrays['face_weights'][:,channels.index(c)]).max() for c in shared))
    return result
