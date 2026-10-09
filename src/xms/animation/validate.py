import re
import numpy as np
from .bundle import VERSION, PROVENANCE


def validate(b, profile_hash=None):
    m,a=b.metadata,b.arrays
    def require(ok, message):
        if not ok: raise ValueError(message)
    require(m.get('schema_version') == VERSION, 'Unsupported AnimationBundle version; explicit migration required')
    require(m.get('units')=='metres' and m.get('coordinate_system')=='RH_Y_UP_Z_FORWARD' and m.get('quaternion_order')=='XYZW' and m.get('rotation_semantics')=='rest_local_times_delta', 'Coordinate/rotation contract mismatch')
    require(bool(re.fullmatch('[0-9a-f]{64}',m.get('character_profile_hash',''))),'Invalid profile hash')
    if profile_hash is not None: require(m['character_profile_hash']==profile_hash,'Profile mismatch')
    for key in ('joint_names','face_channel_names'):
        names=m[key]; require(isinstance(names,list) and all(isinstance(x,str) and x for x in names) and len(set(names))==len(names),'Nonunique/invalid names')
    j,f=len(m['joint_names']),len(m['face_channel_names'])
    require(j>0,'Empty rig')
    n=len(a['sample_times_s']); require(n>0,'Empty timeline')
    shapes={'sample_times_s':(n,), 'source_time_map':(n,), 'parent_indices':(j,), 'rest_translation':(j,3), 'rest_rotation':(j,4), 'rest_scale':(j,3), 'armature_transform':(4,4), 'local_rotation_delta':(n,j,4), 'root_translation':(n,3), 'face_weights':(n,f)}
    for group,shape in [('rotation',(n,j)),('root',(n,)),('face',(n,f))]:
        for suffix in ('validity','confidence','provenance'): shapes[group+'_'+suffix]=shape
    require(set(a)==set(shapes),'Unexpected/missing arrays')
    for key,shape in shapes.items():
        v=a[key]; require(isinstance(v,np.ndarray) and v.shape==shape,'Shape mismatch: '+key)
        dtype='b' if key.endswith('validity') else 'iu' if key.endswith('provenance') or key=='parent_indices' else 'f'
        require(v.dtype.kind in dtype,'Dtype mismatch: '+key)
        require(np.all(np.isfinite(v)),'Nonfinite values: '+key)
    require(np.all(np.diff(a['sample_times_s'])>0) and a['sample_times_s'][0]>=0,'Nonmonotonic sample times')
    require(np.all(np.diff(a['source_time_map'])>=0),'Nonmonotonic source times')
    parents=a['parent_indices']; require(np.all((parents>=-1)&(parents<np.arange(j))),'Malformed parent hierarchy (parent must precede child)')
    root=m['root_carrier_index']; require(type(root) is int and 0<=root<j and parents[root]==-1,'Invalid root carrier')
    require(np.all(a['rest_scale']>0),'Invalid rest scale')
    require(np.allclose(a['armature_transform'][3],[0,0,0,1]) and abs(np.linalg.det(a['armature_transform'][:3,:3]))>1e-12,'Invalid armature transform')
    for key in ('rest_rotation','local_rotation_delta'):
        require(np.max(np.abs(np.linalg.norm(a[key],axis=-1)-1))<=1e-4,'Nonunit quaternion')
    ranges=np.asarray(m['face_ranges'],dtype=float).reshape(f,2)
    require(np.all(np.isfinite(ranges)) and np.all(ranges[:,0]<=0) and np.all(ranges[:,1]>=0),'Invalid face ranges')
    require(np.all(a['face_weights']>=ranges[:,0]) and np.all(a['face_weights']<=ranges[:,1]),'Face range exceeded')
    require(m['provenance_codes']==PROVENANCE,'Unknown provenance codes')
    for group in ('rotation','root','face'):
        valid,conf,prov=(a[group+'_'+s] for s in ('validity','confidence','provenance'))
        require(np.all((conf>=0)&(conf<=1)) and np.all(np.isin(prov,list(PROVENANCE.values()))),'Invalid confidence/provenance')
        require(np.all(conf[~valid]==0) and np.all(prov[~valid]==0) and np.all(prov[valid]!=0),'Invalid unobserved masks')
        value=a[{'rotation':'local_rotation_delta','root':'root_translation','face':'face_weights'}[group]][~valid]
        require(np.allclose(value,[0,0,0,1] if group=='rotation' else 0),'Unobserved channels must be neutral')
    ownership=m['ownership']
    require(set(ownership)=={'head','eyes','jaw'},'Missing ownership roles')
    for role,spec in ownership.items():
        require(isinstance(spec,dict) and set(spec)=={'owner','joints','faces'} and isinstance(spec['owner'],str),'Invalid ownership declaration')
        require(not (spec['joints'] and spec['faces']),'Conflicting bone/morph owners: '+role)
        require(set(spec['joints'])<=set(m['joint_names']) and set(spec['faces'])<=set(m['face_channel_names']),'Unknown owned channel')
    owned=[x for s in ownership.values() for x in s['joints']+s['faces']]
    require(len(owned)==len(set(owned)),'Conflicting owners')
    # Explicit v1 structures; early stages publish empty lists.
    for key,fields in [('contacts',{'start_s','end_s','joint','kind','confidence'}),('events',{'time_s','kind','channels'})]:
        require(isinstance(m[key],list),'Invalid '+key)
        for item in m[key]:
            require(isinstance(item,dict) and set(item)==fields,'Invalid '+key+' record')
            if key=='contacts':
                require(0<=item['start_s']<=item['end_s']<=a['sample_times_s'][-1] and item['joint'] in m['joint_names'] and item['kind'] in ('foot','palm') and 0<=item['confidence']<=1,'Invalid contact')
            else:
                require(0<=item['time_s']<=a['sample_times_s'][-1] and item['kind'] in ('jump','tracking_loss') and set(item['channels'])<=set(m['joint_names']+m['face_channel_names']),'Invalid event')
    return b
