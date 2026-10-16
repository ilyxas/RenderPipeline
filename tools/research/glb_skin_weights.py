"""Skin-weight share per bone class (main / twist / corrective) in an exported animated GLB,
and which rotation channels actually move. Usage: python3 tools/research/glb_skin_weights.py runs/<id>/outputs/animation.glb"""
import json,struct,sys,re,numpy as np
f=open(sys.argv[1],'rb').read()
n=struct.unpack('<I',f[12:16])[0];J=json.loads(f[20:20+n]);off=20+n;bl=struct.unpack('<I',f[off:off+4])[0];B=f[off+8:off+8+bl]
CT={5121:np.uint8,5123:np.uint16,5125:np.uint32,5126:np.float32};NC={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def acc(i):
  a=J['accessors'][i];v=J['bufferViews'][a['bufferView']];dt=np.dtype(CT[a['componentType']]);k=NC[a['type']]
  start=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',dt.itemsize*k)
  raw=np.frombuffer(B,dtype=np.uint8,count=stride*(a['count']-1)+dt.itemsize*k,offset=start)
  out=np.lib.stride_tricks.as_strided(raw,shape=(a['count'],dt.itemsize*k),strides=(stride,1)).copy().view(dt).reshape(a['count'],k)
  if a.get('normalized'):out=out/np.iinfo(dt).max
  return out.astype(float) if dt!=np.float32 or True else out
skin=J['skins'][0];jn=[J['nodes'][j]['name'] for j in skin['joints']]
MAIN=re.compile(r'^(pelvis|spine_0\d|neck_0\d|head|clavicle|upperarm|lowerarm|hand|thigh|calf|foot|ball|(big|index|middle|ring|little)toe_0\d|(thumb|index|middle|ring|pinky)_(0\d|metacarpal))(_[lr])?$')
TW=re.compile(r'_twist_0\d_[lr]$')
cat=np.array([0 if MAIN.match(x) else 1 if TW.search(x) else 2 for x in jn])
names={0:'main',1:'twist',2:'corrective'}
for m in J['meshes']:
  tot=np.zeros(len(jn));nv=0;vc=[]
  for p in m['primitives']:
    at=p['attributes'];W=np.zeros((J['accessors'][at['POSITION']]['count'],len(jn)))
    for s in ('0','1'):
      if 'JOINTS_'+s in at:
        jj=acc(at['JOINTS_'+s]).astype(int);ww=acc(at['WEIGHTS_'+s])
        for c in range(4):np.add.at(W,(np.arange(len(jj)),jj[:,c]),ww[:,c])
    W/=np.maximum(W.sum(1,keepdims=True),1e-9);tot+=W.sum(0);nv+=len(W)
    vc.append(W[:,cat==2].sum(1))
  vc=np.concatenate(vc);share={names[c]:round(float(tot[cat==c].sum()/tot.sum()*100),1) for c in names}
  print(f"{m['name']:22s} verts={nv:6d} share%={share} verts_with_corr>=30%: {int((vc>=.3).sum())} ({(vc>=.3).mean()*100:.1f}%) >=60%: {int((vc>=.6).sum())}")
  if m['name']=='Xandra_Body_Apose3' or m['name']=='body.029':
    top=np.argsort(-tot*(cat==2))[:25];print('   top corrective:',', '.join(f"{jn[i]}={tot[i]:.0f}" for i in top))
    top=np.argsort(-tot*(cat==1))[:12];print('   twist:',', '.join(f"{jn[i]}={tot[i]:.0f}" for i in top))
# animation: which joints animated and do correctives move?
an=J['animations'][0];node2=dict((j,k) for k,j in enumerate(skin['joints']))
moving={0:0,1:0,2:0};static={0:0,1:0,2:0}
for ch in an['channels']:
  t=ch['target'];nd=t['node']
  if nd not in node2 or t['path']!='rotation':continue
  v=acc(an['samplers'][ch['sampler']]['output']);c=cat[node2[nd]]
  if np.abs(v-v[0]).max()>1e-4:moving[c]+=1
  else:static[c]+=1
print('rotation channels moving',moving,'static',static,'counts',{names[c]:int((cat==c).sum()) for c in names})
print('---')
for m in J['meshes']:
  if m['name']!='Xandra_Body_Apose3':continue
  tot=np.zeros(len(jn))
  for p in m['primitives']:
    at=p['attributes'];W=np.zeros((J['accessors'][at['POSITION']]['count'],len(jn)))
    for s in ('0','1'):
      jj=acc(at['JOINTS_'+s]).astype(int);ww=acc(at['WEIGHTS_'+s])
      for c in range(4):np.add.at(W,(np.arange(len(jj)),jj[:,c]),ww[:,c])
    W/=np.maximum(W.sum(1,keepdims=True),1e-9);tot+=W.sum(0)
  for k in ['upperarm_l','upperarm_twist_01_l','upperarm_twist_02_l','upperarm_twistCor_01_l','upperarm_twistCor_02_l','lowerarm_l','lowerarm_twist_01_l','lowerarm_twist_02_l','thigh_l','thigh_twist_01_l','thigh_twist_02_l','thigh_twistCor_01_l','calf_l','calf_twist_01_l','calf_twist_02_l','calf_knee_l','calf_kneeBack_l','hand_l','clavicle_l','clavicle_pec_l','spine_05','spine_04']:
    print(f'{k:24s}{tot[jn.index(k)]:8.0f}')
