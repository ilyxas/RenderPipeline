"""Read canonical asset in Blender; write only versioned registration data."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy
import numpy as np
from xms.animation.fk import decompose
from xms.animation.io import file_hash
from xms.profiles.character import profile_digest

C=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scene',required=True);p.add_argument('--scene-out',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
arm=bpy.data.objects['root']
# The v1 fixture supports the measured standard inheritance, never guesses at constraints.
if any(b.constraints for b in arm.pose.bones): raise RuntimeError('Registration blocked: pose constraints require an explicit contract')
if arm.animation_data and arm.animation_data.drivers: raise RuntimeError('Registration blocked: armature drivers require an explicit contract')
bones=list(arm.data.bones)
bones.sort(key=lambda b:len(b.parent_recursive))
names=[b.name for b in bones]; parents=np.array([names.index(b.parent.name) if b.parent else -1 for b in bones],dtype=np.int32)
T=[];Q=[];S=[]
for b in bones:
    if b.inherit_scale!='FULL' or not b.use_inherit_rotation: raise RuntimeError('Unsupported bone inheritance: '+b.name)
    local=np.array(b.matrix_local) if not b.parent else np.linalg.inv(np.array(b.parent.matrix_local))@np.array(b.matrix_local)
    t,q,s=decompose(C@local@C.T);T.append(t);Q.append(q);S.append(s)
rig={'parent_indices':parents,'rest_translation':np.array(T),'rest_rotation':np.array(Q),'rest_scale':np.array(S),'armature_transform':C@np.array(arm.matrix_world)@C.T}
np.savez_compressed(out/'rig.npz',**rig)
keys=bpy.data.objects['F4_Head'].data.shape_keys.key_blocks
mapping={}
for name in keys.keys():
    if name.startswith(('eye','jaw','mouth','brow','cheek','nose')) and not name.endswith('old'):
        mapping[name]={'object':'F4_Head','key':name,'neutral':0.0,'range':[max(0.,keys[name].slider_min),min(1.,keys[name].slider_max)]}
if 'mouthSmileRight' not in mapping:
    mapping['mouthSmileRight']={'object':'F4_Head','key':'mouthSmileRightold','neutral':0.0,'range':[0.,1.]}
face={'schema_version':'xms.face_map.v1','channels':mapping,'alias_policy':'one semantic channel per key; old duplicates excluded except right smile alias'}
(out/'face_map.json').write_text(json.dumps(face,indent=2))
look={'schema_version':'xms.look.v1','id':'xandra_kit_natural_v1','reference':'clip02 scripts/b_render.py material constants only','hair':{'material':'MI_X3D_Hair_V7','node':'Principled_glTF','tint':[.1697,.1006,.0921,1.]},'skin':{'materials':['MI_X3D_Skin_Face_F4','MI_X3D_Skin_Body_F4','MI_X3D_Skin_Body_Nude_F4'],'specular':.32,'roughness':.58},'teeth':{'variant':'natural','geometry_offsets_m':[0,0,0],'note':'Canonical teeth retained. Legacy soft_lips tuck is a separate unverified variant, not enabled.'}}
(out/'look.json').write_text(json.dumps(look,indent=2))
roles={n:n for n in ['pelvis','spine_01','spine_05','head','neck_01','neck_02']+[x+'_'+s for x in ['upperarm','lowerarm','hand','thigh','calf','foot','ball'] for s in ['l','r']]}
profile={'schema_version':'xms.character.v1','id':'xandra','asset_sha256':file_hash(bpy.data.filepath),'armature':'root','joint_names':names,'face_channel_names':list(mapping),'face_ranges':[v['range'] for v in mapping.values()],'root_carrier_index':names.index('pelvis'),'root_policy':'world_translation_once_at_pelvis','ownership':{'head':{'owner':'body','joints':['head'],'faces':[]},'eyes':{'owner':'morph','joints':[],'faces':[n for n in mapping if n.startswith('eyeLook')]},'jaw':{'owner':'morph','joints':[],'faces':[n for n in mapping if n.startswith('jaw')]}},'roles':roles,'bone_axes':'rig rest TRS basis; all deltas in canonical conjugated bone-local basis','inheritance':{b.name:{'scale':b.inherit_scale,'rotation':b.use_inherit_rotation} for b in bones},'contacts':[],'files':{n:file_hash(out/n) for n in ['rig.npz','face_map.json','look.json']}}
profile['profile_hash']=profile_digest(profile);(out/'character.json').write_text(json.dumps(profile,indent=2))
scene={'schema_version':'xms.scene.v1','id':'bedroom','asset_sha256':file_hash(a.scene),'collection':'Bedroom_Clip02','world':'Clip02_World','camera':{'position_blender_m':[0,-3.8,.88],'rotation_euler_degrees':[90,0,0],'vertical_fov_degrees':36},'floor_transform':np.eye(4).tolist(),'resolution':[540,960],'samples':16,'exposure':-.2,'disable_character_objects':['Xa_Floor','stage'],'disable_character_lights':True}
scene['profile_hash']=profile_digest(scene);Path(a.scene_out).write_text(json.dumps(scene,indent=2))
print('REGISTERED',len(names),'bones',len(mapping),'face channels',profile['profile_hash'])
