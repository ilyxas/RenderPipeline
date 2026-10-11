"""Export-only process: bundle + character -> animated GLB (skinned meshes, bones, face morphs).

Adapted from clip_full scripts/b_export.py (verified exporter settings). The session
never saves the canonical asset; mesh edits below exist only in this process.
"""
import sys,json,argparse,hashlib,struct,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy,bmesh
import numpy as np
from fractions import Fraction
from xms.animation.io import read_bundle,bundle_hash,file_hash
from xms.render.blender_adapter import apply_bundle

EXPORT=['F4_Head','F4_Torso','F4_Legs_Nude.001','F4_Hands','F4_Feet','Hair 2_Back.001','Yuki_Bangs','Earrings_CircleDrop','Xandra_Top','Xandra_Shorts']
ARKIT_KEYS=52
p=argparse.ArgumentParser()
for n in ('bundle','profile','out','fps','name'):p.add_argument('--'+n,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out)
if out.exists():raise ValueError('GLB output already exists: '+str(out))
def blocked(*args,**kwargs):raise RuntimeError('Saving the canonical asset is blocked in the export session')
bpy.app.handlers.save_pre.append(blocked)
started=time.monotonic();fps=Fraction(a.fps);times=read_bundle(a.bundle).arrays['sample_times_s']
b,profile,face=apply_bundle(a.bundle,a.profile,times,fps=float(fps))
arm=bpy.data.objects[profile['armature']];head=bpy.data.objects['F4_Head'];log={'source_asset':bpy.data.filepath}

# Face: keep the ARKit block the bundle drives; the exporter needs one morph target list.
keys=head.data.shape_keys.key_blocks;kept={k.name for k in keys[1:ARKIT_KEYS+1]}
missing=[s['key'] for s in face['channels'].values() if s['object']=='F4_Head' and s['key'] not in kept]
if missing:raise ValueError('Bundle face channels outside the exported ARKit block: '+', '.join(missing))
for k in list(keys)[ARKIT_KEYS+1:]:head.shape_key_remove(k)
head.active_shape_key_index=0;head.show_only_shape_key=False
# Other meshes: bake the rendered shape-key mix into the mesh, then drop the keys.
baked={}
for n in EXPORT:
    o=bpy.data.objects[n];sk=o.data.shape_keys
    if n=='F4_Head' or not sk:continue
    nonzero=[(k.name,k.value) for k in sk.key_blocks[1:] if k.value!=0 and not k.mute];mix=None
    if nonzero:
        tmp=o.shape_key_add(name='__mix',from_mix=True);mix=np.zeros(len(o.data.vertices)*3);tmp.data.foreach_get('co',mix)
    o.shape_key_clear()
    if mix is not None:o.data.vertices.foreach_set('co',mix);o.data.update()
    baked[n]=nonzero
log['baked_shape_keys']=baked
# A zero 'Attribute' colour layer exports as black skin.
for n in EXPORT:
    me=bpy.data.objects[n].data
    for name in [c.name for c in me.color_attributes]:me.color_attributes.remove(me.color_attributes[name])
# Clothes_Hidden is a render-time mask; glTF has no modifiers, so delete those vertices here.
hidden={}
for n in ('F4_Torso','F4_Legs_Nude.001','F4_Head'):
    o=bpy.data.objects[n];group=o.vertex_groups.get('Clothes_Hidden')
    if not group:continue
    for m in list(o.modifiers):
        if m.type=='MASK' and m.vertex_group=='Clothes_Hidden':o.modifiers.remove(m)
    weights={v.index:g.weight for v in o.data.vertices for g in v.groups if g.group==group.index}
    bm=bmesh.new();bm.from_mesh(o.data);bm.verts.ensure_lookup_table();kill=[v for v in bm.verts if weights.get(v.index,0)>.5]
    bmesh.ops.delete(bm,geom=kill,context='VERTS');bm.to_mesh(o.data);bm.free();o.data.update();hidden[n]=len(kill)
log['clothes_hidden_deleted']=hidden

# One glTF animation: NLA_TRACKS mode merges the armature and face tracks of the same name.
for o in bpy.data.objects:
    if o.animation_data and o!=arm:
        o.animation_data.action=None
        for t in list(o.animation_data.nla_tracks):o.animation_data.nla_tracks.remove(t)
body=arm.animation_data.action;face_action=head.data.shape_keys.animation_data.action
for action in list(bpy.data.actions):
    if action not in (body,face_action):bpy.data.actions.remove(action)
body.name=a.name;face_action.name=a.name+'_Face'
for owner,action in ((arm,body),(head.data.shape_keys,face_action)):
    data=owner.animation_data;data.action=None
    for t in list(data.nla_tracks):data.nla_tracks.remove(t)
    track=data.nla_tracks.new();track.name=a.name;track.strips.new(a.name,int(action.frame_range[0]),action);data.use_nla=True
for k in head.data.shape_keys.key_blocks:k.value=0.0
sc=bpy.context.scene;sc.render.fps=round(float(fps));sc.render.fps_base=sc.render.fps/float(fps)
for bone in arm.pose.bones:bone.matrix_basis.identity()

for o in bpy.data.objects:
    try:o.select_set(False)
    except RuntimeError:pass
for n in [arm.name]+EXPORT:
    o=bpy.data.objects[n];o.hide_viewport=False;o.hide_render=False
    try:o.hide_set(False)
    except RuntimeError:pass
    o.select_set(True)
bpy.context.view_layer.objects.active=arm
settings=dict(filepath=str(out),export_format='GLB',use_selection=True,
    export_yup=True,export_apply=False,export_texcoords=True,export_normals=True,export_tangents=False,
    export_materials='EXPORT',export_image_format='AUTO',
    export_vertex_color='NONE',export_all_vertex_colors=False,export_active_vertex_color_when_no_material=False,
    export_cameras=False,export_lights=False,export_extras=False,
    export_skins=True,export_influence_nb=4,export_all_influences=False,export_def_bones=False,
    export_rest_position_armature=True,
    export_morph=True,export_morph_normal=True,export_morph_tangent=False,export_morph_animation=True,
    export_animations=True,export_animation_mode='NLA_TRACKS',export_nla_strips=True,
    export_frame_range=False,export_force_sampling=True,export_frame_step=1,
    export_anim_slide_to_zero=True,export_reset_pose_bones=True,export_anim_single_armature=True,
    export_optimize_animation_size=True,export_optimize_animation_keep_anim_armature=True,
    export_bake_animation=False,export_current_frame=False)
bpy.ops.export_scene.gltf(**settings)

# Post-patch: profile hair tint (Blender look is node-based) and neutral rest morph weights.
look=json.loads((Path(a.profile)/'look.json').read_text());raw=out.read_bytes();size=struct.unpack('<I',raw[12:16])[0];gltf=json.loads(raw[20:20+size]);rest=raw[20+size:]
for m in gltf['materials']:
    if m['name']==look['hair']['material']:
        pbr=m.setdefault('pbrMetallicRoughness',{});pbr['baseColorFactor']=look['hair']['tint'];pbr['metallicFactor']=0.0;pbr.setdefault('roughnessFactor',.42)
        m['alphaMode']='MASK';m['alphaCutoff']=.4;m['doubleSided']=True
for mesh in gltf['meshes']:
    if 'weights' in mesh:mesh['weights']=[0]*len(mesh['weights'])
text=json.dumps(gltf,separators=(',',':')).encode();text+=b' '*((4-len(text)%4)%4)
out.write_bytes(struct.pack('<III',0x46546C67,2,12+8+len(text)+len(rest))+struct.pack('<II',len(text),0x4E4F534A)+text+rest)
log.update(schema_version='xms.glb_export.v1',status='passed',bundle_hash=bundle_hash(a.bundle),character_profile_hash=profile['profile_hash'],character_asset_sha256_after=file_hash(bpy.data.filepath),
           blender_version=bpy.app.version_string,animation=a.name,fps=str(fps),samples=len(times),meshes=EXPORT,exporter_settings=settings,seconds=time.monotonic()-started,glb_sha256=file_hash(out),bytes=out.stat().st_size)
out.with_suffix('.export.json').write_text(json.dumps(log,indent=2,default=str))
print('XMS_GLB',out,log['bytes'],flush=True)
