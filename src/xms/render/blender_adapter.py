"""Blender-only consumer: evaluate approved bundle, convert, key. No solve/tracking."""
import json, math
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle, file_hash
from xms.animation.sample import sample
from xms.animation.fk import evaluate
from xms.profiles.character import load_character

C=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])


def clear_legacy():
    import bpy
    for obj in bpy.data.objects:
        obj.animation_data_clear()
        if obj.type=='MESH' and obj.data.shape_keys:
            obj.data.shape_keys.animation_data_clear()
    # No rest/identity geometry keys are reset; expression channels are explicitly keyed below.


def apply_bundle(path,profile_path,times=None,fps=24):
    import bpy
    from mathutils import Matrix
    p,rig,face=load_character(profile_path)
    if file_hash(bpy.data.filepath)!=p['asset_sha256']: raise ValueError('Canonical asset hash mismatch')
    b=read_bundle(path,p['profile_hash'])
    for k,v in rig.items():
        if not np.array_equal(v,b.arrays[k]): raise ValueError('Bundle rig differs from registered profile: '+k)
    if b.metadata['joint_names']!=p['joint_names'] or b.metadata['face_channel_names']!=p['face_channel_names'] or b.metadata['ownership']!=p['ownership']: raise ValueError('Bundle channel/ownership profile mismatch')
    sampled=sample(b,b.arrays['sample_times_s'] if times is None else times)
    clear_legacy();arm=bpy.data.objects[p['armature']]
    arm.animation_data_create();arm.animation_data.action=bpy.data.actions.new('XMS_Bundle')
    inverse=np.linalg.inv(np.array(arm.matrix_world))
    # Discard all legacy expression values, including aliases excluded from the profile.
    head=bpy.data.objects['F4_Head'].data.shape_keys
    for key in head.key_blocks:
        if key.name.startswith(('eye','jaw','mouth','brow','cheek','nose')): key.value=0
    for i,time_s in enumerate(sampled.arrays['sample_times_s']):
        desired=evaluate(sampled,i);pose=np.array([inverse@C.T@w@C for w in desired])
        frame=1+float(time_s)*fps
        for j,name in enumerate(p['joint_names']):
            bone=arm.pose.bones[name];parent=int(rig['parent_indices'][j])
            kw={} if parent<0 else {'parent_matrix':Matrix(pose[parent]),'parent_matrix_local':bone.parent.bone.matrix_local}
            basis=bone.bone.convert_local_to_pose(Matrix(pose[j]),bone.bone.matrix_local,invert=True,**kw)
            bone.rotation_mode='QUATERNION';bone.matrix_basis=basis
            for field in ('location','rotation_quaternion','scale'): bone.keyframe_insert(field,frame=frame,group=name)
        for j,name in enumerate(p['face_channel_names']):
            spec=face['channels'][name];key=bpy.data.objects[spec['object']].data.shape_keys.key_blocks[spec['key']]
            key.value=float(sampled.arrays['face_weights'][i,j]);key.keyframe_insert('value',frame=frame)
    return sampled,p,face


def apply_look(profile_path):
    import bpy
    look=json.loads((Path(profile_path)/'look.json').read_text())
    h=look['hair'];m=bpy.data.materials[h['material']];tree=m.node_tree;bsdf=tree.nodes[h['node']]
    if 'XMS_HairTint_v1' not in tree.nodes:
        link=bsdf.inputs['Base Color'].links[0];source=link.from_socket
        node=tree.nodes.new('ShaderNodeMixRGB');node.name='XMS_HairTint_v1';node.blend_type='MULTIPLY';node.inputs[0].default_value=1;node.inputs[2].default_value=h['tint']
        tree.links.remove(link);tree.links.new(source,node.inputs[1]);tree.links.new(node.outputs[0],bsdf.inputs['Base Color'])
    for name in look['skin']['materials']:
        if name in bpy.data.materials:
            b=bpy.data.materials[name].node_tree.nodes['Principled_glTF'];b.inputs['Specular IOR Level'].default_value=look['skin']['specular'];b.inputs['Roughness'].default_value=look['skin']['roughness']


def setup_scene(scene_path,scene_profile,character_profile):
    import bpy
    if file_hash(scene_path)!=scene_profile['asset_sha256']: raise ValueError('Scene asset mismatch')
    sc=bpy.context.scene
    if scene_profile['disable_character_lights']:
        for o in bpy.data.objects:
            if o.type=='LIGHT': o.hide_render=True
    for name in scene_profile['disable_character_objects']:
        if name in bpy.data.objects: bpy.data.objects[name].hide_render=True
    with bpy.data.libraries.load(str(scene_path),link=False) as (source,dest):
        dest.collections=[scene_profile['collection']];dest.worlds=[scene_profile['world']]
    sc.collection.children.link(dest.collections[0]);sc.world=dest.worlds[0]
    apply_look(character_profile)
    camera=bpy.data.objects['Xa_Cam'];c=scene_profile['camera'];camera.location=c['position_blender_m'];camera.rotation_euler=[math.radians(x) for x in c['rotation_euler_degrees']]
    camera.data.sensor_fit='VERTICAL';camera.data.angle_y=math.radians(c['vertical_fov_degrees']);camera.data.shift_x=0;camera.data.shift_y=0;sc.camera=camera
    sc.render.engine='BLENDER_EEVEE_NEXT';sc.eevee.taa_render_samples=scene_profile['samples']
    sc.render.resolution_x,sc.render.resolution_y=scene_profile['resolution'];sc.render.resolution_percentage=100
    sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast';sc.view_settings.exposure=scene_profile['exposure'];sc.render.image_settings.file_format='PNG'
    return sc
