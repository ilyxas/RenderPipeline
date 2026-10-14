"""Blender-only consumer: evaluate approved bundle, convert, key. No solve/tracking."""
import json, math
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle, file_hash
from xms.animation.sample import sample
from xms.animation.fk import evaluate
from xms.profiles.character import load_character

C=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])


# Asset drivers that switch outfits (visibility, masks) are part of the authored look;
# everything else animated in the asset is legacy and would fight the bundle.
KEPT_DRIVER_PREFIXES=('hide_render','hide_viewport','modifiers[')


def keep_driver(owner_kind,data_path):
    return owner_kind=='OBJECT' and data_path.startswith(KEPT_DRIVER_PREFIXES)


def clear_animation(data,owner_kind):
    """Drop actions and NLA; drop drivers except kept object visibility/modifier drivers. Returns (kept,removed)."""
    if not data:return 0,0
    data.action=None
    for track in list(data.nla_tracks):data.nla_tracks.remove(track)
    kept=removed=0
    for driver in list(data.drivers):
        if keep_driver(owner_kind,driver.data_path):kept+=1
        else:data.drivers.remove(driver);removed+=1
    return kept,removed


def clear_legacy(objects=None):
    if objects is None:
        import bpy;objects=bpy.data.objects
    record={'kept_drivers':0,'removed_drivers':0,'kept_paths':set()}
    for obj in objects:
        owners=[(obj.animation_data,'OBJECT')]
        if obj.type=='MESH' and obj.data.shape_keys:owners.append((obj.data.shape_keys.animation_data,'SHAPE_KEYS'))
        for data,kind in owners:
            if data and kind=='OBJECT':record['kept_paths'].update(d.data_path for d in data.drivers if keep_driver(kind,d.data_path))
            kept,removed=clear_animation(data,kind);record['kept_drivers']+=kept;record['removed_drivers']+=removed
    record['kept_paths']=sorted(record['kept_paths'])
    # No rest/identity geometry keys are reset; expression channels are explicitly keyed below.
    return record


def driver_inputs(objects=None):
    """Custom-property values read by kept drivers, e.g. {'root["Sport_Set"]': 1}."""
    if objects is None:
        import bpy;objects=bpy.data.objects
    inputs={}
    for obj in objects:
        data=obj.animation_data
        for driver in (data.drivers if data else []):
            for variable in driver.driver.variables:
                for target in variable.targets:
                    if target.id is not None and target.data_path.startswith('['):
                        try:inputs[target.id.name+target.data_path]=target.id.path_resolve(target.data_path)
                        except (ValueError,AttributeError):inputs[target.id.name+target.data_path]=None
    return {k:(v if isinstance(v,(bool,int,float,str)) or v is None else str(v)) for k,v in sorted(inputs.items())}


def apply_bundle(path,profile_path,times=None,fps=24,record=None):
    import bpy
    from mathutils import Matrix
    p,rig,face=load_character(profile_path)
    if file_hash(bpy.data.filepath)!=p['asset_sha256']: raise ValueError('Canonical asset hash mismatch')
    b=read_bundle(path,p['profile_hash'])
    for k,v in rig.items():
        if not np.array_equal(v,b.arrays[k]): raise ValueError('Bundle rig differs from registered profile: '+k)
    if b.metadata['joint_names']!=p['joint_names'] or b.metadata['face_channel_names']!=p['face_channel_names'] or b.metadata['ownership']!=p['ownership']: raise ValueError('Bundle channel/ownership profile mismatch')
    sampled=sample(b,b.arrays['sample_times_s'] if times is None else times)
    drivers=clear_legacy();arm=bpy.data.objects[p['armature']]
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
    # Kept drivers (outfit switches) must see the frame before anything reads visibility.
    bpy.context.scene.frame_set(1)
    if record is not None:record.update(drivers=drivers,driver_inputs=driver_inputs())
    return sampled,p,face


def look_mode(look):
    """'asset': the asset carries its own look and nothing is overridden; 'profile': legacy constants."""
    mode=look.get('mode','profile')
    if mode not in ('asset','profile'):raise ValueError('Unknown look mode: '+str(mode))
    return mode


def set_input(overrides,material,node,socket,value,stage='render',kind='override'):
    """Set an unlinked socket and record before -> after; a linked socket is recorded as ignored."""
    target=node.inputs[socket];entry={'stage':stage,'kind':kind,'material':material,'node':node.name,'input':socket,'after':_plain(value)}
    if target.is_linked:entry.update(before='linked',ignored='linked')
    else:entry['before']=_plain(target.default_value);target.default_value=value
    overrides.append(entry);return entry


def _plain(value):
    try:return [round(float(x),6) for x in value]
    except TypeError:return round(float(value),6)


def apply_look(profile_path,materials=None):
    """Apply the profile look; return the list of changes made to the asset's materials."""
    if materials is None:
        import bpy;materials=bpy.data.materials
    look=json.loads((Path(profile_path)/'look.json').read_text());overrides=[]
    if look_mode(look)=='asset':return overrides
    h=look.get('hair')
    if h:
        m=materials[h['material']];tree=m.node_tree;bsdf=tree.nodes[h['node']]
        if 'XMS_HairTint_v1' not in tree.nodes:
            link=bsdf.inputs['Base Color'].links[0];source=link.from_socket
            node=tree.nodes.new('ShaderNodeMixRGB');node.name='XMS_HairTint_v1';node.blend_type='MULTIPLY';node.inputs[0].default_value=1;node.inputs[2].default_value=h['tint']
            tree.links.remove(link);tree.links.new(source,node.inputs[1]);tree.links.new(node.outputs[0],bsdf.inputs['Base Color'])
            overrides.append({'stage':'render','kind':'override','material':h['material'],'node':'XMS_HairTint_v1','input':'Base Color','before':'texture','after':'texture x '+str(_plain(h['tint']))})
    skin=look.get('skin')
    for name in (skin or {}).get('materials',[]):
        if name in materials:
            b=materials[name].node_tree.nodes['Principled_glTF']
            set_input(overrides,name,b,'Specular IOR Level',skin['specular']);set_input(overrides,name,b,'Roughness',skin['roughness'])
    return overrides


def setup_scene(scene_path,scene_profile,character_profile,record=None):
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
    overrides=apply_look(character_profile)
    if record is not None:record['look_overrides']=overrides
    camera=bpy.data.objects['Xa_Cam'];c=scene_profile['camera'];camera.location=c['position_blender_m'];camera.rotation_euler=[math.radians(x) for x in c['rotation_euler_degrees']]
    camera.data.sensor_fit='VERTICAL';camera.data.angle_y=math.radians(c['vertical_fov_degrees']);camera.data.shift_x=0;camera.data.shift_y=0;sc.camera=camera
    sc.render.engine='BLENDER_EEVEE_NEXT';sc.eevee.taa_render_samples=scene_profile['samples']
    sc.render.resolution_x,sc.render.resolution_y=scene_profile['resolution'];sc.render.resolution_percentage=100
    sc.view_settings.view_transform='AgX';sc.view_settings.look='AgX - Medium High Contrast';sc.view_settings.exposure=scene_profile['exposure'];sc.render.image_settings.file_format='PNG'
    return sc
