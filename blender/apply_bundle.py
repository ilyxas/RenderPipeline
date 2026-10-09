"""Stage 2 diagnostic consumer; numerical comparison and review stills."""
import sys, json, argparse, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy
import numpy as np
from xms.render.blender_adapter import apply_bundle,setup_scene,C
from xms.animation.fk import evaluate,matrix_quat
from xms.animation.io import bundle_hash,file_hash
from xms.profiles.scene import load_scene

p=argparse.ArgumentParser()
p.add_argument('--numerical-only',action='store_true')
for name in ('bundle','profile','scene','scene-profile','out'):p.add_argument('--'+name,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
b,profile,face=apply_bundle(a.bundle,a.profile,fps=1)
sc=setup_scene(a.scene,load_scene(a.scene_profile),a.profile);sc.render.resolution_x=360;sc.render.resolution_y=640;sc.eevee.taa_render_samples=8
rows=[];arm=bpy.data.objects[profile['armature']]
for i,t in enumerate(b.arrays['sample_times_s']):
    sc.frame_set(1+int(t));target=evaluate(b,i);evaluated=arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    actual=np.array([C@np.array(evaluated.matrix_world@evaluated.pose.bones[n].matrix)@C.T for n in profile['joint_names']])
    pos=float(np.max(np.linalg.norm(actual[:,:3,3]-target[:,:3,3],axis=1)))
    angular=[]
    for x,y in zip(actual,target):
        rx=x[:3,:3]/np.linalg.norm(x[:3,:3],axis=0);ry=y[:3,:3]/np.linalg.norm(y[:3,:3],axis=0)
        angular.append(math.degrees(2*math.acos(min(1,abs(float(np.dot(matrix_quat(rx),matrix_quat(ry))))))))
    weights=[bpy.data.objects[s['object']].data.shape_keys.key_blocks[s['key']].value for s in face['channels'].values()]
    werr=float(np.max(np.abs(weights-b.arrays['face_weights'][i])))
    name=b.metadata['diagnostics'][i];row={'name':name,'position_error_m':pos,'angular_error_deg':max(angular),'weight_error':werr}
    np.savez_compressed(out/(name+'-evaluated.npz'),world_matrices=actual,face_weights=weights)
    rows.append(row)
    (out/'numerical.json').write_text(json.dumps(rows,indent=2))
    if pos>.001 or max(angular)>.1 or werr>1e-5:raise RuntimeError('FK/Blender mismatch: '+str(row))
    if a.numerical_only:continue
    # Distinct framed views retain the same pose; face and wrist details are reviewable.
    cam=sc.camera
    cam.location=(0,-3.8,.88);cam.data.angle_y=math.radians(36)
    if name in ('head_yaw','jaw_open','blink_left','blink_right','smile_left','smile_right','face_neutral'):
        head=actual[profile['joint_names'].index('head'),:3,3]
        cam.location=(float(head[0]),float(-head[2]-1.1),float(head[1]+.04));cam.data.angle_y=math.radians(24)
    if name.startswith('wrist_'):
        side=name[-1];hand=actual[profile['joint_names'].index('hand_'+side),:3,3]
        cam.location=(float(hand[0]),float(-hand[2]-1.0),float(hand[1]));cam.data.angle_y=math.radians(28)
    sc.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    print('DIAGNOSTIC',name,row,flush=True)
manifest={'stage':2,'status':'numerical_pass_visual_review_pending','bundle_hash':bundle_hash(a.bundle),'profile_hash':profile['profile_hash'],'scene_profile_hash':load_scene(a.scene_profile)['profile_hash'],'blender':bpy.app.version_string,'asset_hash_after':file_hash(bpy.data.filepath),'scene_hash_after':file_hash(a.scene),'diagnostics':rows}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
