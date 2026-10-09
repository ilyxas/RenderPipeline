"""Render-only process: bundle + character + scene. No source video or tracker."""
import sys,json,argparse,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy
import numpy as np
from xms.animation.io import read_bundle,bundle_hash,file_hash
from xms.render.blender_adapter import apply_bundle,setup_scene
from xms.profiles.scene import load_scene
p=argparse.ArgumentParser()
for n in ('bundle','profile','scene','scene-profile','out'):p.add_argument('--'+n,required=True)
p.add_argument('--quality',choices=['preview','final'],default='preview');p.add_argument('--camera-plan')
p.add_argument('--fps',required=True);p.add_argument('--frame-indices')
p.add_argument('--view',default='main',choices=['main','face','hand_l','hand_r'])
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
from fractions import Fraction
fps=Fraction(a.fps);original=read_bundle(a.bundle);times=original.arrays['sample_times_s']
if not np.allclose(times,np.arange(len(times))/float(fps),atol=1e-8):raise ValueError('Render grid differs from bundle support')
b,profile,face=apply_bundle(a.bundle,a.profile,times,fps=float(fps));sp=load_scene(a.scene_profile);sc=setup_scene(a.scene,sp,a.profile)
from xms.render.camera import PRESETS
sc.render.resolution_x,sc.render.resolution_y=PRESETS[a.quality]['resolution'];sc.eevee.taa_render_samples=PRESETS[a.quality]['samples']
if a.camera_plan:
    plan=json.loads(Path(a.camera_plan).read_text());sc.camera.location=plan['camera']['position_blender_m'];sc.camera.data.angle_y=math.radians(plan['camera']['vertical_fov_degrees'])
diagnostic=sp.get('diagnostic_views',{}).get(a.view)
if a.view!='main' and diagnostic is None:raise ValueError('Diagnostic camera not in scene profile')
if diagnostic:
    sc.render.resolution_x,sc.render.resolution_y=diagnostic['resolution'];sc.camera.data.angle_y=math.radians(diagnostic['vertical_fov_degrees'])
sc.render.fps=round(float(fps));sc.render.fps_base=sc.render.fps/float(fps)
selected=list(range(len(times))) if not a.frame_indices else [int(x) for x in a.frame_indices.split(',')]
if any(i<0 or i>=len(times) for i in selected):raise ValueError('Frame index outside bundle')
start=time.monotonic()
for i in selected:
    sc.frame_set(i+1)
    if diagnostic:
        arm=bpy.data.objects[profile['armature']];position=arm.matrix_world@arm.pose.bones[diagnostic['bone']].head
        sc.camera.location=[float(position[k])+diagnostic['offset_blender_m'][k] for k in range(3)]
    sc.render.filepath=str(out/f'{i:06d}.png');bpy.ops.render.render(write_still=True);print('XMS_FRAME',i,flush=True)
manifest={'schema_version':'xms.render.v1','status':'passed','exit_status':0,'quality':a.quality,'bundle_hash':bundle_hash(a.bundle),'character_profile_hash':profile['profile_hash'],'scene_profile_hash':sp['profile_hash'],'character_asset_sha256_after':file_hash(bpy.data.filepath),'scene_asset_sha256_after':file_hash(a.scene),'blender_version':bpy.app.version_string,'engine':sc.render.engine,'resolution':sp['resolution'],'fps':str(fps),'frame_indices':selected,'seconds':time.monotonic()-start,'inputs':['AnimationBundle','character profile/asset','scene profile/asset'],'source_video_required':False,'tracker_required':False}
manifest['samples']=sc.eevee.taa_render_samples;manifest['view']=a.view;manifest['resolution']=[sc.render.resolution_x,sc.render.resolution_y]
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
