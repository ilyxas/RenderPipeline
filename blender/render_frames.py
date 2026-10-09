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
p.add_argument('--fps',required=True);p.add_argument('--frame-indices')
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
from fractions import Fraction
fps=Fraction(a.fps);original=read_bundle(a.bundle);times=original.arrays['sample_times_s']
if not np.allclose(times,np.arange(len(times))/float(fps),atol=1e-8):raise ValueError('Render grid differs from bundle support')
b,profile,face=apply_bundle(a.bundle,a.profile,times,fps=float(fps));sp=load_scene(a.scene_profile);sc=setup_scene(a.scene,sp,a.profile)
sc.render.fps=round(float(fps));sc.render.fps_base=sc.render.fps/float(fps)
selected=list(range(len(times))) if not a.frame_indices else [int(x) for x in a.frame_indices.split(',')]
if any(i<0 or i>=len(times) for i in selected):raise ValueError('Frame index outside bundle')
start=time.monotonic()
for i in selected:
    sc.frame_set(i+1);sc.render.filepath=str(out/f'{i:06d}.png');bpy.ops.render.render(write_still=True);print('XMS_FRAME',i,flush=True)
manifest={'schema_version':'xms.render.v1','status':'passed','exit_status':0,'quality':'baseline_preview','bundle_hash':bundle_hash(a.bundle),'character_profile_hash':profile['profile_hash'],'scene_profile_hash':sp['profile_hash'],'character_asset_sha256_after':file_hash(bpy.data.filepath),'scene_asset_sha256_after':file_hash(a.scene),'blender_version':bpy.app.version_string,'engine':sc.render.engine,'resolution':sp['resolution'],'fps':str(fps),'frame_indices':selected,'seconds':time.monotonic()-start,'inputs':['AnimationBundle','character profile/asset','scene profile/asset'],'source_video_required':False,'tracker_required':False}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
