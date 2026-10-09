"""Stages 0–4 sequential vertical slice. JSON evidence, no job infrastructure."""
import json,time,uuid,sys,subprocess,traceback
from pathlib import Path
from .animation.io import write_bundle,file_hash,bundle_hash
from .observations.io import read_observations
from .profiles.character import load_character
from .profiles.scene import load_scene
from .solve.baseline_body import solve,LIMITATIONS
from .render.process import render
from .assembly.encode import preflight,encode,verify


def run(video,start,end,config,runs_root='runs'):
    out=Path(runs_root)/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True);(out/'logs').mkdir();(out/'outputs').mkdir()
    manifest={'schema_version':'xms.run.v1','quality':'baseline_preview','status':'running','parameters':{'input':str(Path(video).resolve()),'start_s':start,'end_s':end,'character':'xandra','scene':'bedroom','quality':'preview','solver':'baseline'},'config':config,'python':sys.version,'limitations':LIMITATIONS,'stages':{}}
    began=time.monotonic()
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save();print('RUN',out,flush=True)
    try:
        p,rig,face=load_character(config['character_profile']);sp=load_scene(config['scene_profile'])
        manifest['input_hashes']={k:file_hash(v) for k,v in {'video':video,'character':config['character_asset'],'scene':config['scene_asset'],'pose_model':config['pose_model']}.items()}
        if manifest['input_hashes']['character']!=p['asset_sha256'] or manifest['input_hashes']['scene']!=sp['asset_sha256']:raise ValueError('Registered source asset hash mismatch')
        manifest['character_profile_hash']=p['profile_hash'];manifest['scene_profile_hash']=sp['profile_hash']
        manifest['versions']={n:subprocess.check_output([config[n],'-version'],text=True).splitlines()[0] for n in ['ffmpeg','ffprobe']}
        from .observations.pose_mediapipe import observe
        observe(video,start,end,out/'observations',config['pose_model'],config['ffmpeg'],config['ffprobe'])
        timeline=json.loads((out/'observations/timeline.json').read_text());manifest['stages']['observe']=json.loads((out/'observations/manifest.json').read_text());save()
        t=time.monotonic();b=solve(read_observations(out/'observations/body'),timeline,p,rig)
        b.metadata['observations_arrays_sha256']=file_hash(out/'observations/body/arrays.npz')
        h=write_bundle(b,out/'animation');(out/'calibration.json').write_text(json.dumps(b.metadata['calibration'],indent=2))
        manifest['bundle_hash']=h;manifest['stages']['solve']={'exit_status':0,'seconds':time.monotonic()-t};save()
        preflight(config['ffmpeg'],sp['resolution'],timeline['fps'],out/'outputs/encode-preflight.mp4',out/'logs/encode-preflight.log')
        manifest['stages']['render_process']=render(out/'animation',out/'frames',config,timeline['fps'],out/'logs/blender.log')
        rendered=json.loads((out/'frames/manifest.json').read_text());assert rendered['bundle_hash']==h
        manifest['stages']['render']=rendered;save()
        output=out/'outputs/video.mp4';encode(out/'frames',video,timeline,config['ffmpeg'],output,out/'logs/encode.log')
        manifest['verification']=verify(output,timeline,sp['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs/full-decode.log')
        manifest['output_sha256']=file_hash(output);manifest['outputs']={'video':str(output),'bundle':str(out/'animation'),'observations':str(out/'observations'),'manifest':str(out/'manifest.json')}
        for key,path in [('video',video),('character',config['character_asset']),('scene',config['scene_asset']),('pose_model',config['pose_model'])]:
            assert file_hash(path)==manifest['input_hashes'][key],'Input changed: '+key
        manifest['status']='rendered_pending_visual_review';manifest['exit_status']=0
        (out/'limitations.md').write_text('# Baseline preview limitations\n\n'+'\n'.join('- '+x for x in LIMITATIONS)+'\n')
    except Exception as error:
        manifest.update(status='failed',exit_status=1,error=str(error));(out/'logs/error.log').write_text(traceback.format_exc());raise
    finally:manifest['seconds']=time.monotonic()-began;save()
    return out
