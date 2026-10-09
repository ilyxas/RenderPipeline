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
from .qa.contract import self_peak_bytes,NeedsInput,EXIT_CODES


def run(video,start,end,config,runs_root='runs',channels='body',solver='baseline'):
    out=Path(runs_root)/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True);(out/'logs').mkdir();(out/'outputs').mkdir()
    manifest={'schema_version':'xms.run.v1','quality':'development_candidate' if solver=='temporal' else 'baseline_preview','status':'running','parameters':{'input':str(Path(video).resolve()),'start_s':start,'end_s':end,'character':'xandra','scene':'bedroom','quality':'preview','solver':solver},'config':config,'python':sys.version,'limitations':LIMITATIONS,'stages':{}}
    began=time.monotonic()
    manifest['parameters']['channels']=channels
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save();print('RUN',out,flush=True)
    try:
        p,rig,face=load_character(config['character_profile']);sp=load_scene(config['scene_profile'])
        manifest['input_hashes']={k:file_hash(v) for k,v in {'video':video,'character':config['character_asset'],'scene':config['scene_asset'],'pose_model':config['pose_model']}.items()}
        if manifest['input_hashes']['character']!=p['asset_sha256'] or manifest['input_hashes']['scene']!=sp['asset_sha256']:raise ValueError('Registered source asset hash mismatch')
        if channels=='full':
            for key in ('face_model','hand_model'):manifest['input_hashes'][key]=file_hash(config[key])
        manifest['character_profile_hash']=p['profile_hash'];manifest['scene_profile_hash']=sp['profile_hash']
        manifest['versions']={n:subprocess.check_output([config[n],'-version'],text=True).splitlines()[0] for n in ['ffmpeg','ffprobe']}
        from .observations.pose_mediapipe import observe
        observe(video,start,end,out/'observations',config['pose_model'],config['ffmpeg'],config['ffprobe'])
        manifest['performance']={'tracking_python_peak_rss_bytes':self_peak_bytes(),'memory_semantics':'Process high-water RSS; solve shares the tracking process, not a subtractable incremental peak.'}
        timeline=json.loads((out/'observations/timeline.json').read_text());manifest['stages']['observe']=json.loads((out/'observations/manifest.json').read_text());save()
        if channels=='full':
            from .observations.details import observe_details
            detail_face,detail_hands=observe_details(video,out/'observations',config)
            manifest['stages']['observe_details']=json.loads((out/'observations/details-manifest.json').read_text())
        t=time.monotonic()
        if solver=='temporal':
            from .solve.temporal_body import solve as solve_temporal
            from .solve.temporal_full import solve_full as solve_temporal_full
            if channels=='full':b=solve_temporal_full(read_observations(out/'observations/body'),detail_face,detail_hands,timeline,p,rig,face)
            else:b=solve_temporal(read_observations(out/'observations/body'),timeline,p,rig)
            manifest['limitations']=b.metadata['limitations']
            (out/'temporal_diagnostics.json').write_text(json.dumps(b.metadata['temporal_diagnostics'],indent=2))
            if not b.metadata['temporal_diagnostics']['success']:raise RuntimeError('Temporal optimizer exhausted; not a successful solve')
        elif channels=='full':
            from .solve.baseline_full import solve_full,FULL_LIMITATIONS
            b=solve_full(read_observations(out/'observations/body'),detail_face,detail_hands,timeline,p,rig,face)
            manifest['limitations']=FULL_LIMITATIONS
            b.metadata['face_observations_sha256']=file_hash(out/'observations/face/arrays.npz');b.metadata['hand_observations_sha256']=file_hash(out/'observations/hands/arrays.npz')
        else:b=solve(read_observations(out/'observations/body'),timeline,p,rig)
        if channels=='full':
            b.metadata['face_observations_sha256']=file_hash(out/'observations/face/arrays.npz');b.metadata['hand_observations_sha256']=file_hash(out/'observations/hands/arrays.npz')
        b.metadata['observations_arrays_sha256']=file_hash(out/'observations/body/arrays.npz')
        h=write_bundle(b,out/'animation');(out/'calibration.json').write_text(json.dumps(b.metadata['calibration'],indent=2))
        manifest['bundle_hash']=h;manifest['stages']['solve']={'exit_status':0,'seconds':time.monotonic()-t};save()
        manifest['performance']['solve_python_peak_rss_bytes']=self_peak_bytes()
        preflight(config['ffmpeg'],sp['resolution'],timeline['fps'],out/'outputs/encode-preflight.mp4',out/'logs/encode-preflight.log')
        manifest['stages']['render_process']=render(out/'animation',out/'frames',config,timeline['fps'],out/'logs/blender.log')
        rendered=json.loads((out/'frames/manifest.json').read_text());assert rendered['bundle_hash']==h
        manifest['stages']['render']=rendered;save()
        output=out/'outputs/video.mp4';encode(out/'frames',video,timeline,config['ffmpeg'],output,out/'logs/encode.log')
        manifest['verification']=verify(output,timeline,sp['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs/full-decode.log')
        if channels=='full':
            manifest['diagnostic_videos']={}
            for view in ('face','hand_l','hand_r'):
                directory=out/(view+'-frames');render(out/'animation',directory,config,timeline['fps'],out/'logs'/(view+'-blender.log'),view=view)
                destination=out/'outputs'/(view+'.mp4');encode(directory,video,timeline,config['ffmpeg'],destination,out/'logs'/(view+'-encode.log'))
                verification=verify(destination,timeline,sp['diagnostic_views'][view]['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs'/(view+'-decode.log'))
                manifest['diagnostic_videos'][view]={'path':str(destination),'bundle_hash':json.loads((directory/'manifest.json').read_text())['bundle_hash'],'verification':verification}
        manifest['output_sha256']=file_hash(output);manifest['outputs']={'video':str(output),'bundle':str(out/'animation'),'observations':str(out/'observations'),'manifest':str(out/'manifest.json')}
        for key,path in [('video',video),('character',config['character_asset']),('scene',config['scene_asset']),('pose_model',config['pose_model'])]:
            assert file_hash(path)==manifest['input_hashes'][key],'Input changed: '+key
        manifest['status']='rendered_pending_visual_review';manifest['exit_status']=0
        (out/'limitations.md').write_text('# Baseline preview limitations\n\n'+'\n'.join('- '+x for x in manifest['limitations'])+'\n')
        from .assembly.compare import compare
        from .qa.run_report import quality
        comparison=compare(video,out,out/'comparison',config)
        manifest['performance']['subprocesses']={str(x.relative_to(out)):json.loads(x.read_text()) for x in out.rglob('*.runtime.json')}
        manifest['performance']['blender']=manifest['stages']['render_process']
        save()
        q=quality(out);manifest['status']=q['status'];manifest['result_exit_code']=q['exit_code']
        manifest['outputs'].update(compare=str(comparison),quality=str(out/'report/quality.json'),report=str(out/'report/report.html'))
    except Exception as error:
        status='needs_input' if isinstance(error,NeedsInput) else 'failed'
        manifest.update(status=status,exit_status=EXIT_CODES[status],result_exit_code=EXIT_CODES[status],error=str(error));(out/'logs/error.log').write_text(traceback.format_exc());save()
        if status=='failed':raise
    finally:
        manifest['seconds']=time.monotonic()-began;save()
        from .qa.run_report import write_report
        comparison=out/'comparison/compare.mp4'
        write_report(out,out/'report',comparison if comparison.exists() else None)
    return out
