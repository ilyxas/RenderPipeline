"""One sequential MP4 → observations → bundle → Blender → MP4 command."""
import json,time,uuid,sys,traceback,shutil
from fractions import Fraction
from pathlib import Path
from .animation.io import write_bundle,file_hash,bundle_hash
from .observations.io import read_observations
from .profiles.character import load_character
from .profiles.scene import load_scene
from .render.process import render
from .render.camera import presentation,PRESETS
from .assembly.encode import preflight,encode,verify
from .qa.contract import self_peak_bytes


def run(video,start=None,end=None,config=None,runs_root='runs',channels='full',solver='video',quality='preview',output=None,face_closeup=False,audio_mode='envelope',surface_qa='sampled',camera='fit',refine=True,allow_degraded_final=False,resolution=None,keep_frames=False,engine='eevee',samples=None):
    out=Path(runs_root).resolve()/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]);out.mkdir(parents=True);(out/'logs').mkdir();(out/'outputs').mkdir()
    manifest={'schema_version':'xms.run.v1','quality':'development_candidate','status':'running','parameters':{'input':str(Path(video).resolve()),'start_s':start,'end_s':end,'character':'xandra','scene':'bedroom','quality':quality,'engine':engine,'samples':samples,'solver':solver,'channels':channels,'audio':audio_mode,'surface_qa':surface_qa,'camera':camera},'config':config,'python':sys.version,'limitations':[],'warnings':[],'stages':{}}
    began=time.monotonic()
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def log(message):print(message,flush=True)
    def optional(name,action):
        try:return action()
        except Exception as error:
            warning=f'{name}: {error}';manifest['warnings'].append(warning);log('WARNING '+warning);return None
    save();log('RUN '+str(out))
    try:
        if not Path(video).is_file():raise ValueError('Input MP4 does not exist: '+str(video))
        if output and Path(output).exists():raise ValueError('Output already exists: '+str(output)+'; choose a new --output path')
        p,rig,face=load_character(config['character_profile']);sp=load_scene(config['scene_profile'])
        from .ingest.probe import probe
        info=probe(video,config['ffprobe']);start=0 if start is None else start
        # Source last PTS plus one CFR output step is the same timeline bound.
        end=float(Fraction(info['pts'][-1]-info['pts'][0])*Fraction(info['time_base'])+1/Fraction(info['fps'])) if end is None else end
        manifest['parameters'].update(start_s=start,end_s=end)
        manifest['input_hashes']={k:file_hash(v) for k,v in {'video':video,'character':config['character_asset'],'scene':config['scene_asset'],'pose_model':config['pose_model']}.items()}
        if manifest['input_hashes']['character']!=p['asset_sha256'] or manifest['input_hashes']['scene']!=sp['asset_sha256']:
            if config.get('_config_path'):
                from .registry import check
                check(config)
            raise ValueError('Registered character/scene asset hash mismatch; re-register the correct assets')
        if channels=='full':
            for key in ('face_model','hand_model'):
                if key not in config:raise ValueError(f'Missing {key}; configure it or use --channels body')
                manifest['input_hashes'][key]=file_hash(config[key])
        manifest.update(character_profile_hash=p['profile_hash'],scene_profile_hash=sp['profile_hash']);save()
        log(f'OBSERVE {video} [{start:g}, {end:g}) | solver={solver}, quality={quality}, channels={channels}')
        from .observations.pose_mediapipe import observe
        observe(video,start,end,out/'observations',config['pose_model'],config['ffmpeg'],config['ffprobe'])
        timeline=json.loads((out/'observations/timeline.json').read_text());manifest['stages']['observe']=json.loads((out/'observations/manifest.json').read_text())
        manifest['performance']={'tracking_python_peak_rss_bytes':self_peak_bytes()};audio=None
        if channels=='full':
            from .observations.details import observe_details
            detail_face,detail_hands=observe_details(video,out/'observations',config);manifest['stages']['observe_details']=json.loads((out/'observations/details-manifest.json').read_text())
            from .observations.audio import observe as observe_audio
            audio=optional('audio cues (video-only fallback)',lambda:observe_audio(video,timeline,config,out/'observations/audio',audio_mode))
            if audio:manifest['warnings']+=audio['metadata']['warnings']
        save();t=time.monotonic();body=read_observations(out/'observations/body');log('SOLVE '+solver)
        if solver=='viggle':
            # Experimental external mocap; MediaPipe body observations still define the timeline.
            from .observations.viggle import fetch
            from .solve.viggle import solve as viggle_solve
            glb=fetch(video,float(Fraction(timeline['fps'])),log)
            if channels=='full':
                from .solve.viggle import solve_full as viggle_full
                b=viggle_full(glb,detail_face,timeline,p,rig,face,audio,body[0]['source_sha256'])
            else:b=viggle_solve(glb,timeline,p,rig,body[0]['source_sha256'])
        elif solver=='baseline':
            if channels=='full':
                from .solve.baseline_full import solve_full
                b=solve_full(body,detail_face,detail_hands,timeline,p,rig,face)
            else:
                from .solve.baseline_body import solve
                b=solve(body,timeline,p,rig)
        elif channels=='full':
            from .solve.reconstruction import solve_full
            b=solve_full(body,detail_face,detail_hands,timeline,p,rig,face,solver,audio)
        else:
            if solver=='temporal':from .solve.windows import solve
            else:from .solve.video_body import solve
            b=solve(body,timeline,p,rig)
            # The full path applies these after hands; body-only has no later arm stage.
            from .solve.limits import apply as arm_limits
            from .solve.twist import distribute
            arm_limits(b,p);distribute(b)
        if 'grounding_diagnostics' in b.metadata:
            g=b.metadata['grounding_diagnostics']
            log(f"GROUND support={len(g['support_sample_indices'])}, flight={len(g['flight_sample_indices'])}, unavailable={g['unsupported_samples']}, bounded={g['correction_bound_samples']}, floor_error={g['max_support_floor_error_m']:.4f}m")
        if 'bounded_palm_samples' in b.metadata.get('hand_diagnostics',{}):
            hand=b.metadata['hand_diagnostics']
            log(f"HAND estimated palm samples={hand['bounded_palm_samples']}, finger samples={hand['bounded_finger_samples']}, anatomical constraints={hand.get('anatomical_finger_samples',0)}; angular fallbacks={hand['palm_rate_fallback_deg_s']} / {hand['finger_rate_fallback_deg_s']} deg/s")
        b.metadata['observations_arrays_sha256']=file_hash(out/'observations/body/arrays.npz')
        if channels=='full':
            b.metadata['face_observations_sha256']=file_hash(out/'observations/face/arrays.npz');b.metadata['hand_observations_sha256']=file_hash(out/'observations/hands/arrays.npz')
        manifest['limitations']=b.metadata['limitations'];write_bundle(b,out/'animation-candidate');chosen=out/'animation-candidate'
        (out/'calibration.json').write_text(json.dumps(b.metadata['calibration'],indent=2))
        manifest['stages']['solve']={'exit_status':0,'seconds':time.monotonic()-t,'backend':b.metadata['solver']};manifest['performance']['solve_python_peak_rss_bytes']=self_peak_bytes()
        diagnostics={k:v for k,v in b.metadata.items() if k.endswith('diagnostics')};(out/'solve-diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        from .qa.continuity import measure as continuity
        manifest['stages']['continuity']=continuity(b)
        if surface_qa!='off':
            from .qa.surface_collision import evaluate
            log('SURFACE QA '+surface_qa)
            surface=optional('surface QA',lambda:evaluate(chosen,config,out/'surface',surface_qa));manifest['stages']['surface']=surface or {'status':'unavailable'}
            skin=bool(surface) and any(row.get('witnesses') for row in surface.get('records',[]))
            if skin and refine and surface['status']=='needs_review' and solver!='baseline':
                # Stage-15 windowed refinement from exact skin witnesses: one candidate, fresh
                # exact QA, every frozen gate must pass or the source stays (needs_review).
                from .solve.collision_refine import refine as refine_collisions,accept as accept_collisions
                trial=optional('bounded collision refinement',lambda:refine_collisions(b,surface))
                if trial:
                    write_bundle(trial,out/'animation-refined');trial_surface=optional('refined surface QA',lambda:evaluate(out/'animation-refined',config,out/'surface-refined',surface_qa))
                    verdict=accept_collisions(surface,trial_surface,b,trial) if trial_surface else {'accepted':False,'gates':{'refined_surface_qa':False}}
                    manifest['stages']['surface_refinement']={'passes':1,'method':'collision_refine','accepted':verdict['accepted'],'refinement':trial.metadata['collision_refinement'],'verdict':verdict,'surface':trial_surface}
                    if verdict['accepted']:b=trial;chosen=out/'animation-refined';manifest['stages']['surface']=trial_surface;manifest['stages']['continuity']=continuity(trial)
                    else:manifest['warnings'].append('Collision refinement rejected by frozen gates: '+', '.join(k for k,v in verdict['gates'].items() if not v)+'; source bundle kept (needs_review).')
            elif surface and refine and surface['status']=='needs_review' and solver!='baseline':
                from .solve.surface_refine import refine as refine_surface
                trial=optional('one bounded surface refinement',lambda:refine_surface(b,surface))
                if trial and not trial.metadata['surface_refinement']['passes']:
                    manifest['stages']['surface_refinement']={'passes':0,'accepted':False,'skipped_samples':trial.metadata['surface_refinement'].get('skipped_samples',[])}
                    manifest['warnings'].append('Surface defect remains; insufficient reliable arm/wrist channels for bounded refinement.')
                if trial and trial.metadata['surface_refinement']['passes']:
                    write_bundle(trial,out/'animation-refined');trial_surface=optional('refined surface QA',lambda:evaluate(out/'animation-refined',config,out/'surface-refined',surface_qa))
                    from .qa.contacts import measure as contacts
                    old_contact=contacts(b);new_contact=contacts(trial);old_cont=continuity(b);new_cont=continuity(trial)
                    drift_ok=old_contact['p95_segment_max_drift_m'] is None or (new_contact['p95_segment_max_drift_m'] is not None and new_contact['p95_segment_max_drift_m']<=old_contact['p95_segment_max_drift_m']+.01)
                    accepted=bool(trial_surface and trial_surface['max_candidate_depth_m']<surface['max_candidate_depth_m'] and new_cont['max_rotation_step_deg']<=old_cont['max_rotation_step_deg']+5 and drift_ok)
                    manifest['stages']['surface_refinement']={'passes':1,'accepted':accepted,'continuity':new_cont,'contacts':new_contact,'surface':trial_surface}
                    if accepted:b=trial;chosen=out/'animation-refined';manifest['stages']['surface']=trial_surface;manifest['stages']['continuity']=new_cont
        # Publish an immutable byte-identical copy of the evaluated candidate.
        shutil.copytree(chosen,out/'animation');h=bundle_hash(out/'animation');manifest['bundle_hash']=h;save()
        # This inexpensive gate prevents known catastrophic motion from entering
        # a costly final render. It is separate from frozen quality benchmarks.
        from .stability_gate import reasons as motion_reasons
        blockers=motion_reasons(b,manifest['stages'].get('surface'))
        manifest['stages']['motion_gate']={'blocked_final':bool(blockers),'reasons':blockers,'override':allow_degraded_final};save()
        for reason in blockers:log('MOTION WARNING '+reason)
        if quality=='final' and blockers and not allow_degraded_final:
            raise ValueError('Final render blocked by motion defects: '+ '; '.join(blockers)+'. Use --quality preview to inspect, or --allow-degraded-final to explicitly override. Bundle: '+str(out/'animation'))
        plan=presentation(b,sp,quality,camera,resolution);manifest['parameters']['resolution']=plan['resolution'];from .qa.framing import measure as framing
        manifest['stages']['framing']=framing(plan);(out/'camera-plan.json').write_text(json.dumps(plan,indent=2))
        preflight(config['ffmpeg'],plan['resolution'],timeline['fps'],out/'outputs/encode-preflight.mp4',out/'logs/encode-preflight.log')
        if quality=='final':
            log('FINAL preflight: whole-clip preview')
            render(out/'animation',out/'preview-frames',config,timeline['fps'],out/'logs/preview-blender.log',camera_plan=out/'camera-plan.json')
            encode(out/'preview-frames',video,timeline,config['ffmpeg'],out/'outputs/preview.mp4',out/'logs/preview-encode.log')
            manifest['stages']['preview_verification']=verify(out/'outputs/preview.mp4',timeline,PRESETS['preview']['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs/preview-decode.log')
        log(f'RENDER {quality} engine={engine} samples={samples or "auto"}')
        manifest['stages']['render_process']=render(out/'animation',out/'frames',config,timeline['fps'],out/'logs/blender.log',quality=quality,camera_plan=out/'camera-plan.json',resolution=plan['resolution'],engine=engine,samples=samples)
        rendered=json.loads((out/'frames/manifest.json').read_text());assert rendered['bundle_hash']==h
        manifest['stages']['render']=rendered;manifest['look_overrides']=rendered.get('look_overrides',[])
        if any(r.get('kind')=='override' for r in manifest['look_overrides']):
            manifest['warnings'].append('Render look differs from the asset: the profile overrides '+', '.join(sorted({r['material'] for r in manifest['look_overrides'] if r.get('kind')=='override'})))
        save()
        log('ENCODE main video');destination=out/'outputs/video.mp4';encode(out/'frames',video,timeline,config['ffmpeg'],destination,out/'logs/encode.log')
        manifest['verification']=verify(destination,timeline,plan['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs/full-decode.log')
        if face_closeup:
            def closeup():
                log('RENDER face close-up')
                render(out/'animation',out/'face-frames',config,timeline['fps'],out/'logs/face-blender.log',view='face',quality=quality,engine=engine,samples=samples)
                close=out/'outputs/face.mp4';encode(out/'face-frames',video,timeline,config['ffmpeg'],close,out/'logs/face-encode.log')
                checked=verify(close,timeline,sp['diagnostic_views']['face']['resolution'],config['ffmpeg'],config['ffprobe'],out/'logs/face-decode.log');return {'path':str(close),'bundle_hash':h,'verification':checked}
            result=optional('face close-up',closeup);manifest['diagnostic_videos']={'face':result} if result else {}
        manifest['output_sha256']=file_hash(destination);manifest['outputs']={'video':str(destination),'bundle':str(out/'animation'),'observations':str(out/'observations'),'manifest':str(out/'manifest.json')}
        def glb():
            from .export.glb import export as export_glb,verify as verify_glb
            log('EXPORT animated GLB');path=out/'outputs/animation.glb';stage=export_glb(out/'animation',config,timeline['fps'],path,out/'logs/glb-export.log')
            stage['verification']=verify_glb(path,out/'animation',face);stage['path']=str(path)
            record=json.loads(path.with_suffix('.export.json').read_text())
            stage.update({k:record.get(k) for k in ('look_mode','influences','masks','driver_inputs','visible_sets','primitives_with_joints_1')})
            stage['influences']={k:v for k,v in (record.get('influences') or {}).items() if k!='per_mesh'}
            manifest['look_overrides']=manifest.get('look_overrides',[])+record.get('look_overrides',[])
            manifest['warnings']+=['GLB: '+w for w in record.get('warnings',[])]
            if stage['verification']['max_joint_position_error_m']>.001:manifest['warnings'].append('GLB joint positions differ from the bundle by more than 1 mm')
            return stage
        exported=optional('animated GLB export',glb);manifest['stages']['glb_export']=exported or {'status':'unavailable'}
        if exported:manifest['outputs']['glb']=exported['path']
        if output:
            output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create prevents accidental replacement, including races.
            with destination.open('rb') as source,output.open('xb') as target:shutil.copyfileobj(source,target)
            manifest['outputs']['video']=str(output)
        manifest.update(status='rendered_pending_visual_review',exit_status=0,result_exit_code=0,quality_accepted=False);save()
        from .assembly.compare import compare
        log('COMPARE source and render')
        comparison=optional('comparison video',lambda:compare(video,out,out/'comparison',config))
        if comparison:manifest['outputs']['compare']=str(comparison)
        (out/'limitations.md').write_text('\n'.join('- '+x for x in manifest['limitations']+manifest['warnings']))
        from .qa.run_report import quality as measure_quality
        q=measure_quality(out);manifest['quality_status']=q['status'];manifest['outputs'].update(quality=str(out/'report/quality.json'),report=str(out/'report/report.html'))
        if not keep_frames:
            # PNG sequences are encoder input only; the MP4s, bundle, GLB and report remain.
            from .clean import prune_frames
            manifest['frames_pruned_bytes']=prune_frames(out)
        log('VIDEO '+manifest['outputs']['video']);log('BUNDLE '+str(out/'animation'));log('QUALITY '+q['status']+' (rendered candidate; visual acceptance pending)')
    except Exception as error:
        manifest.update(status='failed',exit_status=1,result_exit_code=1,error=str(error));(out/'logs/error.log').write_text(traceback.format_exc());log('ERROR '+str(error)+' | diagnostics: '+str(out/'logs/error.log'))
    except KeyboardInterrupt:
        # Ctrl+C, SIGTERM or SIGHUP (see cli.py): a stopped run is not a failed one.
        manifest.update(status='cancelled',exit_status=130,result_exit_code=130,error='Interrupted');log('CANCELLED run interrupted');raise
    finally:
        manifest['seconds']=time.monotonic()-began;save()
        from .qa.run_report import write_report
        write_report(out,out/'report',out/'comparison/compare.mp4' if (out/'comparison/compare.mp4').exists() else None)
    return out
