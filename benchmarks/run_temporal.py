"""Stage 8 frozen comparison: one configuration, same bundle consumer and renderer."""
import argparse,json,time,sys,platform
import scipy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image,ImageDraw
from fractions import Fraction
from xms.animation.io import read_bundle,write_bundle,file_hash,bundle_hash
from xms.observations.io import read_observations
from xms.observations.details_io import read
from xms.profiles.character import load_character
from xms.solve.calibration import array_digest,validate_calibration
from xms.solve.temporal_full import solve_full
from xms.qa.pose import measure_pose
from xms.qa.kinematics import measure_kinematics
from xms.qa.contract import write_json,self_peak_bytes
from xms.render.process import render
from xms.assembly.encode import encode,verify
from xms.assembly.compare import letterbox
from xms.ingest.decode import decode


def inputs(identifier):
    ref=json.loads(Path(f'benchmarks/baseline/{identifier}/measurements.json').read_text());obs=Path(ref['provenance']['source_observations_path'])
    body=read_observations(obs/'body');face=read(obs/'face');hands=read(obs/'hands');timeline=json.loads((obs/'timeline.json').read_text())
    p,rig,fmap=load_character('profiles/characters/xandra/v2');c=json.loads(Path(f'benchmarks/baseline/{identifier}/calibration.json').read_text())
    for value,key in [(body,'body'),(face,'face'),(hands,'hand')]:
        if array_digest(value[1])!=ref['provenance'][key+'_observations_digest']:raise ValueError('Frozen '+key+' observations changed')
    validate_calibration(c,body,timeline,p)
    assert p['profile_hash']==ref['provenance']['character_profile_hash']
    baseline=Path(f'runs/stage7/{identifier}/baseline/animation')
    assert bundle_hash(baseline)==ref['provenance']['bundle_hash']
    return ref,obs,body,face,hands,timeline,p,rig,fmap,c,baseline


def gates(ref,metrics,kin,acceptance,timeline,diag):
    policy=acceptance['stage8_regression'];result={}
    def add(name,value,threshold,available=True):result[name]={'status':'unavailable' if not available else 'pass' if value<=threshold else 'fail','value':value,'maximum':threshold}
    for name in ('median','p95'):
        key='normalized_reprojection_'+name;before=ref['metrics'][key]['value'];after=metrics[key]['value'];add(key,after,before*policy[key+'_max_ratio']+policy[key+'_additive_tolerance'])
    before=ref['metrics']['body_angular_acceleration_p95']['value'];add('body_acceleration_p95',metrics['body_angular_acceleration_p95']['value'],before*(1-policy['body_angular_acceleration_p95_improvement_min_fraction']))
    add('scalar_limit_fraction',metrics['joint_limit_violation_fraction']['value'],ref['metrics']['joint_limit_violation_fraction']['value']+policy['joint_limit_violation_fraction_increase_max'])
    for key in ('foot_bone_drift_proxy_median','foot_bone_drift_proxy_p95'):
        before=ref['metrics'][key]['value'];after=metrics[key]['value'];add(key,after,None if before is None else before+policy['foot_bone_drift_proxy_regression_max_m'],before is not None and after is not None)
    for event in kin['fast_gesture_reference']:
        old=next(x for x in ref['kinematics']['fast_gesture_reference'] if (x['label'],x['joint'])==(event['label'],event['joint']))
        prefix=event['label']+'/'+event['joint'];add(prefix+'/amplitude_loss',old['amplitude_degrees']-event['amplitude_degrees'],policy['annotated_fast_gesture_amplitude_loss_max_degrees']);add(prefix+'/peak_time_frames',abs(round((event['peak_source_s']-timeline['absolute_start_s'])*float(Fraction(timeline['fps'])))-round((old['peak_source_s']-timeline['absolute_start_s'])*float(Fraction(timeline['fps'])))),policy['annotated_fast_gesture_peak_time_error_output_frames'])
    for key,threshold in [('observed_adjacent_rotation_jump_max','observed_adjacent_rotation_jump_max_degrees'),('root_adjacent_step_max','root_adjacent_step_max_m')]:add(key,metrics[key]['value'],acceptance['continuity'][threshold])
    add('solve_wall_s',diag['full_solve_seconds'],acceptance['performance_budget']['solver_wall_s_max']);add('peak_rss_bytes',diag['python_peak_rss_bytes'],acceptance['performance_budget']['python_solver_peak_rss_bytes_max']);add('termination_success',0 if diag['success'] else 1,0)
    return result


def run(identifier,phase):
    acceptance_path=Path('benchmarks/acceptance.v1.json');expected=Path('benchmarks/acceptance.v1.sha256').read_text().split()[0];assert file_hash(acceptance_path)==expected
    acceptance=json.loads(acceptance_path.read_text());assert file_hash('benchmarks/manifest.json')==acceptance['splits_sha256']
    ref,obs,body,face,hands,timeline,p,rig,fmap,c,baseline=inputs(identifier);out=Path('runs/stage8')/identifier
    annotations=json.loads(Path(f'benchmarks/annotations/{identifier}.json').read_text())
    if phase=='solve':
        out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();write_json(out/'calibration.json',c)
        started=time.monotonic();b=solve_full(body,face,hands,timeline,p,rig,fmap,c);duration=time.monotonic()-started
        b.metadata.update(observations_arrays_sha256=file_hash(obs/'body/arrays.npz'),face_observations_sha256=file_hash(obs/'face/arrays.npz'),hand_observations_sha256=file_hash(obs/'hands/arrays.npz'))
        h=write_bundle(b,out/'animation');diag=b.metadata['temporal_diagnostics'];diag['full_solve_seconds']=duration;diag['python_peak_rss_bytes']=self_peak_bytes();write_json(out/'diagnostics.json',diag)
        pose,world=measure_pose(b,body,timeline,c,annotations);kin=measure_kinematics(b,world,timeline,annotations);metrics={**pose['metrics'],**kin['metrics']};g=gates(ref,metrics,kin,acceptance,timeline,diag)
        write_json(out/'root_trajectory.json',kin.pop('root_trajectory'));measurement={'id':identifier,'split':ref['split'],'status':'experimental_needs_review','metrics':metrics,'gates':g,'kinematics':kin,'pose':pose,'performance':diag,'baseline_performance':ref['performance'],'provenance':ref['provenance']|{'temporal_bundle_hash':h,'acceptance_sha256':expected},'quality_gain':'pending_visual_review'};write_json(out/'measurements.json',measurement)
        code_files=list(Path('src/xms/solve').glob('*.py'))+[Path(__file__)]
        write_json(out/'manifest.json',{'schema_version':'xms.stage8_run.v1','status':'experimental_needs_review','solver_exit_status':0 if diag['success'] else 1,'parameters':{'max_nfev':80,'single_window':True,'contacts_weight':0,'collision_weight':0,'camera_refinement':False},'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform()},'code_sha256':{str(path):file_hash(path) for path in code_files},'inputs':measurement['provenance'],'artifacts':{'bundle':str(out/'animation'),'measurements':str(out/'measurements.json'),'diagnostics':str(out/'diagnostics.json')}})
        print(identifier,json.dumps({'seconds':duration,'termination':diag['termination'],'metrics':{k:v['value'] for k,v in metrics.items()},'failed_gates':[k for k,v in g.items() if v['status']=='fail']}),flush=True)
    else:
        cfg=json.loads(Path('runs/stage6/config.json').read_text());source=next(s for s in json.loads(Path('benchmarks/manifest.json').read_text())['sources'] if s['id']==identifier)['path'];assert file_hash(source)==ref['provenance']['source_sha256']
        evidence={}
        for label,path in [('stage7',baseline),('stage8',out/'animation')]:
            frames=out/(label+'-frames');log=out/'logs'/(label+'-blender.log');print('RENDER',identifier,label,flush=True)
            evidence[label]=render(path,frames,cfg,timeline['fps'],log)
            video=out/(label+'.mp4');encode(frames,source,timeline,cfg['ffmpeg'],video,out/'logs'/(label+'-encode.log'));evidence[label]['verification']=verify(video,timeline,[540,960],cfg['ffmpeg'],cfg['ffprobe'],out/'logs'/(label+'-decode.log'))
            evidence[label]['bundle_hash']=bundle_hash(path);evidence[label]['video_sha256']=file_hash(video)
        probe=json.loads((obs/'probe.json').read_text());source_frames,_=decode(source,probe,timeline,cfg['ffmpeg'],out/'logs/source-decode.log')
        mapping=read_bundle(out/'animation').arrays['source_time_map'];old_map=read_bundle(baseline).arrays['source_time_map'];np.testing.assert_array_equal(mapping,old_map)
        indices=np.abs(np.asarray(timeline['source_times_s'])[:,None]-mapping).argmin(axis=0);np.testing.assert_allclose(np.asarray(timeline['source_times_s'])[indices],mapping,atol=1e-8)
        sar=float(Fraction(probe['sample_aspect_ratio'].replace(':','/')));sar=1/sar if probe['rotation_degrees']%180 else sar
        directory=out/'comparison-frames';directory.mkdir()
        for i,idx in enumerate(indices):
            sheet=Image.new('RGB',(1080,640));panels=[Image.fromarray(source_frames[idx])]+[Image.open(out/(label+'-frames')/f'{i:06d}.png').convert('RGB') for label in ('stage7','stage8')]
            for j,panel in enumerate(panels):sheet.paste(letterbox(panel,(360,640),sar if j==0 else 1),(j*360,0))
            draw=ImageDraw.Draw(sheet)
            for j,label in enumerate(['SOURCE','STAGE 7 / FROZEN BASELINE','STAGE 8 / EXPERIMENTAL']):draw.rectangle((j*360,0,j*360+360,24),fill='black');draw.text((j*360+8,7),f'{label} {mapping[i]:.3f}s',fill='white')
            sheet.save(directory/f'{i:06d}.png')
        encode(directory,source,timeline,cfg['ffmpeg'],out/'comparison.mp4',out/'logs/comparison-encode.log');evidence['comparison_verification']=verify(out/'comparison.mp4',timeline,[1080,640],cfg['ffmpeg'],cfg['ffprobe'],out/'logs/comparison-decode.log')
        evidence.update(source_pts_map=mapping.tolist(),camera_scene_config=cfg,comparison_sha256=file_hash(out/'comparison.mp4'),policy='Exact source PTS, same 60 bundle samples and output times; identical profile, scene, camera, materials, resolution, samples and render adapter.')
        write_json(out/'visual-evidence.json',evidence);print('COMPARISON',out/'comparison.mp4',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('id',choices=['sing','dance','non_neutral']);parser.add_argument('--phase',choices=['solve','render'],required=True);args=parser.parse_args();run(args.id,args.phase)
