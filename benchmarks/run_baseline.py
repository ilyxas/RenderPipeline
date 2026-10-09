"""Frozen observations/profile/calibration -> repeatable baseline measurements.

No tracking, rendering, temporal optimization, caching or job scheduling.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from xms.animation.io import file_hash,write_bundle
from xms.observations.io import read_observations
from xms.observations.details_io import read
from xms.profiles.character import load_character
from xms.solve.calibration import calibrate,array_digest
from xms.solve.baseline_full import solve_full
from xms.qa.pose import measure_pose
from xms.qa.kinematics import measure_kinematics
from xms.qa.coverage import missing_intervals
from xms.qa.contract import metric,self_peak_bytes,write_json
from xms.report.html import report


def run(identifier,observations,profile,out,acceptance_path='benchmarks/acceptance.v1.json'):
    acceptance=json.loads(Path(acceptance_path).read_text());expected=Path('benchmarks/acceptance.v1.sha256').read_text().split()[0]
    if file_hash(acceptance_path)!=expected:raise ValueError('Frozen acceptance criteria changed')
    manifest=json.loads(Path('benchmarks/manifest.json').read_text());source=next(s for s in manifest['sources'] if s['id']==identifier)
    if file_hash('benchmarks/manifest.json')!=acceptance['splits_sha256']:raise ValueError('Frozen split manifest changed')
    body=read_observations(Path(observations)/'body');face=read(Path(observations)/'face');hands=read(Path(observations)/'hands');timeline=json.loads((Path(observations)/'timeline.json').read_text());p,rig,fmap=load_character(profile);annotations=json.loads(Path(f'benchmarks/annotations/{identifier}.json').read_text())
    if body[0]['source_sha256']!=source['sha256'] or annotations['source_sha256']!=source['sha256']:raise ValueError('Benchmark source hash mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);c=calibrate(body,timeline,p,rig);write_json(out/'calibration.json',c)
    c=json.loads((out/'calibration.json').read_text());started=time.monotonic();b=solve_full(body,face,hands,timeline,p,rig,fmap,c);duration=time.monotonic()-started;peak=self_peak_bytes()
    b.metadata['observations_arrays_sha256']=file_hash(Path(observations)/'body/arrays.npz');b.metadata['face_observations_sha256']=file_hash(Path(observations)/'face/arrays.npz');b.metadata['hand_observations_sha256']=file_hash(Path(observations)/'hands/arrays.npz')
    h=write_bundle(b,out/'animation')
    repeated=solve_full(body,face,hands,timeline,p,rig,fmap,c)
    errors={key:float(np.max(np.abs(value.astype(float)-repeated.arrays[key].astype(float)))) if value.size else 0 for key,value in b.arrays.items()}
    maximum=max(errors.values());assert maximum<=acceptance['continuity']['repeat_solve_array_atol']
    pose,world=measure_pose(b,body,timeline,c,annotations);kin=measure_kinematics(b,world,timeline,annotations);metrics={**pose['metrics'],**kin['metrics']}
    absolute=np.array(timeline['source_times_s']);visible=np.zeros(len(absolute),bool)
    for start,end in annotations['visible_face_intervals']:visible|=(absolute>=start)&(absolute<end)
    coverage=float(np.mean(face[1]['validity'][visible])) if visible.any() else None
    metrics['visible_face_coverage']=metric(coverage,'fraction',reason='No independently visible face interval',passed=None if coverage is None else coverage>=acceptance['final_architecture_goals']['visible_face_coverage_min'])
    metrics['left_hand_coverage']=metric(float(np.mean(hands[1]['validity'][:,0])),'fraction');metrics['right_hand_coverage']=metric(float(np.mean(hands[1]['validity'][:,1])),'fraction');metrics['head_face_owned_fraction']=metric(float(np.mean(np.array(b.metadata['head_source_by_sample'])=='face')),'fraction')
    metrics['lip_sync']=metric(reason='Audio event correspondence and geometric lip closure not measured',unit='s')
    mouth=[];indices=np.array(timeline['output_observation_indices']);source_times=b.arrays['source_time_map']
    if annotations['mouth_events']:
        policy=annotations['mouth_proxy_policy'];j=b.metadata['face_channel_names'].index(policy['channel']);weights=b.arrays['face_weights'][:,j];valid=b.arrays['face_validity'][:,j]
        for event in annotations['mouth_events']:
            threshold=policy[event['kind']+'_threshold'];state=(weights<=threshold) if event['kind']=='closure' else (weights>=threshold);candidates=np.flatnonzero(state[1:]&~state[:-1]&valid[1:]&valid[:-1])+1
            midpoint=(event['start_source_s']+event['end_source_s'])/2
            close=[i for i in candidates if abs(source_times[i]-midpoint)<=.3]
            predicted=float(source_times[min(close,key=lambda i:abs(source_times[i]-midpoint))]) if close else None
            error=None if predicted is None else max(event['start_source_s']-predicted,0,predicted-event['end_source_s'])
            mouth.append({**event,'predicted_channel_crossing_s':predicted,'distance_to_annotated_bracket_s':error})
        measured=[x['distance_to_annotated_bracket_s'] for x in mouth if x['distance_to_annotated_bracket_s'] is not None]
        metrics['mouth_event_channel_proxy_median_error']=metric(float(np.median(measured)) if measured else None,'s',reason='No matching channel crossings');metrics['mouth_event_channel_proxy_matched_fraction']=metric(len(measured)/len(mouth),'fraction')
    else:metrics['mouth_event_channel_proxy_median_error']=metric(reason=annotations.get('mouth_event_note','No independent annotations'),unit='s')
    continuity={}
    for metric_name,key in [('observed_adjacent_rotation_jump_max','observed_adjacent_rotation_jump_max_degrees'),('root_adjacent_step_max','root_adjacent_step_max_m')]:
        value=metrics[metric_name]['value'];continuity[metric_name]={'status':'unavailable' if value is None else 'pass' if value<=acceptance['continuity'][key] else 'fail','value':value,'threshold':acceptance['continuity'][key]}
    measurement={'schema_version':'xms.baseline_measurements.v1','id':identifier,'split':source['split'],'status':'needs_review','metrics':metrics,'continuity_gates':continuity,'repeat_solve':{'status':'pass','maximum_array_difference':maximum,'atol':acceptance['continuity']['repeat_solve_array_atol']},'performance':{'solve_wall_s':duration,'python_peak_rss_bytes':peak},'calibration_confidence':c['confidence'],'missing_face_intervals':missing_intervals(face[1]['validity'],absolute),'missing_hand_intervals':{side:missing_intervals(hands[1]['validity'][:,j],absolute) for j,side in enumerate(['left','right'])},'mouth_events':mouth,'pose':pose,'kinematics':{k:v for k,v in kin.items() if k!='root_trajectory'},'provenance':{'source_sha256':source['sha256'],'body_observations_digest':array_digest(body[1]),'face_observations_digest':array_digest(face[1]),'hand_observations_digest':array_digest(hands[1]),'character_profile_hash':p['profile_hash'],'bundle_hash':h,'calibration_sha256':file_hash(out/'calibration.json'),'annotations_sha256':file_hash(f'benchmarks/annotations/{identifier}.json'),'acceptance_sha256':expected,'source_observations_path':str(Path(observations).resolve())},'limitations':b.metadata['limitations']+['Reference camera and source landmarks are monocular priors/proxies, not ground truth.','No temporal solver has been compared; measured baseline does not establish quality gain.']}
    write_json(out/'root_trajectory.json',kin['root_trajectory']);write_json(out/'measurements.json',measurement);report(out/'report',measurement)
    # Small frozen numeric artifacts belong in benchmarks; large arrays/videos stay in runs.
    dest=Path('benchmarks/baseline')/identifier
    if not dest.exists():
        dest.mkdir(parents=True,exist_ok=False)
        for name in ('calibration.json','measurements.json','root_trajectory.json'):(dest/name).write_bytes((out/name).read_bytes())
    # Later reproductions stay in their requested output; never overwrite the frozen reference.
    print(identifier,'reprojection',metrics['normalized_reprojection_median']['value'],'repeat',maximum,flush=True);return measurement


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('id');p.add_argument('--observations',required=True);p.add_argument('--profile',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.id,a.observations,a.profile,a.out)
