"""Run the eight supplied clips, assess each published render, retain evidence.
No auto-acceptance from execution success or unavailable ground truth.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle,file_hash
from xms.animation.fk import evaluate
from xms.solve.grounding import sole_proxies,sole_positions
from xms.qa.contacts import measure as contact_measure
from xms.observations.io import read_observations

ROOT=Path(__file__).resolve().parents[1]

def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def source_motion(run,b,fk):
    meta,obs=read_observations(run/'observations/body')
    timeline=json.loads((run/'observations/timeline.json').read_text())
    idx=np.array(timeline['output_observation_indices']);xy=obs['image_landmarks_raw'][idx,:,:2]*[meta['width']/meta['height'],-1]
    names=b.metadata['joint_names'];results={}
    for side,s,e,w,h,k,f in [('l',11,13,15,23,25,27),('r',12,14,16,24,26,28)]:
        for bone,child,p,q in [('upperarm','lowerarm',s,e),('lowerarm','hand',e,w),('thigh','calf',h,k),('calf','foot',k,f)]:
            actual=fk[:,names.index(child+'_'+side),:2,3]-fk[:,names.index(bone+'_'+side),:2,3]
            expected=xy[:,q]-xy[:,p];length=np.linalg.norm(expected,axis=1);target_length=np.linalg.norm(actual,axis=1)
            visible=obs['landmark_validity'][idx,p]&obs['landmark_validity'][idx,q]&(length>.015)&(target_length>.03)
            angles=np.degrees(np.arccos(np.clip(np.sum(actual*expected,axis=1)/np.maximum(length*target_length,1e-9),-1,1)))
            values=angles[visible]
            results[bone+'_'+side]={'visible_samples':int(visible.sum()),'median_direction_error_deg':float(np.median(values)) if len(values) else None,'p95_direction_error_deg':float(np.percentile(values,95)) if len(values) else None,'worst_frame':int(np.argmax(np.where(visible,angles,-1)))}
    return {'reference':'input image landmarks; orthographic XY bone directions, not 3D ground truth','segments':results,'source_frames':len(idx),'unique_source_frames':len(set(idx.tolist()))}

def assess(run,acceptance):
    m=json.loads((run/'manifest.json').read_text())
    if m.get('result_exit_code')!=0:return {'execution':'failed','quality_accepted':False,'error':m.get('error'),'visual_review':'not_rendered'}
    b=read_bundle(run/'animation');a=b.arrays;n=len(a['sample_times_s']);fk=np.array([evaluate(b,i) for i in range(n)])
    neutral=copy.deepcopy(b);neutral.arrays['root_translation'][:]=0;neutral.arrays['local_rotation_delta'][:]=[0,0,0,1];points=sole_proxies(b,evaluate(neutral,0))
    heights=np.array([sole_positions(f,points)[:,:,1].min(axis=1) for f in fk]);g=b.metadata.get('grounding_diagnostics',{});supported=np.array(g.get('support_sample_indices',[]),int)
    q=a['local_rotation_delta'];angles=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],axis=-1)),0,1)));observed=a['rotation_validity'][1:]&a['rotation_validity'][:-1]
    worst=np.unravel_index(angles.argmax(),angles.shape);observed_max=float(angles[observed].max(initial=0));root_max=float(np.linalg.norm(np.diff(a['root_translation'],axis=0),axis=1).max(initial=0));norm=float(np.abs(np.linalg.norm(q,axis=-1)-1).max(initial=0));v=m['verification'];fps=n/v['expected_duration_s'];limits=acceptance['continuity'];arch=acceptance['final_architecture_goals'];contacts=contact_measure(b);surface=m['stages'].get('surface',{})
    motion=source_motion(run,b,fk)
    checks={'source_frame_mapping_unique':motion['source_frames']==motion['unique_source_frames'],'full_decode':v['full_decode_exit_status']==0,'frame_count':v['frame_count']==n,'resolution':v['resolution']==[720,1280],'duration':abs(v['video_duration_s']-v['expected_duration_s'])<=arch['duration_error_output_frames']/fps,'quaternion_norm':norm<=limits['quaternion_unit_error_max'],'observed_rotation_continuity':observed_max<=limits['observed_adjacent_rotation_jump_max_degrees'],'root_continuity':root_max<=limits['root_adjacent_step_max_m']}
    checks['foot_skating_proxy']=None if contacts['median_segment_max_drift_m'] is None else contacts['median_segment_max_drift_m']<=arch['foot_skating_median_max_drift_m'] and contacts['p95_segment_max_drift_m']<=arch['foot_skating_p95_drift_m']
    checks['surface_penetration']=None if not surface.get('signed_interior_available') or surface.get('sample_coverage')!=1 else surface.get('max_candidate_depth_m',float('inf'))<=arch['surface_penetration_max_m'] and (surface.get('p95_candidate_depth_m') or 0)<=arch['surface_penetration_p95_m']
    checks['hand_identity_ground_truth']=None;checks['visible_face_coverage_ground_truth']=None;checks['lip_event_ground_truth']=None
    return {'execution':'passed','quality_accepted':False,'automated_status':'failed' if any(x is False for x in checks.values()) else 'unavailable' if any(x is None for x in checks.values()) else 'needs_visual_review','checks':checks,'source_motion':motion,'continuity':{'max_observed_rotation_step_deg':observed_max,'max_all_rotation_step_deg':float(angles.max()),'worst_frame':int(worst[0]+1),'worst_joint':b.metadata['joint_names'][worst[1]],'max_root_step_m':root_max,'quaternion_norm_error':norm},'floor_proxy':{'supported_samples':len(supported),'flight_samples':len(g.get('flight_sample_indices',[])),'unsupported_samples':g.get('unsupported_samples'),'max_supported_gap_m':float(np.abs(heights[supported].min(axis=1)).max(initial=0)) if len(supported) else None,'max_all_penetration_m':float(np.maximum(-heights,0).max(initial=0)),'correction_bound_samples':g.get('correction_bound_samples'),'geometry':'rest-derived sole proxies; skin clearance unavailable'},'contacts':contacts,'surface':{k:surface.get(k) for k in ['status','sample_coverage','signed_interior_available','max_candidate_depth_m','p95_candidate_depth_m']},'verification':{k:v[k] for k in ['frame_count','resolution','video_duration_s','audio_present','full_decode_exit_status']},'visual_review':'pending','limitations':m.get('limitations',[]),'warnings':m.get('warnings',[]),'outputs':m['outputs']}

def evidence(run,result):
    if result['execution']!='passed':return
    m=json.loads((run/'manifest.json').read_text());p=run/'validation';p.mkdir(exist_ok=True);n=result['verification']['frame_count']
    indices=sorted(set(round(x) for x in np.linspace(0,n-1,6)));worst=result['continuity']['worst_frame'];near=list(range(max(0,worst-2),min(n,worst+3)))
    for name,selected in [('overview',indices),('worst-transition',near)]:
        expr='+'.join(f'eq(n\\,{i})' for i in selected);vf=f'select={expr},scale=720:-1,tile=2x{(len(selected)+1)//2}'
        r=subprocess.run([m['config']['ffmpeg'],'-v','error','-i',m['outputs']['compare'],'-vf',vf,'-frames:v','1',str(p/(name+'.jpg'))],capture_output=True,text=True)
        if r.returncode:result.setdefault('evidence_errors',[]).append(r.stderr[-1000:])
    result['evidence_frame_indices']={'overview':indices,'worst_transition':near}

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',default=str(ROOT/'base video'));p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    clips=sorted(Path(args.inputs).resolve().glob('*.mp4'))
    if len(clips)!=8:raise ValueError(f'Expected exactly eight videos, found {len(clips)}')
    acceptance_path=ROOT/'benchmarks/acceptance.v1.json';acceptance=json.loads(acceptance_path.read_text());revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    implementation_files=sorted((ROOT/'src').rglob('*.py'));implementation_hash=hashlib.sha256(''.join(str(p.relative_to(ROOT))+file_hash(p) for p in implementation_files).encode()).hexdigest()
    suite={'implementation_sha256':implementation_hash,'revision':revision,'settings':{'solver':'video','channels':'full','quality':'preview','resolution':'720x1280','surface_qa':'all'},'acceptance_file':str(acceptance_path),'acceptance_sha256':file_hash(acceptance_path),'baseline_profile_differs':True,'quality_accepted':False,'clips':[]};save(out/'suite.json',suite)
    for number,source in enumerate(clips,1):
        clip_id=source.name.split('generated_')[1][:8] if 'generated_' in source.name else source.stem;entry={'number':number,'id':clip_id,'input':str(source),'input_sha256':file_hash(source),'status':'running'};suite['clips'].append(entry);save(out/'suite.json',suite);print(f'CLIP {number}/8 {clip_id} START',flush=True)
        runs=out/(f'{number:02d}-{clip_id}');runs.mkdir();log=runs/'pipeline.log'
        cmd=[sys.executable,'-u','-m','xms.cli','run',str(source),'--runs-root',str(runs),'--quality','preview','--resolution','720x1280','--solver','video','--channels','full','--surface-qa','all']
        with log.open('w') as stream:process=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        manifests=sorted(runs.glob('*/manifest.json'));run=manifests[-1].parent if manifests else None
        try:
            result=assess(run,acceptance) if run else {'execution':'failed','quality_accepted':False,'error':'No run manifest','visual_review':'not_rendered'}
            if run:evidence(run,result);(run/'validation').mkdir(exist_ok=True);save(run/'validation/result.json',result)
            entry.update(status=result['execution'],run_dir=str(run) if run else None,validation=result)
        except Exception as error:entry.update(status='validation_failed',error=str(error),quality_accepted=False)
        save(out/'suite.json',suite);print(f'CLIP {number}/8 {clip_id} DONE {entry["status"]}',flush=True)
    suite['status']='rendered_pending_visual_review';save(out/'suite.json',suite);print('SUITE '+str(out/'suite.json'),flush=True)

if __name__=='__main__':main()
