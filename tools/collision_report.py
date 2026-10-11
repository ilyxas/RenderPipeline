"""Collision experiment report for one run: penetration, fidelity, velocity, refinement outcome.

Usage: PYTHONPATH=src .venv/bin/python tools/collision_report.py runs/<...>/<run-id>
Writes collision_report.json next to the manifest and prints a short table.
"""
import json,sys
from pathlib import Path
import numpy as np
from xms.animation.io import read_bundle
from xms.animation.fk import evaluate


def kind(row):
    if row.get('kind')=='skin_witness':return 'skin_arm_arm' if 'arm_' in row['target'] else 'skin_body'
    return 'garment_candidate'


def penetration(quality):
    out={}
    for row in quality.get('records',[]):
        k=kind(row);depth=max(row.get('candidate_depths_m',[]),default=0.);out.setdefault(k,{})
        out[k][row['sample']]=max(out[k].get(row['sample'],0.),depth)
    summary={}
    for k,frames in out.items():
        d=np.array(list(frames.values()))*1000
        summary[k]={'frames_checked':len(d),'frames_over_5mm':int((d>5).sum()),'max_mm':round(float(d.max()),1) if len(d) else 0,'p95_mm':round(float(np.percentile(d,95)),1) if len(d) else 0}
    return summary


def arm_steps(bundle):
    names=bundle.metadata['joint_names'];ids=[names.index(f'{x}_{s}') for x in ('upperarm','lowerarm','hand') for s in 'lr'];q=bundle.arrays['local_rotation_delta'][:,ids]
    return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],-1)),0,1)))


def fidelity(before,after):
    names=before.metadata['joint_names'];joints=['hand_l','hand_r','lowerarm_l','lowerarm_r'];n=len(before.arrays['sample_times_s'])
    d=np.array([[np.linalg.norm(evaluate(before,i)[names.index(j),:3,3]-evaluate(after,i)[names.index(j),:3,3]) for j in joints] for i in range(n)])*100
    changed=d.max(axis=1)>1e-6
    return {'frames_changed':int(changed.sum()),**{f'{j}_max_cm':round(float(d[:,k].max()),2) for k,j in enumerate(joints)}}


def report(run):
    run=Path(run);m=json.loads((run/'manifest.json').read_text());stages=m['stages']
    before=read_bundle(run/'animation-candidate');result={'run':str(run),'status':m.get('status'),'surface_status_before':stages.get('surface',{}).get('status')}
    q0=json.loads((run/'surface/quality.json').read_text()) if (run/'surface/quality.json').exists() else {}
    result['penetration_before']=penetration(q0);refinement=stages.get('surface_refinement',{'passes':0});result['refinement']={k:refinement.get(k) for k in ('passes','accepted')}
    if (run/'surface/quality.json').exists() and not refinement.get('passes'):result['refinement']['skipped_samples']=len(refinement.get('skipped_samples',[]))
    steps0=arm_steps(before);result['velocity_before']={'max_arm_step_deg':round(float(steps0.max()),1),'p95_arm_step_deg':round(float(np.percentile(steps0,95)),1)}
    if (run/'animation-refined').exists():
        after=read_bundle(run/'animation-refined');meta=after.metadata.get('surface_refinement',{})
        result['refinement'].update(corrected_samples=len(meta.get('samples',[])),max_correction_deg=round(max((x['max_correction_degrees'] for x in meta.get('samples',[])),default=0),2))
        q1=json.loads((run/'surface-refined/quality.json').read_text()) if (run/'surface-refined/quality.json').exists() else {}
        result['penetration_after']=penetration(q1);result['fidelity']=fidelity(before,after)
        steps1=arm_steps(after);result['velocity_after']={'max_arm_step_deg':round(float(steps1.max()),1),'p95_arm_step_deg':round(float(np.percentile(steps1,95)),1),'max_added_step_deg':round(float((steps1-steps0).max()),1)}
    (run/'collision_report.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':print(json.dumps(report(sys.argv[1]),indent=2))
