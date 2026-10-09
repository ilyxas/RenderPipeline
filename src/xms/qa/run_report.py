"""Read existing run, write report into a distinct directory; never mutate baseline."""
import json,hashlib
from pathlib import Path
from xms.animation.io import read_bundle,bundle_hash,file_hash
from xms.observations.io import read_observations
from .coverage import coverage_metrics
from .timing import timing_metrics
from .contract import metric,EXIT_CODES
from xms.report.html import report


def quality(run):
    run=Path(run);m=json.loads((run/'manifest.json').read_text())
    if m.get('status') in ('failed','needs_input'):return {'schema_version':'xms.quality.v1','status':m['status'],'exit_code':EXIT_CODES[m['status']],'metrics':{'execution':metric(False,passed=False)},'limitations':[m.get('error','Execution failed')],'provenance':{'run_manifest_sha256':file_hash(run/'manifest.json')}}
    b=read_bundle(run/'animation');t=json.loads((run/'observations/timeline.json').read_text())
    metrics=coverage_metrics(b,read_observations(run/'observations/body'));metrics.update(timing_metrics(t,m['verification']))
    if (run/'observations/face').exists():
        from xms.observations.details_io import read
        import numpy as np
        fm,fa=read(run/'observations/face');hm,ha=read(run/'observations/hands')
        metrics['face_tracking']=metric(float(np.mean(fa['validity'])),'fraction of all frames')
        metrics['hand_tracking']=metric(np.mean(ha['validity'],axis=0).tolist(),'left/right fraction of all frames')
        from .coverage import missing_intervals
        metrics['face_missing_intervals']=metric(missing_intervals(fa['validity'],fa['source_times_s']),'source seconds')
        for j,side in enumerate(('left','right')):metrics[side+'_hand_missing_intervals']=metric(missing_intervals(ha['validity'][:,j],ha['source_times_s']),'source seconds')
        metrics['visible_face_coverage']=metric(reason='Visible-face denominator requires independent annotation; all-frame tracking fraction is only a proxy')
        metrics['hand_identity_swaps']=metric(reason='Frame-local body-wrist association checked synthetically; no temporal identity ground truth')
    status='needs_review' if any(v['status'] in ('fail','unavailable') for v in metrics.values()) else 'warnings' if any(v['status']=='measured' for v in metrics.values()) else 'success'
    runtime={'python':m['python'],'versions':m.get('versions',{}),'blender':m['stages']['render']['blender_version']}
    return {'schema_version':'xms.quality.v1','quality':'baseline_preview','status':status,'exit_code':EXIT_CODES[status],'metrics':metrics,'limitations':m.get('limitations',[]),'provenance':{'run_manifest_sha256':file_hash(run/'manifest.json'),'input_hashes':m['input_hashes'],'character_profile_hash':m['character_profile_hash'],'scene_profile_hash':m['scene_profile_hash'],'bundle_hash':bundle_hash(run/'animation'),'runtime':runtime,'runtime_hash':hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()},'performance':m.get('performance',{})}


def write_report(run,out,comparison=None):
    q=quality(run);report(out,q,Path(run)/'outputs/video.mp4' if (Path(run)/'outputs/video.mp4').exists() else None,comparison);return q
