"""Surface evidence with explicit coverage and sign limitations."""
import json,subprocess,time
from pathlib import Path
from xms.qa.contract import self_peak_bytes


def assess(records,planned,selected):
    depths=[depth for row in records for depth in row.get('candidate_depths_m',[])]
    import numpy as np
    max_depth=max(depths,default=0);p95=float(np.percentile(depths,95)) if depths else None
    complete=set(planned)==set(selected);signed=bool(records) and all(x.get('signed_interior_available',False) and x.get('intersection_coverage_complete',False) for x in records)
    status='needs_review' if max_depth>.005 or (p95 is not None and p95>.002) else 'unavailable' if not signed or not complete else 'passed'
    return {'status':status,'max_candidate_depth_m':max_depth,'p95_candidate_depth_m':p95,'sample_coverage':len(selected)/len(planned),'signed_interior_available':signed,'surface_quality_accepted':status=='passed','excluded':['hair','finger–finger','internal overlapping skin variants'],'limitations':['Open garment meshes use nearest-normal candidate depths, not certified penetration.','Acceptance covers declared hand–garment pairs only; other body surfaces are excluded.'],'records':records,'refinement_passes':0}


def evaluate(bundle,config,out,mode='sampled'):
    from xms.animation.io import read_bundle
    b=read_bundle(bundle);n=len(b.arrays['sample_times_s']);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    indices=list(range(n)) if mode=='all' else sorted(set(round(x) for x in __import__('numpy').linspace(0,n-1,min(8,n))))
    script=Path(__file__).resolve().parents[3]/'blender/evaluate_surfaces.py';cmd=[config['blender'],'-b',config['character_asset'],'--python-exit-code','1','--python',str(script),'--','--bundle',str(Path(bundle).resolve()),'--profile',config['character_profile'],'--out',str(out.resolve()),'--indices',','.join(map(str,indices))]
    start=time.monotonic()
    with (out/'blender.log').open('w') as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
    if r.returncode:raise RuntimeError(f'Surface evaluator exit {r.returncode}; see {out}/blender.log')
    result=json.loads((out/'quality.json').read_text());result.update(seconds=time.monotonic()-start,mode=mode);(out/'quality.json').write_text(json.dumps(result,indent=2));return result
