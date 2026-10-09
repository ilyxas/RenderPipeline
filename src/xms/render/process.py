import subprocess,time
import re,sys
from pathlib import Path


def render(bundle,out,config,fps,log,indices=None,view='main',quality='preview',camera_plan=None):
    script=Path(__file__).resolve().parents[3]/'blender/render_frames.py'
    cmd=[config['blender'],'-b',config['character_asset'],'--python-exit-code','1','--python',str(script),'--','--bundle',str(Path(bundle).resolve()),'--profile',config['character_profile'],'--scene',config['scene_asset'],'--scene-profile',config['scene_profile'],'--out',str(Path(out).resolve()),'--fps',str(fps)]
    if indices is not None:cmd+=['--frame-indices',','.join(str(i) for i in indices)]
    cmd+=['--view',view,'--quality',quality]
    if camera_plan:cmd+=['--camera-plan',str(Path(camera_plan).resolve())]
    start=time.monotonic()
    measured=(['/usr/bin/time','-l'] if sys.platform=='darwin' else [])+cmd
    with open(log,'w') as stream:r=subprocess.run(measured,stdout=stream,stderr=subprocess.STDOUT,timeout=None)
    if r.returncode:raise RuntimeError(f'Blender exit {r.returncode}; see {log}')
    match=re.search(r'(\d+)\s+maximum resident set size',Path(log).read_text())
    return {'exit_status':r.returncode,'seconds':time.monotonic()-start,'peak_rss_bytes':int(match.group(1)) if match else None,'command':cmd}
