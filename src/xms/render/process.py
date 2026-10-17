import os,signal,subprocess,time
import re,sys
from pathlib import Path


def stop_tree(pid):
    children=subprocess.run(['pgrep','-P',str(pid)],capture_output=True,text=True).stdout.split()
    for child in children:stop_tree(int(child))
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass


def render(bundle,out,config,fps,log,indices=None,view='main',quality='preview',camera_plan=None,resolution=None,engine='eevee',samples=None):
    script=Path(__file__).resolve().parents[3]/'blender/render_frames.py'
    cmd=[config['blender'],'-b',config['character_asset'],'--python-exit-code','1','--python',str(script),'--','--bundle',str(Path(bundle).resolve()),'--profile',config['character_profile'],'--scene',config['scene_asset'],'--scene-profile',config['scene_profile'],'--out',str(Path(out).resolve()),'--fps',str(fps)]
    if indices is not None:cmd+=['--frame-indices',','.join(str(i) for i in indices)]
    cmd+=['--view',view,'--quality',quality,'--engine',engine]
    if camera_plan:cmd+=['--camera-plan',str(Path(camera_plan).resolve())]
    if resolution:cmd+=['--resolution',f'{resolution[0]}x{resolution[1]}']
    if samples:cmd+=['--samples',str(samples)]
    start=time.monotonic()
    measured=(['/usr/bin/time','-l'] if sys.platform=='darwin' else [])+cmd
    with open(log,'w') as stream:
        process=subprocess.Popen(measured,stdout=stream,stderr=subprocess.STDOUT)
        try:returncode=process.wait()
        except BaseException:
            # Interrupted: /usr/bin/time does not forward signals, so stop Blender beneath it too.
            stop_tree(process.pid);raise
    if returncode:raise RuntimeError(f'Blender exit {returncode}; see {log}')
    match=re.search(r'(\d+)\s+maximum resident set size',Path(log).read_text())
    return {'exit_status':returncode,'seconds':time.monotonic()-start,'peak_rss_bytes':int(match.group(1)) if match else None,'command':cmd}
