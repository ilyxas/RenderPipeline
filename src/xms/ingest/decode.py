import re,tempfile,os
import subprocess
from pathlib import Path
import numpy as np
from xms.qa.contract import measured_command,write_json


def decode(path,probe,timeline,ffmpeg,log_path):
    w,h=probe['video']['width'],probe['video']['height']
    start,end=timeline['source_frame_indices'][0],timeline['source_frame_indices'][-1]
    # select by decoded frame order, verify actual integer PTS independently from showinfo.
    raw=tempfile.NamedTemporaryFile(prefix='xms-decode-',suffix='.rgb',dir=Path(log_path).parent,delete=False);raw.close()
    command=[str(ffmpeg),'-y','-hide_banner','-copyts','-noautorotate','-i',str(path),'-map',f"0:{probe['video']['index']}",'-vf',f"select=between(n\\,{start}\\,{end}),showinfo",'-fps_mode','passthrough','-f','rawvideo','-pix_fmt','rgb24',raw.name]
    r,performance=measured_command(command,capture_output=True)
    write_json(str(log_path)+'.runtime.json',performance)
    log=r.stderr.decode(errors='replace');Path(log_path).write_text(log)
    if r.returncode:
        os.unlink(raw.name);raise RuntimeError('Decode failed: '+str(log_path))
    decoded_pts=[int(x) for x in re.findall(r'\bn:\s*\d+\s+pts:\s*(-?\d+)',log)]
    if decoded_pts!=timeline['source_pts']:
        os.unlink(raw.name);raise ValueError('Decoded PTS differ from timeline')
    n=len(decoded_pts)
    if os.path.getsize(raw.name)!=n*w*h*3:
        os.unlink(raw.name);raise ValueError('Decoded frame size/count mismatch')
    frames=np.memmap(raw.name,dtype=np.uint8,mode='r',shape=(n,h,w,3));os.unlink(raw.name)
    rotation=probe['rotation_degrees']%360
    # FFprobe display matrix positive angle uses counterclockwise display correction.
    if rotation:frames=np.rot90(frames,k=rotation//90,axes=(1,2))
    transforms={0:[[1,0,0],[0,1,0],[0,0,1]],90:[[0,1,0],[-1,0,1],[0,0,1]],180:[[-1,0,1],[0,-1,1],[0,0,1]],270:[[0,-1,1],[1,0,0],[0,0,1]]}
    return frames,np.array(transforms[rotation],dtype=float)
