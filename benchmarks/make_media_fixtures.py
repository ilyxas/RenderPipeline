"""Small local derivatives for timing verification, never new benchmark identities."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from xms.animation.io import file_hash


def make(source,out,ffmpeg):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);rows=[]
    cases={'silent':['-an'], 'rational':['-r','30000/1001'], 'vfr':['-vf',"select=not(eq(mod(n\\,3)\\,1))",'-fps_mode','vfr'], 'offset':['-output_ts_offset','4.25'], 'sar':['-vf','setsar=4/3'], 'short':['-t','0.12']}
    for name,opts in cases.items():
        path=out/(name+'.mp4');cmd=[ffmpeg,'-v','error','-ss','3','-i',str(source),'-t','0.75','-map','0:v:0','-map','0:a:0?','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',*opts,str(path)]
        subprocess.run(cmd,check=True);rows.append({'id':name,'path':str(path.resolve()),'sha256':file_hash(path),'command':cmd})
    # Store portrait pixels clockwise in a landscape frame, with inverse display matrix.
    physical=out/'rotated-pixels.mp4';subprocess.run([ffmpeg,'-v','error','-ss','3','-i',str(source),'-t','0.75','-map','0:v:0','-map','0:a:0?','-vf','transpose=clock','-c:v','libx264','-c:a','aac',str(physical)],check=True)
    path=out/'rotated.mp4';subprocess.run([ffmpeg,'-v','error','-display_rotation:v:0','90','-i',str(physical),'-c','copy',str(path)],check=True);rows.append({'id':'rotated','path':str(path.resolve()),'sha256':file_hash(path)})
    invalid=out/'invalid.mp4';invalid.write_bytes(b'not a valid media file\n');rows.append({'id':'invalid','path':str(invalid.resolve()),'sha256':file_hash(invalid)})
    m={'schema_version':'xms.media_fixtures.v1','source_sha256':file_hash(source),'cases':rows,'policy':'Same-source technical fixtures only; never holdout material'};(out/'manifest.json').write_text(json.dumps(m,indent=2));return m


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',required=True);p.add_argument('--ffmpeg',required=True);a=p.parse_args();make(a.source,a.out,a.ffmpeg)
