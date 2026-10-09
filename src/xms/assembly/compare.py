"""Comparison samples the source's recorded PTS map, never stretches its duration."""
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from xms.ingest.decode import decode
from xms.animation.io import read_bundle,bundle_hash,file_hash
from .encode import encode
from xms.qa.contract import write_json


def letterbox(image,size,sar=1):
    if sar!=1:image=image.resize((max(1,round(image.width*sar)),image.height))
    return ImageOps.pad(image,size,method=Image.Resampling.LANCZOS,color=(20,20,20))


def compare(video,run,out,config):
    run,out=Path(run),Path(out);out.mkdir(parents=True,exist_ok=False)
    timeline=json.loads((run/'observations/timeline.json').read_text());probe=json.loads((run/'observations/probe.json').read_text());b=read_bundle(run/'animation')
    if file_hash(video)!=b.metadata['observations_source_sha256']:raise ValueError('Comparison source hash mismatch')
    frames,_=decode(video,probe,timeline,config['ffmpeg'],out/'decode.log')
    source=np.array(timeline['source_times_s']);mapping=b.arrays['source_time_map'];
    from xms.ingest.timeline import nearest_indices
    indices=nearest_indices(source,mapping)
    if not np.allclose(source[indices],mapping,atol=1e-8):raise ValueError('Comparison requires exact recorded source PTS')
    sar=probe['sample_aspect_ratio'];sar=float(Fraction(sar.replace(':','/'))) if sar not in ('N/A','0:1') else 1
    if probe['rotation_degrees']%180:sar=1/sar
    images=out/'frames';images.mkdir()
    for i,source_i in enumerate(indices):
        sheet=Image.new('RGB',(540,480));sheet.paste(letterbox(Image.fromarray(frames[source_i]),(270,480),sar),(0,0));sheet.paste(letterbox(Image.open(run/'frames'/f'{i:06d}.png').convert('RGB'),(270,480)),(270,0))
        d=ImageDraw.Draw(sheet);d.rectangle((0,0,540,22),fill='black');d.text((7,5),f'SOURCE {mapping[i]:.3f}s',fill='white');d.text((277,5),'XANDRA / CANDIDATE',fill='white');sheet.save(images/f'{i:06d}.png')
    encode(images,video,timeline,config['ffmpeg'],out/'compare.mp4',out/'encode.log')
    write_json(out/'manifest.json',{'schema_version':'xms.compare.v1','bundle_hash':bundle_hash(run/'animation'),'source_sha256':file_hash(video),'source_pts_map':mapping.tolist(),'frame_count':len(indices),'resolution':[540,480],'sample_aspect_ratio':'1:1','policy':'exact recorded source PTS; preserve image display aspect; same output times, no speed stretch','video_sha256':file_hash(out/'compare.mp4')})
    return out/'compare.mp4'
