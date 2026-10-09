"""Focused ingest/PTS/encode/audio-content matrix, using tiny frames, no Blender."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from xms.ingest.probe import probe
from xms.ingest.timeline import make_timeline
from xms.ingest.decode import decode
from xms.assembly.encode import encode,verify
from xms.qa.timing import timing_metrics
from xms.qa.contract import write_json


def check(fixtures,out,ffmpeg,ffprobe,only=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);cases=json.loads(Path(fixtures).read_text())['cases'];rows=[]
    for case in cases:
        if only and case['id'] not in only.split(','):continue
        target=out/case['id'];target.mkdir()
        if case['id']=='invalid':
            try:probe(case['path'],ffprobe)
            except (ValueError,subprocess.CalledProcessError):rows.append({'id':'invalid','status':'passed','rejected_before_render':True});continue
            raise AssertionError('Invalid media accepted')
        info=probe(case['path'],ffprobe);duration=.083 if case['id']=='short' else .5;t=make_timeline(info,0,duration);frames,transform=decode(case['path'],info,t,ffmpeg,target/'decode.log')
        imgs=target/'frames';imgs.mkdir()
        for i,j in enumerate(t['output_observation_indices']):Image.fromarray(frames[j]).resize((180,320)).save(imgs/f'{i:06d}.png')
        encode(imgs,case['path'],t,ffmpeg,target/'output.mp4',target/'encode.log');v=verify(target/'output.mp4',t,[180,320],ffmpeg,ffprobe,target/'verify.log');metrics=timing_metrics(t,v)
        assert all(x['status']!='fail' for x in metrics.values()),metrics
        if case['id']=='rotated':assert frames.shape[1:3]==(1280,720)
        if case['id']=='rational':assert t['fps']=='30000/1001'
        if case['id']=='offset':assert t['video_start_s']>4
        if case['id']=='vfr':assert info['is_vfr']
        row={'id':case['id'],'status':'passed','timing':metrics,'timeline':t,'audio_present':v['audio_present'],'rotation_transform':transform.tolist()}
        # Signal alignment test: decoded expected source audio vs encoded/muxed output.
        if v['audio_present']:
            trim=max(0,-t['audio_offset_from_clip_s']);duration=t['frame_count']/float(__import__('fractions').Fraction(t['fps']))
            def audio(path,filter_):return np.frombuffer(subprocess.check_output([ffmpeg,'-v','error','-i',str(path),'-map','0:a:0','-af',filter_,'-ar','8000','-ac','1','-f','f32le','pipe:1']),dtype=np.float32)
            expected=audio(case['path'],f'atrim=start={trim}:duration={duration},asetpts=PTS-STARTPTS');actual=audio(target/'output.mp4',f'atrim=duration={duration}')
            count=min(len(expected),len(actual));expected=expected[:count];actual=actual[:count];lags=range(-500,501)
            scores=[float(np.dot(expected[max(0,l):min(count,count+l)],actual[max(0,-l):min(count,count-l)])) for l in lags]
            lag=list(lags)[int(np.argmax(scores))]/8000
            assert abs(lag)<=1/float(__import__('fractions').Fraction(t['fps'])),lag
            row['audio_content_lag_s']=lag
        rows.append(row);write_json(target/'result.json',row)
    write_json(out/'results.json',{'cases':rows,'status':'passed'});print('PASS',len(rows),'media cases')


if __name__=='__main__':check(*sys.argv[1:])
