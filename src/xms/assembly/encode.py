import subprocess,json
from pathlib import Path
from fractions import Fraction
from xms.qa.contract import measured_command,write_json


def command(cmd,log):
    r,performance=measured_command(cmd,capture_output=True,text=True,timeout=180)
    write_json(str(log)+'.runtime.json',performance)
    Path(log).write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('FFmpeg failed; see '+str(log))
    return r


def preflight(ffmpeg,resolution,fps,path,log):
    w,h=resolution
    command([ffmpeg,'-v','error','-f','lavfi','-i',f'color=size={w}x{h}:rate={fps}','-frames:v','1','-c:v','libx264','-pix_fmt','yuv420p',path],log)


def encode(frames,video,timeline,ffmpeg,out,log):
    rate=timeline['fps'];n=timeline['frame_count'];duration=float(Fraction(n)/Fraction(rate))
    cmd=[ffmpeg,'-v','error','-framerate',rate,'-start_number','0','-i',str(Path(frames)/'%06d.png')]
    if timeline['audio_start_s'] is not None:
        cmd+=['-i',str(video)];offset=timeline['audio_offset_from_clip_s'];trim=max(0.,-offset);delay=max(0.,offset)
        audio=f'[1:a:0]atrim=start={trim}:duration={max(0,duration-delay)},asetpts=PTS-STARTPTS,adelay={round(delay*1000)}:all=1,apad=whole_dur={duration},atrim=duration={duration}[audio]'
        cmd+=['-filter_complex',audio,'-map','0:v:0','-map','[audio]','-c:a','aac','-b:a','192k']
    else:cmd+=['-map','0:v:0','-an']
    cmd+=['-frames:v',str(n),'-c:v','libx264','-pix_fmt','yuv420p','-crf','20','-movflags','+faststart',str(out)]
    command(cmd,log)


def verify(video,timeline,resolution,ffmpeg,ffprobe,log):
    r=subprocess.run([ffprobe,'-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True,timeout=180)
    if r.returncode:raise RuntimeError(r.stderr)
    data=json.loads(r.stdout);v=next(s for s in data['streams'] if s['codec_type']=='video');audio=next((s for s in data['streams'] if s['codec_type']=='audio'),None)
    assert [v['width'],v['height']]==resolution,'Output size mismatch'
    assert int(v['nb_read_frames'])==timeline['frame_count'],'Output frame count mismatch'
    expected=timeline['frame_count']/float(Fraction(timeline['fps']));tolerance=1/float(Fraction(timeline['fps']))
    assert abs(float(v['duration'])-expected)<=tolerance,'Output duration mismatch'
    assert (audio is not None)==(timeline['audio_start_s'] is not None),'Audio presence mismatch'
    assert abs(float(v.get('start_time',0)))<=tolerance,'Video starts late'
    if audio:
        assert abs(float(audio.get('start_time',0)))<=tolerance,'Audio starts late'
        assert abs(float(audio['duration'])-expected)<=tolerance,'Audio duration mismatch'
    command([ffmpeg,'-v','error','-xerror','-i',video,'-f','null','-'],log)
    return {'full_decode_exit_status':0,'frame_count':int(v['nb_read_frames']),'resolution':resolution,'video_duration_s':float(v['duration']),'audio_present':audio is not None,'audio_start_s':float(audio.get('start_time',0)) if audio else None,'audio_duration_s':float(audio['duration']) if audio else None,'expected_duration_s':expected,'duration_tolerance_s':tolerance,'ffprobe':data}
