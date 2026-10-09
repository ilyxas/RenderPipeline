"""Audio cues aligned through the same source PTS offset as soundtrack mux."""
import json,subprocess,wave
from pathlib import Path
import numpy as np
from xms.audio.envelope import extract


def observe(video,timeline,config,out,mode='envelope'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);times=np.array(timeline['output_times_s']);warnings=[]
    if timeline['audio_start_s'] is None or mode=='off':
        result={'envelope':np.zeros(len(times)),'voicing':np.zeros(len(times),bool),'confidence':np.zeros(len(times)),'policy':'silent or audio disabled'};cues=[]
    else:
        wav=out/'aligned.wav';offset=timeline['audio_offset_from_clip_s'];trim=max(0.,-offset);delay=max(0.,offset)
        filt=f'atrim=start={trim},asetpts=PTS-STARTPTS,adelay={round(delay*1000)}:all=1,apad,atrim=duration={timeline["duration_s"]}'
        subprocess.run([config['ffmpeg'],'-v','error','-i',str(video),'-vn','-af',filt,'-ar','16000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True,capture_output=True,timeout=60)
        cue_wav=wav;cues=[]
        if mode=='singing':
            try:
                from xms.audio.demucs import separate
                cue_wav=separate(wav,config.get('demucs_python'),out/'separation')
            except (RuntimeError,subprocess.TimeoutExpired) as e:warnings.append(str(e)+'; mixed envelope remains low-confidence')
        with wave.open(str(cue_wav)) as stream:samples=np.frombuffer(stream.readframes(stream.getnframes()),dtype='<i2').astype(float)/32768;rate=stream.getframerate()
        result=extract(samples,rate,times)
        if mode=='speech':
            try:
                from xms.audio.rhubarb import cues as speech_cues
                cues=speech_cues(wav,config.get('rhubarb','rhubarb'))
            except (OSError,RuntimeError,subprocess.TimeoutExpired) as e:warnings.append(str(e)+'; video/envelope fallback')
    result['cues']=cues;result['times']=times
    np.savez_compressed(out/'arrays.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    metadata={'schema_version':'xms.audio_observations.v1','mode':mode,'alignment_correction_s':0,'source_audio_offset_s':timeline['audio_offset_from_clip_s'],'cues':cues,'warnings':warnings,'policy':result['policy']}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2));result['metadata']=metadata;return result
