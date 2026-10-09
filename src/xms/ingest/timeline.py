from fractions import Fraction
import math
import numpy as np


def make_timeline(probe,start,end,fps=None):
    start,end=Fraction(str(start)),Fraction(str(end))
    if start<0 or end<=start:raise ValueError('Invalid [start,end) interval')
    rate=Fraction(fps or probe['fps']);tb=Fraction(probe['time_base']);pts=probe['pts'];origin=Fraction(pts[0])*tb
    if rate<=0:raise ValueError('Invalid FPS')
    absolute_start=origin+start;absolute_end=origin+end
    if absolute_end>Fraction(pts[-1])*tb+1/rate+Fraction(1,1000000):raise ValueError('Interval extends beyond source')
    selected=[i for i,t in enumerate(pts) if absolute_start<=t*tb<absolute_end]
    if not selected:raise ValueError('No source frames in interval')
    n=math.ceil((end-start)*rate)
    output=[float(Fraction(i)/rate) for i in range(n)]
    source=np.array([float(pts[i]*tb) for i in selected]);targets=np.array(output)+float(absolute_start)
    nearest=nearest_indices(source,targets)
    audio=probe.get('audio');audio_start=float(audio.get('start_time',0)) if audio else None
    return {'schema_version':'xms.timeline.v1','interval_semantics':'[start,end)','start_s':float(start),'end_s':float(end),'duration_s':float(end-start),'video_start_s':float(origin),'absolute_start_s':float(absolute_start),'absolute_end_s':float(absolute_end),'fps':str(rate),'frame_count':n,'output_times_s':output,'source_frame_indices':selected,'source_pts':[pts[i] for i in selected],'source_time_base':str(tb),'source_times_s':source.tolist(),'output_observation_indices':nearest.tolist(),'source_time_map':source[nearest].tolist(),'audio_start_s':audio_start,'audio_offset_from_clip_s':None if audio is None else audio_start-float(absolute_start),'rotation_degrees':probe['rotation_degrees'],'sample_aspect_ratio':probe['sample_aspect_ratio']}


def nearest_indices(source,targets):
    right=np.clip(np.searchsorted(source,targets),0,len(source)-1);left=np.maximum(0,right-1)
    return np.where(abs(source[left]-targets)<=abs(source[right]-targets),left,right)
