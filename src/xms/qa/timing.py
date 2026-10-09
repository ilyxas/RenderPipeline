from fractions import Fraction
import numpy as np
from .contract import metric


def timing_metrics(timeline,verification):
    tolerance=1/float(Fraction(timeline['fps']))
    duration=abs(verification['video_duration_s']-timeline['duration_s'])
    targets=np.array(timeline['output_times_s'])+timeline['absolute_start_s']
    mapping=np.array(timeline['source_time_map']);map_error=float(np.max(abs(mapping-targets)))
    # A sparse VFR source may have gaps longer than one output frame. This is a
    # sampling residual, not an A/V or duration gate; report it without a pass.
    out={'duration_error':metric(duration,'s',passed=duration<=tolerance+1e-6),'source_mapping_max_error':metric(map_error,'s'),'frame_count':metric(verification['frame_count'],'frames',passed=verification['frame_count']==timeline['frame_count'])}
    if verification['audio_present']:
        start=abs(verification['audio_start_s']);end=abs(verification['audio_duration_s']-verification['video_duration_s'])
        out['av_timestamp_skew']=metric(max(start,end),'s',passed=max(start,end)<=tolerance+1e-6)
        out['audio_content_sync']=metric(reason='Timestamp checks do not measure speech/lip synchronization; annotated event measurement required',unit='s')
    else:out['av_timestamp_skew']=metric(reason='Source has no audio',unit='s')
    return out
