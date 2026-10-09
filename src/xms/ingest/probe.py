import json
import subprocess
from fractions import Fraction


def probe(path,ffprobe):
    def query(*args):
        return json.loads(subprocess.check_output([str(ffprobe),'-v','error',*args,'-of','json',str(path)],text=True))
    data=query('-show_streams','-show_format')
    videos=[s for s in data['streams'] if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')]
    if len(videos)!=1:raise ValueError('Stage 3 requires one primary video stream')
    v=videos[0];audio=next((s for s in data['streams'] if s['codec_type']=='audio'),None)
    frames=query('-select_streams',str(v['index']),'-show_frames','-show_entries','frame=pts,best_effort_timestamp')['frames']
    pts=[int(f.get('pts',f.get('best_effort_timestamp'))) for f in frames]
    if not pts or any(b<=a for a,b in zip(pts,pts[1:])):raise ValueError('Missing/nonmonotonic decoded PTS')
    rotation=next((int(s['rotation']) for s in v.get('side_data_list',[]) if 'rotation' in s),int(v.get('tags',{}).get('rotate',0)))
    if rotation%90:raise ValueError('Only orthogonal source rotation supported')
    fps=v['avg_frame_rate'] if Fraction(v['avg_frame_rate'])>0 else v['r_frame_rate']
    return {'video':v,'audio':audio,'pts':pts,'time_base':v['time_base'],'fps':fps,'rotation_degrees':rotation,'sample_aspect_ratio':v.get('sample_aspect_ratio','1:1'),'format':data['format']}
