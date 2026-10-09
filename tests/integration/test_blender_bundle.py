"""Run fixture creation with Python; inspect Blender's independent evaluated evidence."""
import json
from pathlib import Path
import numpy as np
from xms.profiles.character import load_character
from xms.animation.bundle import neutral
from xms.animation.fk import evaluate,quat_matrix,matrix_quat
from xms.animation.io import write_bundle


def make_fixture(profile_path,out):
    p,r,f=load_character(profile_path)
    names=['rest','root_shift','arm_left','arm_right','leg_left','leg_right','wrist_neutral_l','wrist_l','wrist_neutral_r','wrist_r','face_neutral','head_yaw','jaw_open','blink_left','blink_right','smile_left','smile_right']
    b=neutral(p,r,np.arange(len(names),dtype=float));b.metadata['diagnostics']=names
    rest=evaluate(b,0)
    rotations={'arm_left':('upperarm_l',[0,0,1],50),'arm_right':('upperarm_r',[0,0,1],-50),'leg_left':('thigh_l',[1,0,0],-30),'leg_right':('thigh_r',[1,0,0],-30),'wrist_l':('hand_l',[0,0,1],65),'wrist_r':('hand_r',[0,0,1],-65),'head_yaw':('head',[0,1,0],30)}
    for label,(bone,axis,degrees) in rotations.items():
        i=names.index(label);j=p['joint_names'].index(bone);axis=np.array(axis);angle=np.deg2rad(degrees)
        q=np.r_[axis*np.sin(angle/2),np.cos(angle/2)];world=rest[j,:3,:3];world=world/np.linalg.norm(world,axis=0)
        b.arrays['local_rotation_delta'][i,j]=matrix_quat(world.T@quat_matrix(q)@world)
        b.arrays['rotation_validity'][i,j]=True;b.arrays['rotation_confidence'][i,j]=1;b.arrays['rotation_provenance'][i,j]=3
    b.arrays['root_translation'][1]=[.15,.1,0];b.arrays['root_validity'][1]=True;b.arrays['root_confidence'][1]=1;b.arrays['root_provenance'][1]=3
    for label,channel in [('jaw_open','jawOpen'),('blink_left','eyeBlinkLeft'),('blink_right','eyeBlinkRight'),('smile_left','mouthSmileLeft'),('smile_right','mouthSmileRight')]:
        i=names.index(label);j=p['face_channel_names'].index(channel);b.arrays['face_weights'][i,j]=.8
        b.arrays['face_validity'][i,j]=True;b.arrays['face_confidence'][i,j]=1;b.arrays['face_provenance'][i,j]=3
    return write_bundle(b,out)


def verify(path):
    m=json.loads((Path(path)/'manifest.json').read_text())
    assert len(m['diagnostics'])==17
    assert max(x['position_error_m'] for x in m['diagnostics'])<=.001
    assert max(x['angular_error_deg'] for x in m['diagnostics'])<=.1
    assert max(x['weight_error'] for x in m['diagnostics'])<=1e-5
    for row in m['diagnostics']:assert (Path(path)/(row['name']+'.png')).is_file()
    return m


if __name__=='__main__':
    import sys
    if sys.argv[1]=='create':print(make_fixture(sys.argv[2],sys.argv[3]))
    else:print(verify(sys.argv[2])['status'])
