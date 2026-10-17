"""Read-only check that a .blend still fits a registered profile (used by xms.registry.register).

character: same armature, bone order/parents and rest pose as the profile's rig.npz, and every
face_map shape key present. scene: the profile's collection and world exist in the file.
Prints one line `XMS_CHECK {"ok": ..., "reason": ...}`; never saves.
"""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['character','scene']);p.add_argument('--profile',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);profile=Path(a.profile)


def done(ok,reason=''):print('XMS_CHECK '+json.dumps({'ok':ok,'reason':reason}),flush=True);sys.exit(0)


if a.kind=='scene':
    sp=json.loads((profile/'scene.json').read_text())
    if sp['collection'] not in bpy.data.collections:done(False,'нет коллекции '+sp['collection'])
    if sp['world'] not in bpy.data.worlds:done(False,'нет мира '+sp['world'])
    done(True)

from xms.animation.fk import decompose
C=np.array([[1.,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
character=json.loads((profile/'character.json').read_text());arm=bpy.data.objects.get(character['armature'])
if arm is None or arm.type!='ARMATURE':done(False,'нет арматуры '+character['armature'])
if arm.data.pose_position!='POSE':done(False,'арматура сохранена в Rest Position — персонаж не будет двигаться')
bones=sorted(arm.data.bones,key=lambda b:len(b.parent_recursive));names=[b.name for b in bones]
if names!=character['joint_names']:done(False,'изменился состав или порядок костей')
with np.load(profile/'rig.npz') as z:rig={k:z[k] for k in z.files}
parents=np.array([names.index(b.parent.name) if b.parent else -1 for b in bones])
if not np.array_equal(parents,rig['parent_indices']):done(False,'изменилась иерархия костей')
T=[];Q=[]
for b in bones:
    local=np.array(b.matrix_local) if not b.parent else np.linalg.inv(np.array(b.parent.matrix_local))@np.array(b.matrix_local)
    t,q,s=decompose(C@local@C.T);T.append(t);Q.append(q)
dq=np.minimum(np.abs(np.array(Q)-rig['rest_rotation']).max(1),np.abs(np.array(Q)+rig['rest_rotation']).max(1))
if np.abs(np.array(T)-rig['rest_translation']).max()>1e-4 or dq.max()>1e-4:done(False,'изменилась поза покоя скелета (rest pose)')
if not np.allclose(C@np.array(arm.matrix_world)@C.T,rig['armature_transform'],atol=1e-5):done(False,'изменилось положение или масштаб арматуры')
face=json.loads((profile/'face_map.json').read_text())
for ch in face['channels'].values():
    obj=bpy.data.objects.get(ch['object'])
    if obj is None or not obj.data.shape_keys or ch['key'] not in obj.data.shape_keys.key_blocks:done(False,'нет формы лица '+ch['key'])
done(True)
