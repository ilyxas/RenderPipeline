"""Pose probes for the registered character: what the asset's own skinning does in test poses.

Builds one AnimationBundle with a rest sample and a few test poses (bone directions in the
bundle frame: Y up, Z forward, +X left), applies it through the pipeline's Blender adapter
and renders each pose from the front and from the side with a neutral camera and light rig.
No solver, limits, twist or collision stage is involved: this is the asset alone.

Run inside Blender's Python (`blender -b --python` or the `bpy` module):
    python tools/research/pose_probe.py --profile profiles/characters/xandra/v10 \
        --asset blender/Xandra_Wardrobe_v10.blend --out OUT [--samples 32] [--resolution 360x640]
"""
import argparse,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation

CHILD={'upperarm':'lowerarm','lowerarm':'hand','thigh':'calf','calf':'foot'}
ORDER=['upperarm','lowerarm','thigh','calf']


def unit(v):v=np.asarray(v,float);return v/np.linalg.norm(v)


def deg(a):return math.radians(a)


# Each pose: {bone_side: target direction of bone -> child}, plus optional root drop (m).
POSES={
    'rest':{},
    'arms_overhead':{'upperarm_l':[.15,1,0],'lowerarm_l':[.1,1,0],'upperarm_r':[-.15,1,0],'lowerarm_r':[-.1,1,0]},
    'arm_side_T':{'upperarm_l':[1,0,0],'lowerarm_l':[1,0,0],'upperarm_r':[-1,0,0],'lowerarm_r':[-1,0,0]},
    'elbow_120':{'upperarm_l':[.25,0,1],'lowerarm_l':[0,math.sin(deg(120)),math.cos(deg(120))],'upperarm_r':[-.25,-1,.25],'lowerarm_r':[-.1,-1,.6]},
    'arms_crossed':{'upperarm_l':[.25,-.55,.8],'lowerarm_l':[-1,.15,.25],'upperarm_r':[-.25,-.6,.75],'lowerarm_r':[1,.05,.3]},
    'knee_raise_90':{'thigh_l':[0,0,1],'calf_l':[0,-1,0]},
    'squat':{'thigh_l':[.25,-.25,1],'calf_l':[.05,-1,-.35],'thigh_r':[-.25,-.25,1],'calf_r':[-.05,-1,-.35],'_root_drop':.42},
}


CLOSEUPS=[('arms_overhead','upperarm_l','front',1.1),('arms_overhead','upperarm_l','side',1.1),('arm_side_T','upperarm_l','front',1.0),
          ('elbow_120','lowerarm_l','side',.9),('elbow_120','lowerarm_l','front',.9),('arms_crossed','spine_04','front',1.0),
          ('knee_raise_90','calf_l','side',1.1),('knee_raise_90','thigh_l','front',1.2),('squat','pelvis','side',1.4),('squat','pelvis','front',1.4)]


def build(profile_path):
    from xms.profiles.character import load_character
    from xms.animation.bundle import neutral
    from xms.animation.fk import evaluate,quat_matrix
    from xms.animation.validate import validate
    p,rig,_=load_character(profile_path);names=p['joint_names'];idx={s:i for i,s in enumerate(names)}
    n=len(POSES);b=neutral(p,rig,np.arange(n)/24.);a=b.arrays;rest=evaluate(b,0)
    H={s:rest[j,:3,3] for s,j in idx.items()};R={s:rest[j,:3,:3]/np.linalg.norm(rest[j,:3,:3],axis=0) for s,j in idx.items()}
    for k,(pose,spec) in enumerate(POSES.items()):
        D={}
        for side in ('l','r'):
            for part in ORDER:
                name=f'{part}_{side}';parent=names[int(a['parent_indices'][idx[name]])]
                pd=D.get(parent,np.eye(3))
                # parents that are not posed here inherit their own parent's delta
                q=parent
                while q not in D and int(a['parent_indices'][idx[q]])>=0:q=names[int(a['parent_indices'][idx[q]])]
                pd=D.get(q,np.eye(3))
                if name not in spec:D[name]=pd;continue
                current=pd@unit(H[CHILD[part]+'_'+side]-H[name]);target=unit(spec[name])
                axis=np.cross(current,target);s=np.linalg.norm(axis);c=float(current@target)
                swing=np.eye(3) if s<1e-9 else Rotation.from_rotvec(axis/s*math.atan2(s,c)).as_matrix()
                D[name]=swing@pd
        actual=[]
        for j,name in enumerate(names):
            parent=int(a['parent_indices'][j])
            if name not in D:
                q=j
                while names[q] not in D and int(a['parent_indices'][q])>=0:q=int(a['parent_indices'][q])
                D[name]=D.get(names[q],np.eye(3))
            desired=D[name]@R[name];parent_world=a['armature_transform'][:3,:3]/np.linalg.norm(a['armature_transform'][:3,:3],axis=0) if parent<0 else actual[parent]
            a['local_rotation_delta'][k,j]=Rotation.from_matrix(quat_matrix(a['rest_rotation'][j]).T@parent_world.T@desired).as_quat();actual.append(desired)
        a['root_translation'][k,1]=-spec.get('_root_drop',0.)
    for g in ('rotation','root'):a[g+'_validity'][:]=True;a[g+'_confidence'][:]=1;a[g+'_provenance'][:]=3
    b.metadata.update(solver={'backend':'pose_probe','version':'1','channels':'body'},calibration={},limitations=['Synthetic pose probes; no observations.'],diagnostics=list(POSES))
    return validate(b)


def render(bundle_dir,profile_path,asset,out,samples,resolution):
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=str(asset))
    from xms.render.blender_adapter import apply_bundle
    apply_bundle(str(bundle_dir),str(profile_path),fps=24)
    sc=bpy.context.scene
    for o in bpy.data.objects:
        if o.type in ('LIGHT','CAMERA'):o.hide_render=True
    world=bpy.data.worlds.new('probe');world.use_nodes=True;world.node_tree.nodes['Background'].inputs[0].default_value=(.35,.36,.38,1);world.node_tree.nodes['Background'].inputs[1].default_value=1.2;sc.world=world
    def light(name,rot,energy):
        d=bpy.data.lights.new(name,'SUN');d.energy=energy;o=bpy.data.objects.new(name,d);o.rotation_euler=[math.radians(x) for x in rot];sc.collection.objects.link(o)
    light('key',(50,0,-35),3.5);light('rim',(60,0,150),1.5)
    cam=bpy.data.objects.new('probe_cam',bpy.data.cameras.new('probe_cam'));sc.collection.objects.link(cam);sc.camera=cam;cam.data.lens=50
    sc.render.engine='CYCLES';sc.cycles.device='CPU';sc.cycles.samples=samples;sc.cycles.use_denoising=True
    sc.render.resolution_x,sc.render.resolution_y=resolution;sc.render.film_transparent=False
    arm=bpy.data.objects['root'];out.mkdir(parents=True,exist_ok=True);shots=[]
    views={'front':(Vector((0,-1,0)),(90,0,0)),'side':(Vector((1,0,0)),(90,0,90))}
    for k,pose in enumerate(POSES):
        sc.frame_set(k+1);dg=bpy.context.evaluated_depsgraph_get()
        pts=[arm.matrix_world@b.head for b in arm.pose.bones if b.name in ('head','foot_l','foot_r','hand_l','hand_r','pelvis')]
        lo=Vector([min(p[i] for p in pts) for i in range(3)]);hi=Vector([max(p[i] for p in pts) for i in range(3)]);centre=(lo+hi)/2;height=max(hi.z-lo.z+.35,1.2)
        for view,(direction,rot) in views.items():
            distance=height/2/math.tan(cam.data.angle_y/2)*1.12
            cam.location=centre+direction*distance;cam.rotation_euler=[math.radians(x) for x in rot]
            path=out/f'{k:02d}_{pose}_{view}.png';sc.render.filepath=str(path);bpy.ops.render.render(write_still=True);shots.append(path);print('XMS_PROBE',path,flush=True)
    # Close-ups at the joints the pose loads: (pose, bone, view, distance m).
    for pose,bone,view,dist in CLOSEUPS:
        k=list(POSES).index(pose);sc.frame_set(k+1);centre=arm.matrix_world@arm.pose.bones[bone].head
        direction,rot=views[view];cam.location=centre+direction*dist;cam.rotation_euler=[math.radians(x) for x in rot]
        sc.render.resolution_x=sc.render.resolution_y=resolution[1]
        path=out/f'zoom_{pose}_{bone}_{view}.png';sc.render.filepath=str(path);bpy.ops.render.render(write_still=True);print('XMS_PROBE',path,flush=True)
    return shots


def sheet(shots,out):
    from PIL import Image,ImageDraw
    images=[Image.open(s) for s in shots];w,h=images[0].size;cols=2;rows=len(images)//2
    canvas=Image.new('RGB',(w*cols*2,h*((rows+1)//2)),'white')
    for i,(im,path) in enumerate(zip(images,shots)):
        r,c=divmod(i,4);canvas.paste(im,(c*w,r*h));ImageDraw.Draw(canvas).text((c*w+8,r*h+8),path.stem[3:],fill='white')
    canvas.save(out);return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--profile',required=True);ap.add_argument('--asset',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--samples',type=int,default=32);ap.add_argument('--resolution',default='360x640')
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]);out=Path(args.out).resolve()
    from xms.animation.io import write_bundle
    bundle=build(args.profile);write_bundle(bundle,out/'bundle')
    shots=render(out/'bundle',Path(args.profile).resolve(),Path(args.asset).resolve(),out/'frames',args.samples,tuple(int(x) for x in args.resolution.split('x')))
    print('SHEET',sheet(shots,out/'sheet.png'))
