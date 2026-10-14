"""Read-only evaluated mesh diagnostics; no motion edits in Blender."""
import sys,json,argparse,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from xms.render.blender_adapter import apply_bundle
from xms.animation.io import bundle_hash
from xms.geometry.surface_distance import watertight
from xms.qa.surface_collision import assess
p=argparse.ArgumentParser()
for key in ('bundle','profile','out','indices'):p.add_argument('--'+key,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.out);indices=[int(x) for x in a.indices.split(',')]
from xms.animation.io import read_bundle
original=read_bundle(a.bundle);dt=np.diff(original.arrays['sample_times_s']);fps=1/np.median(dt) if len(dt) else 24
b,profile,face=apply_bundle(a.bundle,a.profile,fps=fps);records=[];planned=list(range(len(b.arrays['sample_times_s'])))
# Registered surface identities from the actual asset, independent of collider sizes.
look=json.loads((Path(a.profile)/'look.json').read_text())
pairs=[tuple(x) for x in look.get('surface_qa_pairs',[('F4_Hands','Xandra_Top'),('F4_Hands','Xandra_Shorts')])]
for i in indices:
    sc=bpy.context.scene;sc.frame_set(i+1);graph=bpy.context.evaluated_depsgraph_get();meshes={}
    for name in set(x for pair in pairs for x in pair):
        obj=bpy.data.objects.get(name)
        if obj is None or obj.hide_render:continue
        evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles();vertices=[evaluated.matrix_world@v.co for v in mesh.vertices];triangles=[tuple(t.vertices) for t in mesh.loop_triangles];meshes[name]=(vertices,triangles,BVHTree.FromPolygons(vertices,triangles,all_triangles=True),watertight(triangles));evaluated.to_mesh_clear()
    for source,target in pairs:
        row={'sample':i,'source':source,'target':target,'candidate_depths_m':[],'signed_interior_available':False}
        if source not in meshes or target not in meshes:row['reason']='required surface hidden or missing';records.append(row);continue
        verts,source_tri,source_tree,_=meshes[source];target_verts,target_tri,tree,closed=meshes[target];row.update(source_vertices=len(verts),target_triangles=len(target_tri),target_watertight=closed)
        # All visible source vertices are queried. Open cloth admits only a
        # local-normal candidate sign; coverage cannot imply surface acceptance.
        row['signed_interior_available']=closed;row['worst_evidence']=None;row['sign_disagreements']=0;row['ambiguous_candidate_depths_m']=[]
        for point in verts:
            location,normal,index,distance=tree.find_nearest(point)
            if location is None:continue
            is_inside=False
            if closed:
                direction=Vector((1.,.371,.529)).normalized();origin=point.copy();hits=0
                for _ in range(128):
                    hit,_,_,_=tree.ray_cast(origin,direction)
                    if hit is None:break
                    hits+=1;origin=hit+direction*1e-6
                else:row['signed_interior_available']=False
                is_inside=hits%2==1
            else:is_inside=(point-location).dot(normal)<0 and distance<.1
            if is_inside and closed and (point-location).dot(normal)>1e-5:
                # Ray parity can disagree with the closest outward face on
                # self-intersecting/deformed cloth. Preserve the evidence but
                # do not report this as certified interior penetration.
                row['signed_interior_available']=False;row['sign_disagreements']+=1;row['ambiguous_candidate_depths_m'].append(float(distance));continue
            if is_inside:
                row['candidate_depths_m'].append(float(distance))
                if row['worst_evidence'] is None or distance>row['worst_evidence']['depth_m']:row['worst_evidence']={'depth_m':float(distance),'point_blender_m':list(point),'normal_blender':list(normal)}
        candidates=source_tree.overlap(tree);row['triangle_candidates']=len(candidates);row['intersection_coverage_complete']=len(candidates)<=5000;crossings=0
        from mathutils.geometry import intersect_ray_tri
        for si,ti in candidates[:5000]:
            st=[verts[k] for k in source_tri[si]];tt=[target_verts[k] for k in target_tri[ti]]
            crossed=False
            for edges,triangle in ((st,tt),(tt,st)):
                for k in range(3):
                    start,end=edges[k],edges[(k+1)%3];delta=end-start
                    if delta.length<1e-9:continue
                    hit=intersect_ray_tri(*triangle,delta.normalized(),start,True)
                    if hit is not None and (hit-start).length<=delta.length+1e-7:crossed=True
            crossings+=int(crossed)
        row['triangle_intersections']=crossings
        row['geometry']='evaluated skin/cloth BVH; ray-parity sign when watertight, edge/triangle intersections';records.append(row)
# Skin witnesses: forearm/hand skin against body skin and against the other arm.
import re
from xms.geometry.winding import winding_numbers
skin_path=Path(a.profile).parent/'skin_witnesses.v1.json';skin=json.loads(skin_path.read_text()) if skin_path.exists() else None
if skin and skin['rig_sha256']!=profile['files']['rig.npz']:skin=None
if skin:
    arm=bpy.data.objects[profile['armature']];bones=set(arm.data.bones.keys())
    def region_of(bone):
        side=re.search(r'_([lr])$',bone)
        if side:
            for region,pattern in skin['regions'].items():
                if re.match(pattern,bone):return region+'_'+side.group(1)
        return 'body'
    def chain_of(bone):
        # The arm chain bone a skin vertex rides with (twist/corrective bones fold into it).
        side=re.search(r'_([lr])$',bone)
        if not side:return 'body'
        for prefix,chain in (('lowerarm','lowerarm'),('upperarm','upperarm'),('hand','hand'),('wrist','hand'),('thumb','hand'),('index','hand'),('middle','hand'),('ring','hand'),('pinky','hand')):
            if bone.startswith(prefix):return chain+'_'+side.group(1)
        return 'body'
    regions=[];attach=[];is_hand=[]
    for name in skin['meshes']:
        o=bpy.data.objects[name];groups={g.index:g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            weights={};chains={}
            for g in v.groups:
                bone=groups.get(g.group)
                if bone in bones:weights[region_of(bone)]=weights.get(region_of(bone),0)+g.weight;chains[chain_of(bone)]=chains.get(chain_of(bone),0)+g.weight
            regions.append(max(weights,key=weights.get) if weights else 'none');attach.append(max(chains,key=chains.get) if chains else 'body');is_hand.append(name=='F4_Hands')
    regions=np.array(regions);attach=np.array(attach);is_hand=np.array(is_hand)
    skin_result={}
    masks=[(m,m.show_viewport) for name in skin['meshes'] for m in bpy.data.objects[name].modifiers if m.type=='MASK']
    # Kept outfit drivers would switch the masks back on during evaluation; mute them meanwhile.
    mask_drivers=[d for name in skin['meshes'] if bpy.data.objects[name].animation_data for d in bpy.data.objects[name].animation_data.drivers if d.data_path.startswith('modifiers[') and d.data_path.endswith('.show_viewport')]
    for i in indices:
        sc=bpy.context.scene;sc.frame_set(i+1)
        for d in mask_drivers:d.mute=True
        for m,_ in masks:m.show_viewport=False
        graph=bpy.context.evaluated_depsgraph_get();vertices=[];triangles=[];offset=0
        for name in skin['meshes']:
            e=bpy.data.objects[name].evaluated_get(graph);mesh=e.to_mesh();mesh.calc_loop_triangles()
            vertices+= [list(e.matrix_world@v.co) for v in mesh.vertices];triangles+=[[k+offset for k in t.vertices] for t in mesh.loop_triangles];offset=len(vertices);e.to_mesh_clear()
        for m,state in masks:m.show_viewport=state
        for d in mask_drivers:d.mute=False
        vertices=np.array(vertices);triangles=np.array(triangles)
        if len(vertices)!=len(regions):raise RuntimeError('Evaluated skin topology differs from the registered mesh')
        triangle_region=regions[triangles[:,0]]
        if 'capsules' not in skin_result:
            # Forearm capsule radii from the skin around the elbow->wrist axis (rigid, so any frame works).
            skin_result['capsules']={}
            for side in 'lr':
                e=np.array(arm.matrix_world@arm.pose.bones['lowerarm_'+side].head);h=np.array(arm.matrix_world@arm.pose.bones['hand_'+side].head);axis=h-e;length=float(np.linalg.norm(axis))
                pts=vertices[(regions=='forearm_hand_'+side)&(attach=='lowerarm_'+side)];t=(pts-e)@axis/length**2;radial=np.linalg.norm(pts-(e+np.outer(t,axis)),axis=1)
                skin_result['capsules'][side]={'r_elbow_m':float(np.median(radial[(t>=0)&(t<.25)])),'r_wrist_m':float(np.median(radial[(t>.75)&(t<=1)])),'source':'median skin distance to the lowerarm->hand axis'}
        for source,target in skin['pairs']:
            side=source[-1];query=np.flatnonzero(regions==source);tri=triangles[np.isin(triangle_region,skin['targets'][target])]
            tree=BVHTree.FromPolygons([Vector(v) for v in vertices],tri.tolist(),all_triangles=True)
            row={'sample':i,'source':'skin:'+source,'target':'skin:'+target,'kind':'skin_witness','candidate_depths_m':[],'worst_evidence':None,'signed_interior_available':True,'intersection_coverage_complete':True,'sign_method':'generalized_winding_number','query_points':int(len(query)),'target_triangles':int(len(tri))}
            near=[];found=[];row['witnesses']=[]
            for k in query:
                location,normal,face,distance=tree.find_nearest(Vector(vertices[k]))
                if location is not None and distance<skin['candidate_radius_m']:near.append(k);found.append((location,distance,face))
            if near:
                w=np.abs(winding_numbers(vertices[near],vertices,tri))
                for k,(location,distance,face),wk in zip(near,found,w):
                    if wk>.5 and distance>skin['margin_m']:
                        row['candidate_depths_m'].append(float(distance))
                        # Hand skin on anything is intentional contact; forearm skin is penetration.
                        kind='contact' if is_hand[k] else 'body' if target=='body' else 'arm_arm'
                        row['witnesses'].append({'vertex':int(k),'class':kind,'side':side,'depth_m':float(distance),'point_blender_m':list(vertices[k]),'target_blender_m':list(location),'attach':str(attach[k]),'target_attach':'body' if target=='body' else str(attach[tri[face][0]])})
                        if row['worst_evidence'] is None or distance>row['worst_evidence']['depth_m']:
                            push=(location-Vector(vertices[k])).normalized()
                            row['worst_evidence']={'depth_m':float(distance),'point_blender_m':list(vertices[k]),'normal_blender':list(push),'side':side,'pair':f'{source}->{target}'}
            row['witnesses']=sorted(row['witnesses'],key=lambda x:-x['depth_m'])[:80]
            records.append(row)
result=assess(records,planned,indices);result['skin_capsules']=skin_result.get('capsules') if skin else None;result.update(bundle_hash=bundle_hash(a.bundle),blender_version=bpy.app.version_string,blender_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(out/'quality.json').write_text(json.dumps(result,indent=2))
