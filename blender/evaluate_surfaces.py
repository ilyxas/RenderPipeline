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
pairs=[('F4_Hands','Xandra_Top'),('F4_Hands','Xandra_Shorts')]
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
result=assess(records,planned,indices);result.update(bundle_hash=bundle_hash(a.bundle),blender_version=bpy.app.version_string,blender_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(out/'quality.json').write_text(json.dumps(result,indent=2))
