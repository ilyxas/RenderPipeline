"""Stage-15 bounded collision refinement over a window, for any published bundle.

Bundle-agnostic: the source bundle (Viggle, video, temporal) is never modified; one
candidate is built from exact skin witnesses and must pass a fresh exact QA before it
may replace the source (accept()). Architecture agreed with the arbiter (CGC.txt,
2026-10-12 03:02); witness/contact math revised after Grok's review (03:30):

- variables per frame, both arms jointly: shoulder 3 (rotation vector in the upper arm
  delta frame), elbow 1 (HumanoidAlpha hinge axis), wrist 2 (flexion/deviation); torso
  and root fixed;
- forearm–forearm crossing is a capsule term on the two elbow->wrist axes: separation
  along the crossing normal, whose sign (which arm is in front) comes from the source and
  is frozen per contact episode; radii are measured on the skin. Point witnesses of this
  pair stay in the exact QA only (pushing points to the nearest surface slides the
  crossing along the arm instead of resolving it);
- hand skin is contact only where the source already had contact (within 10 mm for at
  least 3 frames, hysteresis 10/25 mm); contact may sit on the surface within 2 mm and
  is held there; every other witness is penetration;
- penetration is a quadratic cost on a soft maximum (log-sum-exp, 3 mm) per cluster
  (frame, class, side, bone pair): a deep sparse region is not outvoted by a dense
  shallow one, and depth is not saturated;
- temporal velocity/acceleration penalties act on the correction delta only; soft
  confidence-weighted anchors keep elbow/wrist near the source (the front arm of a
  crossing is anchored more loosely, so it moves more);
- increments are bounded by the frozen 5°/joint trust region; a wide solve records the
  required-correction estimate for diagnostics only.
"""
import copy,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from xms.animation.fk import evaluate,quat_matrix,trs

LIMITS=Path(__file__).resolve().parents[3]/'profiles/characters/xandra/arm_limits.v1.json'
CHAIN=('upperarm','lowerarm','hand');SIDES=('l','r')
SETTINGS={'margin_m':.003,'contact_tolerance_m':.002,'contact_separation_m':.005,'contact_enter_m':.010,'contact_exit_m':.025,'contact_min_frames':3,
          'class_weight':{'body':2.,'hand_arm':2.,'capsule':2.,'contact':1.},'softmax_temperature_m':.003,'capsule_margin_m':.003,'front_arm_anchor_scale':.5,
          'anchor_position_per_cm':.4,'anchor_angle_per_rad':2.,'velocity_per_rad':10.,'acceleration_per_rad':10.,
          'trust_region_deg':5.,'diagnostic_bound_deg':45.}


def blender_to_bundle(v):
    v=np.asarray(v,float);return np.stack([v[...,0],v[...,2],-v[...,1]],axis=-1)


def smooth_hinge(x,eps=.001):
    return .5*(x+np.sqrt(x*x+eps*eps))


def segment_closest(p0,p1,q0,q1):
    """Closest points of segments p0p1, q0q1 (vectorised over the first axis): s, t, P, Q."""
    d1=p1-p0;d2=q1-q0;r=p0-q0;a=np.sum(d1*d1,-1);e=np.sum(d2*d2,-1);f=np.sum(d2*r,-1);c=np.sum(d1*r,-1);b=np.sum(d1*d2,-1);den=a*e-b*b
    s=np.clip(np.where(den>1e-12,(b*f-c*e)/np.where(den>1e-12,den,1),0.),0,1);t=(b*s+f)/np.maximum(e,1e-12)
    s=np.where(t<0,np.clip(-c/np.maximum(a,1e-12),0,1),np.where(t>1,np.clip((b-c)/np.maximum(a,1e-12),0,1),s));t=np.clip(t,0,1)
    return s,t,p0+d1*s[...,None],q0+d2*t[...,None]


class Arms:
    """Arm-chain FK with local corrections; everything above the clavicle stays as in the source."""
    def __init__(self,bundle,frames):
        a=bundle.arrays;self.names=bundle.metadata['joint_names'];self.frames=np.asarray(frames);self.a=a
        world=np.array([evaluate(bundle,i) for i in self.frames]);self.idx={s:[self.names.index(f'{c}_{s}') for c in CHAIN] for s in SIDES}
        self.parent={s:world[:,int(a['parent_indices'][self.idx[s][0]])] for s in SIDES};self.source={s:world[:,self.idx[s]] for s in SIDES}
        rest=copy.deepcopy(bundle);rest.arrays['local_rotation_delta'][:]=[0,0,0,1];rest.arrays['root_translation'][:]=0;rw=evaluate(rest,0)
        unit=lambda m:m[:3,:3]/np.linalg.norm(m[:3,:3],axis=0)
        spec=json.loads(LIMITS.read_text());self.axes={}
        for s in SIDES:
            u,l,h=self.idx[s]
            self.axes[s]=(unit(rw[l]).T@np.array(spec['elbow']['axis'][s]),unit(rw[h]).T@np.array(spec['wrist']['flexion_axis'][s]),unit(rw[h]).T@np.array(spec['wrist']['deviation_axis'][s]))
        self.q={s:[Rotation.from_quat(a['local_rotation_delta'][self.frames,j]) for j in self.idx[s]] for s in SIDES}

    def corrected(self,delta):
        """delta (n,2,6) -> new local quats and world matrices {side: (n,3,4,4)}."""
        quats={};worlds={}
        for k,s in enumerate(SIDES):
            d=delta[:,k];hinge,flex,dev=self.axes[s]
            qu=self.q[s][0]*Rotation.from_rotvec(d[:,0:3]);ql=Rotation.from_rotvec(np.outer(d[:,3],hinge))*self.q[s][1];qh=Rotation.from_rotvec(np.outer(d[:,4],flex)+np.outer(d[:,5],dev))*self.q[s][2]
            quats[s]=[qu,ql,qh];w=[];parent=self.parent[s]
            for j,q in zip(self.idx[s],quats[s]):
                local=np.tile(trs(self.a['rest_translation'][j],[0,0,0,1],self.a['rest_scale'][j]),(len(self.frames),1,1))
                local[:,:3,:3]=quat_matrix(self.a['rest_rotation'][j])@q.as_matrix()@np.diag(self.a['rest_scale'][j])
                parent=parent@local;w.append(parent)
            worlds[s]=np.stack(w,axis=1)
        return quats,worlds


def witnesses_from(report,frames):
    position={f:i for i,f in enumerate(frames)};rows=[]
    for row in report.get('records',[]):
        for w in row.get('witnesses',[]):
            if row['sample'] in position:rows.append({**w,'frame':position[row['sample']]})
    return rows


def contact_episodes(witnesses,n):
    """Frames where each (side, target kind) hand contact existed in the source, with hysteresis."""
    depth={}
    for w in witnesses:
        if w['class']=='contact':
            key=(w['side'],'body' if w['target_attach']=='body' else 'arm');depth.setdefault(key,np.full(n,np.inf))
            depth[key][w['frame']]=min(depth[key][w['frame']],w['depth_m'])
    S=SETTINGS;episodes={}
    for key,d in depth.items():
        active=np.zeros(n,bool);on=False
        for t in range(n):
            # A penetrating hand witness reports its distance inside; a shallow one is touching.
            on=(d[t]<=S['contact_enter_m']) if not on else (d[t]<=S['contact_exit_m'])
            active[t]=on
        start=None
        for t in range(n+1):
            if t<n and active[t]:
                if start is None:start=t
            elif start is not None:
                if t-start<S['contact_min_frames']:active[start:t]=False
                start=None
        episodes[key]=active
    return episodes


def build_terms(arms,witnesses):
    """Point terms (penetration/contact) attached to arm-chain bones in the source pose."""
    n=len(arms.frames);episodes=contact_episodes(witnesses,n);terms=[]
    for w in witnesses:
        if w['class']=='arm_arm':continue  # forearm–forearm: capsule term, the exact QA keeps the points
        kind=w['class']
        if kind=='contact':
            key=(w['side'],'body' if w['target_attach']=='body' else 'arm')
            kind='contact' if episodes.get(key,np.zeros(n,bool))[w['frame']] else ('body' if w['target_attach']=='body' else 'hand_arm')
        t=w['frame'];p=blender_to_bundle(w['point_blender_m']);q=blender_to_bundle(w['target_blender_m'])
        chain=lambda name:(name[-1],CHAIN.index(name[:-2]))
        ps,pj=chain(w['attach']);p_local=np.linalg.solve(arms.source[ps][t,pj],np.r_[p,1])
        if w['target_attach']=='body':qs,qj,q_local=None,None,None
        else:qs,qj=chain(w['target_attach']);q_local=np.linalg.solve(arms.source[qs][t,qj],np.r_[q,1])
        u=q-p;norm=np.linalg.norm(u);u=u/norm if norm>1e-9 else np.zeros(3)
        terms.append({**w,'class':kind,'p':p,'q':q,'u':u,'p_side':SIDES.index(ps),'p_joint':pj,'p_local':p_local,'q_side':None if qs is None else SIDES.index(qs),'q_joint':qj,'q_local':q_local})
    keys=sorted({(x['frame'],x['class'],x['side'],x['attach'],x['target_attach']) for x in terms});index={k:i for i,k in enumerate(keys)}
    for x in terms:x['cluster']=index[(x['frame'],x['class'],x['side'],x['attach'],x['target_attach'])]
    return terms,keys,episodes


def capsule_setup(arms,witnesses,capsules):
    """Frames, frozen crossing sign and radii for the forearm capsule term."""
    n=len(arms.frames);crossing=np.zeros(n,bool)
    for w in witnesses:
        if w['class']=='arm_arm':crossing[w['frame']]=True
    if not crossing.any() or not capsules:return None
    e={s:arms.source[s][:,1,:3,3] for s in SIDES};h={s:arms.source[s][:,2,:3,3] for s in SIDES};z=np.array([0,0,1.])
    sign=np.zeros(n);start=None
    for t in range(n+1):
        if t<n and crossing[t]:
            if start is None:start=t
        elif start is not None:
            # Which forearm is in front (toward the camera) over the whole episode, frozen.
            _,_,P,Q=segment_closest(e['l'][start:t],h['l'][start:t],e['r'][start:t],h['r'][start:t]);sign[start:t]=1. if np.mean((P-Q)@z)>=0 else -1.;start=None
    r={s:(capsules[s]['r_elbow_m'],capsules[s]['r_wrist_m']) for s in SIDES}
    return {'frames':np.flatnonzero(crossing),'sign':sign,'radii':r,'front':{int(t):('l' if sign[t]>0 else 'r') for t in np.flatnonzero(crossing)}}


def capsule_separation(worlds,cap,frames):
    e={s:worlds[s][frames,1,:3,3] for s in SIDES};h={s:worlds[s][frames,2,:3,3] for s in SIDES}
    s,t,P,Q=segment_closest(e['l'],h['l'],e['r'],h['r']);dl=h['l']-e['l'];dr=h['r']-e['r'];n=np.cross(dl,dr);norm=np.linalg.norm(n,axis=-1)
    # Crossing normal; near-parallel axes fall back to the closest-point direction.
    v=P-Q;fallback=norm<.2*np.linalg.norm(dl,axis=-1)*np.linalg.norm(dr,axis=-1)
    n=np.where(fallback[:,None],v,n);n=n/np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-9)
    n=n*np.sign(np.where(np.abs(n[:,2])>1e-6,n[:,2],1.))[:,None]*cap['sign'][frames][:,None]
    rl=cap['radii']['l'][0]+(cap['radii']['l'][1]-cap['radii']['l'][0])*s;rr=cap['radii']['r'][0]+(cap['radii']['r'][1]-cap['radii']['r'][0])*t
    return np.sum(v*n,axis=-1),rl+rr


def solve(arms,terms,clusters,cap,confidence,bound_deg):
    n=len(arms.frames);size=n*12;S=SETTINGS;src_elbow={s:arms.source[s][:,1,:3,3] for s in SIDES};src_wrist={s:arms.source[s][:,2,:3,3] for s in SIDES}
    K=len(terms);T=np.array([x['frame'] for x in terms],int);U=np.array([x['u'] for x in terms]).reshape(-1,3)
    P0=np.array([x['p'] for x in terms]).reshape(-1,3);Q0=np.array([x['q'] for x in terms]).reshape(-1,3);depth=np.array([x['depth_m'] for x in terms])
    contact=np.array([x['class']=='contact' for x in terms],bool);C=np.array([x['cluster'] for x in terms],int)
    PS=np.array([x['p_side'] for x in terms],int);PJ=np.array([x['p_joint'] for x in terms],int);PL=np.array([x['p_local'] for x in terms]).reshape(-1,4)
    moving=np.array([x['q_side'] is not None for x in terms],bool);QS=np.array([x['q_side'] or 0 for x in terms],int);QJ=np.array([x['q_joint'] or 0 for x in terms],int)
    QL=np.array([x['q_local'] if x['q_local'] is not None else np.zeros(4) for x in terms]).reshape(-1,4)
    cw=np.array([S['class_weight'][k[1]] for k in clusters]);ct=np.array([k[0] for k in clusters],int);temp=S['softmax_temperature_m']
    anchor={s:np.ones(n) for s in SIDES}
    if cap:
        for t,front in cap['front'].items():anchor[front][t]=S['front_arm_anchor_scale']
    def residuals(x):
        delta=x.reshape(n,2,6);_,worlds=arms.corrected(delta);out=[]
        if K:
            stacked=np.stack([worlds[s] for s in SIDES]);pw=np.einsum('kij,kj->ki',stacked[PS,T,PJ],PL)[:,:3];qw=Q0.copy()
            if moving.any():qw[moving]=np.einsum('kij,kj->ki',stacked[QS[moving],T[moving],QJ[moving]],QL[moving])[:,:3]
            rel=np.sum(((pw-P0)-(qw-Q0))*U,axis=1)
            violation=np.where(contact,smooth_hinge((depth-S['contact_tolerance_m'])-rel)+smooth_hinge(rel-depth-S['contact_separation_m']),smooth_hinge(depth+S['margin_m']-rel))
            # Soft maximum per cluster (log-sum-exp), quadratic in centimetres: no depth saturation.
            peak=np.zeros(len(clusters));np.maximum.at(peak,C,violation);acc=np.zeros(len(clusters));np.add.at(acc,C,np.exp((violation-peak[C])/temp))
            out.append(cw*(peak+temp*np.log(acc)-temp*np.log(np.bincount(C,minlength=len(clusters))))/.01)
        if cap:
            f=cap['frames'];sep,radius=capsule_separation(worlds,cap,f);out.append(S['class_weight']['capsule']*smooth_hinge(radius+S['capsule_margin_m']-sep)/.01)
        for s in SIDES:
            c=confidence[s]*anchor[s]
            out.append((S['anchor_position_per_cm']*c[:,None]*(worlds[s][:,1,:3,3]-src_elbow[s])/.01).ravel())
            out.append((S['anchor_position_per_cm']*c[:,None]*(worlds[s][:,2,:3,3]-src_wrist[s])/.01).ravel())
        out.append(S['anchor_angle_per_rad']*x)
        if n>1:out.append((S['velocity_per_rad']*np.diff(delta,axis=0)).ravel())
        if n>2:out.append((S['acceleration_per_rad']*np.diff(delta,2,axis=0)).ravel())
        return np.concatenate(out)
    r0=residuals(np.zeros(size));sparsity=lil_matrix((len(r0),size),dtype=int);row=0
    if K:
        for t in ct:sparsity[row,t*12:(t+1)*12]=1;row+=1
    if cap:
        for t in cap['frames']:sparsity[row,t*12:(t+1)*12]=1;row+=1
    for k in range(2):
        for _ in range(2):
            for t in range(n):
                for _ in range(3):sparsity[row,t*12+k*6:t*12+k*6+6]=1;row+=1
    for i in range(size):sparsity[row,i]=1;row+=1
    for order in (1,2):
        if n>order:
            for t in range(n-order):
                for d in range(12):
                    for o in range(order+1):sparsity[row,(t+o)*12+d]=1
                    row+=1
    assert row==len(r0)
    b=np.radians(bound_deg);per=np.array([b/np.sqrt(3)]*3+[b]+[b/np.sqrt(2)]*2);upper=np.tile(np.r_[per,per],n)
    result=least_squares(residuals,np.zeros(size),bounds=(-upper,upper),jac_sparsity=sparsity,method='trf',x_scale='jac',max_nfev=80)
    return result.x.reshape(n,2,6),result


def angles(delta):
    return {'shoulder_deg':np.degrees(np.linalg.norm(delta[...,0:3],axis=-1)),'elbow_deg':np.degrees(np.abs(delta[...,3])),'wrist_deg':np.degrees(np.linalg.norm(delta[...,4:6],axis=-1))}


def refine(bundle,report):
    """One candidate from exact skin witnesses. The source bundle is not modified."""
    n=len(bundle.arrays['sample_times_s']);frames=np.arange(n);witnesses=witnesses_from(report,frames)
    candidate=copy.deepcopy(bundle);info={'method':'windowed multi-witness arm refinement (capsule forearm crossing)','settings':SETTINGS,'witnesses':len(witnesses),'passes':int(bool(witnesses)),'max_passes':1}
    if not witnesses:candidate.metadata['collision_refinement']=info;return candidate
    arms=Arms(bundle,frames);terms,clusters,episodes=build_terms(arms,witnesses);cap=capsule_setup(arms,witnesses,report.get('skin_capsules'))
    confidence={s:np.mean(bundle.arrays['rotation_confidence'][:,arms.idx[s]],axis=1) for s in SIDES}
    delta,result=solve(arms,terms,clusters,cap,confidence,SETTINGS['trust_region_deg'])
    estimate,_=solve(arms,terms,clusters,cap,confidence,SETTINGS['diagnostic_bound_deg'])
    quats,worlds=arms.corrected(delta)
    for s in SIDES:
        for j,q in zip(arms.idx[s],quats[s]):
            v=q.as_quat()
            for k in range(1,len(v)):
                if v[k]@v[k-1]<0:v[k]=-v[k]
            candidate.arrays['local_rotation_delta'][:,j]=v
    from .twist import distribute
    distribute(candidate)
    used={k:float(v.max()) for k,v in angles(delta).items()};needed={k:float(v.max()) for k,v in angles(estimate).items()}
    shift={s:float(np.max(np.linalg.norm(worlds[s][:,2,:3,3]-arms.source[s][:,2,:3,3],axis=1))) for s in SIDES}
    capsule_info=None
    if cap:
        _,src=arms.corrected(np.zeros((n,2,6)));s0,r0=capsule_separation(src,cap,cap['frames']);s1,r1=capsule_separation(worlds,cap,cap['frames'])
        capsule_info={'frames':int(len(cap['frames'])),'radii_m':cap['radii'],'front_arm':sorted(set(cap['front'].values())),'overlap_before_max_m':float(np.max(r0-s0)),'overlap_after_max_m':float(np.max(r1-s1))}
    info.update(solver_status=int(result.status),solver_cost=float(result.cost),increment_max_deg=used,trust_region_deg=SETTINGS['trust_region_deg'],
                required_correction_estimate_max_deg=needed,required_exceeds_trust_region=any(v>SETTINGS['trust_region_deg']+1e-6 for v in needed.values()),
                wrist_shift_max_m=shift,terms={c:int(sum(1 for x in terms if x['class']==c)) for c in ('body','hand_arm','contact')},
                contact_frames={f'{k[0]}_{k[1]}':int(v.sum()) for k,v in episodes.items()},capsule=capsule_info,quality_accepted=False)
    candidate.metadata['collision_refinement']=info
    return candidate


# ---------- acceptance (frozen gates; any failure keeps the source) ----------

def class_metrics(report):
    per={}
    for row in report.get('records',[]):
        for w in row.get('witnesses',[]):
            per.setdefault(w['class'],{});per[w['class']][row['sample']]=max(per[w['class']].get(row['sample'],0.),w['depth_m'])
    out={}
    for c in ('body','arm_arm','contact'):
        d=np.array(list(per.get(c,{}).values()))*1000
        out[c]={'frames_over_5mm':int((d>5).sum()),'max_mm':float(d.max()) if len(d) else 0.,'p95_mm':float(np.percentile(d,95)) if len(d) else 0.}
    return out


def same_set_depths(before,after,kind):
    """p95 depth over the same (frame, vertex, target) set before and after (absent = 0 mm)."""
    def table(report):
        return {(row['sample'],w.get('vertex'),w['target_attach']):w['depth_m'] for row in report.get('records',[]) for w in row.get('witnesses',[]) if w['class']==kind}
    a,b=table(before),table(after)
    if not a:return None
    keys=list(a);x=np.array([a[k] for k in keys])*1000;y=np.array([b.get(k,0.) for k in keys])*1000
    return {'points':len(keys),'p95_before_mm':float(np.percentile(x,95)),'p95_after_mm':float(np.percentile(y,95))}


def arm_dynamics(bundle):
    names=bundle.metadata['joint_names'];ids=[names.index(f'{c}_{s}') for c in CHAIN for s in SIDES];q=bundle.arrays['local_rotation_delta'][:,ids]
    step=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[1:]*q[:-1],-1)),0,1)));acc=np.abs(np.diff(step,axis=0)) if len(step)>1 else np.zeros((0,len(ids)))
    return {'max_step_deg':float(step.max()) if step.size else 0.,'max_step_change_deg':float(acc.max()) if acc.size else 0.}


def accept(source_report,candidate_report,source,candidate,reduction=.2,velocity_slack_deg=.5):
    before=class_metrics(source_report);after=class_metrics(candidate_report);gates={}
    for c in ('body','arm_arm'):
        b,a=before[c],after[c];not_worse=a['frames_over_5mm']<=b['frames_over_5mm'] and a['max_mm']<=b['max_mm']+.5 and a['p95_mm']<=b['p95_mm']+.5
        meaningful=b['p95_mm']<=5 or a['p95_mm']<=(1-reduction)*b['p95_mm']
        gates[f'{c}_not_worse']=bool(not_worse);gates[f'{c}_meaningful_reduction']=bool(meaningful)
    gates['contact_not_worse']=after['contact']['max_mm']<=before['contact']['max_mm']+.5
    names=source.metadata['joint_names'];n=len(source.arrays['sample_times_s'])
    dev={j:max(float(np.linalg.norm(evaluate(source,i)[names.index(j),:3,3]-evaluate(candidate,i)[names.index(j),:3,3])) for i in range(n)) for j in ('hand_l','hand_r','lowerarm_l','lowerarm_r')}
    gates['wrist_deviation_le_3cm']=max(dev['hand_l'],dev['hand_r'])<=.03;gates['elbow_deviation_le_3cm']=max(dev['lowerarm_l'],dev['lowerarm_r'])<=.03
    info=candidate.metadata.get('collision_refinement',{});gates['angular_gate_le_5deg']=all(v<=SETTINGS['trust_region_deg']+1e-6 for v in info.get('increment_max_deg',{}).values())
    ds,dc=arm_dynamics(source),arm_dynamics(candidate)
    gates['velocity_not_worse']=dc['max_step_deg']<=ds['max_step_deg']+velocity_slack_deg;gates['acceleration_not_worse']=dc['max_step_change_deg']<=ds['max_step_change_deg']+velocity_slack_deg
    return {'accepted':all(gates.values()),'gates':gates,'penetration_before':before,'penetration_after':after,'contact_same_set':same_set_depths(source_report,candidate_report,'contact'),
            'deviation_m':dev,'dynamics_before':ds,'dynamics_after':dc,'video_review':'pending (human)'}
