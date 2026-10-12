"""Stage-15 bounded collision refinement over a window, for any published bundle.

Bundle-agnostic: the source bundle (Viggle, video, temporal) is never modified; one
candidate is built from exact skin witnesses and must pass a fresh exact QA before
it may replace the source (see accept()). Design agreed with the architecture arbiter
(CGC.txt, 2026-10-12 03:02):

- variables per frame, both arms jointly: shoulder 3 (rotation vector in the upper arm
  delta frame), elbow 1 (HumanoidAlpha hinge axis), wrist 2 (flexion/deviation axes);
  torso and root fixed;
- witnesses are skin points attached to their arm chain bone; body targets are fixed,
  arm targets move with the other arm;
- classes: body and arm_arm are penetration (must exit by depth + margin); contact
  (hand skin) may sit on the surface within a tolerance and is held there (no pushing
  away beyond a separation allowance; sliding along the surface is free);
- witness residuals are pseudo-Huber and normalised per cluster (frame, class, side,
  bone pair), so a dense region cannot outvote a sparse deep one;
- temporal velocity/acceleration penalties act on the correction delta only, so the
  source's own fast motion is not smoothed; soft confidence-weighted anchors keep
  elbow/wrist positions and joint corrections near the source;
- the increment is bounded by the frozen 5°/joint trust region; a separate wide solve
  records the required-correction estimate for diagnostics only.
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
SETTINGS={'margin_m':{'body':.003,'arm_arm':.003},'contact_tolerance_m':.002,'contact_separation_m':.005,
          'class_weight':{'body':2.,'arm_arm':2.,'contact':1.},'huber_cm':.5,
          'anchor_position_per_cm':.4,'anchor_angle_per_rad':2.,'velocity_per_rad':10.,'acceleration_per_rad':10.,
          'trust_region_deg':5.,'diagnostic_bound_deg':45.,'window':160,'overlap':24}


def blender_to_bundle(v):
    v=np.asarray(v,float);return np.stack([v[...,0],v[...,2],-v[...,1]],axis=-1)


def pseudo_huber(r,f):
    return np.sign(r)*np.sqrt(2*f*f*(np.sqrt(1+(r/f)**2)-1))


def smooth_hinge(x,eps=.001):
    return .5*(x+np.sqrt(x*x+eps*eps))


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


def build_terms(arms,witnesses):
    """Attach witness points to bones in the source pose."""
    terms=[]
    for w in witnesses:
        t=w['frame'];s=w['side'];p=blender_to_bundle(w['point_blender_m']);q=blender_to_bundle(w['target_blender_m'])
        chain=lambda name:(name[-1],CHAIN.index(name[:-2]))
        ps,pj=chain(w['attach']);P=arms.source[ps][t,pj];p_local=np.linalg.solve(P,np.r_[p,1])
        if w['target_attach']=='body':qs,qj,q_local=None,None,None
        else:
            qs,qj=chain(w['target_attach']);Q=arms.source[qs][t,qj];q_local=np.linalg.solve(Q,np.r_[q,1])
        u=q-p;norm=np.linalg.norm(u);u=u/norm if norm>1e-9 else np.zeros(3)
        terms.append({**w,'p':p,'q':q,'u':u,'p_side':SIDES.index(ps),'p_joint':pj,'p_local':p_local,'q_side':None if qs is None else SIDES.index(qs),'q_joint':qj,'q_local':q_local})
    counts={}
    for x in terms:key=(x['frame'],x['class'],x['side'],x['attach'],x['target_attach']);counts[key]=counts.get(key,0)+1
    for x in terms:x['weight']=SETTINGS['class_weight'][x['class']]/np.sqrt(counts[(x['frame'],x['class'],x['side'],x['attach'],x['target_attach'])])
    return terms


def solve(arms,terms,confidence,bound_deg):
    n=len(arms.frames);size=n*12;src_elbow={s:arms.source[s][:,1,:3,3] for s in SIDES};src_wrist={s:arms.source[s][:,2,:3,3] for s in SIDES}
    T=np.array([x['frame'] for x in terms],int);W=np.array([x['weight'] for x in terms]);U=np.array([x['u'] for x in terms]).reshape(-1,3)
    P0=np.array([x['p'] for x in terms]).reshape(-1,3);Q0=np.array([x['q'] for x in terms]).reshape(-1,3);depth=np.array([x['depth_m'] for x in terms])
    cls=np.array([x['class'] for x in terms]);contact=cls=='contact';need=depth+np.array([SETTINGS['margin_m'].get(c,0) for c in cls])
    S=SETTINGS;f=S['huber_cm']
    PS=np.array([x['p_side'] for x in terms],int);PJ=np.array([x['p_joint'] for x in terms],int);PL=np.array([x['p_local'] for x in terms]).reshape(-1,4)
    moving=np.array([x['q_side'] is not None for x in terms],bool)
    QS=np.array([x['q_side'] if x['q_side'] is not None else 0 for x in terms],int);QJ=np.array([x['q_joint'] if x['q_joint'] is not None else 0 for x in terms],int)
    QL=np.array([x['q_local'] if x['q_local'] is not None else np.zeros(4) for x in terms]).reshape(-1,4)
    def residuals(x):
        delta=x.reshape(n,2,6);_,worlds=arms.corrected(delta);out=[]
        if len(terms):
            stacked=np.stack([worlds[s] for s in SIDES])
            pw=np.einsum('kij,kj->ki',stacked[PS,T,PJ],PL)[:,:3]
            qw=Q0.copy()
            if moving.any():qw[moving]=np.einsum('kij,kj->ki',stacked[QS[moving],T[moving],QJ[moving]],QL[moving])[:,:3]
            rel=np.sum(((pw-P0)-(qw-Q0))*U,axis=1)
            pen=np.where(contact,smooth_hinge((depth-S['contact_tolerance_m'])-rel),smooth_hinge(need-rel))
            out.append(pseudo_huber(W*pen/.01,f))
            out.append(np.where(contact,pseudo_huber(W*smooth_hinge(rel-depth-S['contact_separation_m'])/.01,f),0.))
        for k,s in enumerate(SIDES):
            c=confidence[s]
            out.append((S['anchor_position_per_cm']*c[:,None]*(worlds[s][:,1,:3,3]-src_elbow[s])/.01).ravel())
            out.append((S['anchor_position_per_cm']*c[:,None]*(worlds[s][:,2,:3,3]-src_wrist[s])/.01).ravel())
        out.append(S['anchor_angle_per_rad']*x)
        if n>1:out.append((S['velocity_per_rad']*np.diff(delta,axis=0)).ravel())
        if n>2:out.append((S['acceleration_per_rad']*np.diff(delta,2,axis=0)).ravel())
        return np.concatenate(out)
    # Sparsity: witness/anchor rows touch their frame; temporal rows touch neighbouring frames.
    r0=residuals(np.zeros(size));sparsity=lil_matrix((len(r0),size),dtype=int);row=0
    for block in (0,1):
        for t in T:sparsity[row,t*12:(t+1)*12]=1;row+=1
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
    result=least_squares(residuals,np.zeros(size),bounds=(-upper,upper),jac_sparsity=sparsity,method='trf',x_scale='jac',max_nfev=60)
    return result.x.reshape(n,2,6),result


def angles(delta):
    return {'shoulder_deg':np.degrees(np.linalg.norm(delta[...,0:3],axis=-1)),'elbow_deg':np.degrees(np.abs(delta[...,3])),'wrist_deg':np.degrees(np.linalg.norm(delta[...,4:6],axis=-1))}


def refine(bundle,report):
    """One candidate from exact skin witnesses. The source bundle is not modified."""
    n=len(bundle.arrays['sample_times_s']);frames=np.arange(n);witnesses=witnesses_from(report,frames)
    candidate=copy.deepcopy(bundle);info={'method':'windowed multi-witness arm refinement','settings':SETTINGS,'witnesses':len(witnesses),'passes':int(bool(witnesses)),'max_passes':1}
    if not witnesses:candidate.metadata['collision_refinement']=info;return candidate
    arms=Arms(bundle,frames);terms=build_terms(arms,witnesses)
    confidence={s:np.mean(bundle.arrays['rotation_confidence'][:,arms.idx[s]],axis=1) for s in SIDES}
    delta,result=solve(arms,terms,confidence,SETTINGS['trust_region_deg'])
    estimate,_=solve(arms,terms,confidence,SETTINGS['diagnostic_bound_deg'])
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
    info.update(solver_status=int(result.status),solver_cost=float(result.cost),increment_max_deg=used,trust_region_deg=SETTINGS['trust_region_deg'],
                required_correction_estimate_max_deg=needed,required_exceeds_trust_region=any(v>SETTINGS['trust_region_deg']+1e-6 for v in needed.values()),
                wrist_shift_max_m=shift,classes={c:int(sum(1 for w in witnesses if w['class']==c)) for c in ('body','arm_arm','contact')},quality_accepted=False)
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
    return {'accepted':all(gates.values()),'gates':gates,'penetration_before':before,'penetration_after':after,'deviation_m':dev,'dynamics_before':ds,'dynamics_after':dc,'video_review':'pending (human)'}
