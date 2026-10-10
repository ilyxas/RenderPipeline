import argparse,json
from pathlib import Path
from .animation.io import read_bundle,bundle_hash
from .config import load_config


def main():
    p=argparse.ArgumentParser(prog='xms',description='MP4 → motion observations → Xandra AnimationBundle → rendered MP4')
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bundle');bs=b.add_subparsers(dest='action',required=True);v=bs.add_parser('validate');v.add_argument('path');v.add_argument('--profile-hash')
    o=sub.add_parser('observe');o.add_argument('input');o.add_argument('--start',type=float,required=True);o.add_argument('--end',type=float,required=True);o.add_argument('--out',required=True);o.add_argument('--config')
    r=sub.add_parser('run');r.add_argument('input',nargs='?');r.add_argument('--start',type=float);r.add_argument('--end',type=float);r.add_argument('--config');r.add_argument('--job');r.add_argument('--runs-root',default='runs');r.add_argument('--output','--out',dest='output')
    r.add_argument('--channels',choices=['body','full'],default='full');r.add_argument('--character',choices=['xandra'],default='xandra');r.add_argument('--scene',choices=['bedroom'],default='bedroom');r.add_argument('--quality',choices=['preview','final'],default='preview')
    r.add_argument('--solver',choices=['baseline','video','temporal'],default='video',help='video: stable torso/arm retarget and video face/hands (default); temporal: experimental bounded window optimizer; baseline: preserved backend')
    r.add_argument('--allow-degraded-final',action='store_true',help='Explicitly render final even when the motion defect gate fails')
    r.add_argument('--face-closeup',action='store_true');r.add_argument('--audio',choices=['off','envelope','speech','singing'],default='envelope');r.add_argument('--surface-qa',choices=['off','sampled','all'],default='sampled');r.add_argument('--no-refine',action='store_true');r.add_argument('--camera',choices=['fit','fixed'],default='fit')
    rr=sub.add_parser('render');rr.add_argument('bundle');rr.add_argument('--config');rr.add_argument('--out',required=True);rr.add_argument('--fps',required=True);rr.add_argument('--frame-indices');rr.add_argument('--view',default='main',choices=['main','face','hand_l','hand_r']);rr.add_argument('--quality',choices=['preview','final'],default='preview')
    rep=sub.add_parser('report');rep.add_argument('run');rep.add_argument('--out',required=True);rep.add_argument('--config')
    args=p.parse_args()
    try:
        if args.command=='bundle':read_bundle(args.path,args.profile_hash);print('Valid AnimationBundle:',bundle_hash(args.path))
        elif args.command=='observe':
            from .observations.pose_mediapipe import observe
            c=load_config(args.config);print(observe(args.input,args.start,args.end,args.out,c['pose_model'],c['ffmpeg'],c['ffprobe']))
        elif args.command=='report':
            from .assembly.compare import compare
            from .qa.run_report import write_report
            runpath=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False);m=json.loads((runpath/'manifest.json').read_text());c=load_config(args.config,render_only=True)
            comparison=compare(m['parameters']['input'],runpath,out/'comparison',c);q=write_report(runpath,out,comparison);print(out);raise SystemExit(q['exit_code'])
        elif args.command=='render':
            from .render.process import render
            out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);indices=None if args.frame_indices is None else [int(x) for x in args.frame_indices.split(',')]
            print(render(args.bundle,out,load_config(args.config,render_only=True),args.fps,str(out)+'.log',indices,view=args.view,quality=args.quality))
        else:
            from .pipeline import run
            if args.job:
                job=json.loads(Path(args.job).read_text())
                if job['schema_version']!='xms.baseline_job.v1':raise ValueError('Unsupported job version')
                args.input=job['input'];args.start=job['start_s'];args.end=job['end_s'];args.config=job['config'];args.runs_root=job.get('runs_root','runs');args.solver=job['solver'];args.quality=job['quality'];args.channels=job.get('channels','body')
            if args.input is None:p.error('run requires INPUT (or --job)')
            if not Path(args.input).is_file():p.error('Input MP4 does not exist: '+args.input)
            if args.output and Path(args.output).exists():p.error('Output already exists; choose a new --output path: '+args.output)
            if args.solver=='temporal':
                import importlib.util
                if not importlib.util.find_spec('scipy'):p.error('Temporal solver needs SciPy. Install the temporal dependency group or use --solver video.')
            result=run(args.input,args.start,args.end,load_config(args.config),args.runs_root,args.channels,args.solver,args.quality,args.output,args.face_closeup,args.audio,args.surface_qa,args.camera,not args.no_refine,args.allow_degraded_final)
            print('RUN MANIFEST',result/'manifest.json');raise SystemExit(json.loads((result/'manifest.json').read_text()).get('result_exit_code',1))
    except (ValueError,FileNotFoundError,KeyError) as error:p.error(str(error))


if __name__=='__main__':main()
