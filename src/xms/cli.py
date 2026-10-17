import argparse,json,os
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
    r.add_argument('--solver',choices=['baseline','video','temporal','viggle'],default='video',help='video: stable torso/arm retarget and video face/hands (default); temporal: experimental bounded window optimizer; baseline: preserved backend; viggle: experimental external Viggle mocap (body, head, fingers; face morphs from video with --channels full; uploads need VIGGLE_API_KEY and XMS_VIGGLE_ALLOW_UPLOAD=1, cached results do not)')
    r.add_argument('--allow-degraded-final',action='store_true',help='Explicitly render final even when the motion defect gate fails')
    r.add_argument('--face-closeup',action='store_true');r.add_argument('--audio',choices=['off','envelope','speech','singing'],default='envelope');r.add_argument('--surface-qa',choices=['off','sampled','all'],default='sampled');r.add_argument('--no-refine',action='store_true');r.add_argument('--camera',choices=['fit','fixed'],default='fit')
    r.add_argument('--engine',choices=['eevee','cycles'],default='eevee',help='eevee: fast (default); cycles: path traced with denoiser and motion blur, for finals (much slower)');r.add_argument('--resolution',help='Even WIDTHxHEIGHT; independent of render quality (e.g. 720x1280)')
    r.add_argument('--keep-frames',action='store_true',help='Keep the PNG frame sequences after a successful encode (removed by default)')
    cl=sub.add_parser('clean',help='Remove regenerable PNG frames from run folders (dry run unless --apply)');cl.add_argument('--runs-root',default='runs');cl.add_argument('--apply',action='store_true',help='Move frame PNGs to the Trash');cl.add_argument('--delete',action='store_true',help='With --apply: delete permanently instead of moving to the Trash')
    g=sub.add_parser('gui',help='Open the local graphical pipeline launcher');g.add_argument('--port',type=int,default=0,help='Default: random on loopback, 8765 when another address is added');g.add_argument('--no-browser',action='store_true');g.add_argument('--tailscale',action='store_true',help="Also listen on this Mac's Tailscale address (phone access over the tailnet)");g.add_argument('--host',action='append',default=[],help='Also listen on this IP address (repeatable); 127.0.0.1 is always included')
    rr=sub.add_parser('render');rr.add_argument('bundle');rr.add_argument('--config');rr.add_argument('--out',required=True);rr.add_argument('--fps',required=True);rr.add_argument('--frame-indices');rr.add_argument('--view',default='main',choices=['main','face','hand_l','hand_r']);rr.add_argument('--quality',choices=['preview','final'],default='preview');rr.add_argument('--engine',choices=['eevee','cycles'],default='eevee')
    a=sub.add_parser('assets',help='Registered character/scene profiles and their .blend files');a.add_argument('action',choices=['status','use','register']);a.add_argument('name',nargs='?',help="use: profile like loft/v4; register: path to .blend");a.add_argument('--base',help="register: profile to copy, e.g. xandra/v9");a.add_argument('--config')
    rep=sub.add_parser('report');rep.add_argument('run');rep.add_argument('--out',required=True);rep.add_argument('--config')
    args=p.parse_args()
    try:
        if args.command=='gui':
            from .gui import serve,resolve_hosts
            tailscale=args.tailscale or os.environ.get('XMS_GUI_TAILSCALE')=='1'
            serve(args.port,not args.no_browser,resolve_hosts(args.host,tailscale))
        elif args.command=='assets':
            from . import registry
            if args.action!='status' and not args.name:p.error('assets '+args.action+' needs a name')
            if args.action=='register' and not args.base:p.error('assets register needs --base PROFILE')
            s=registry.use(args.name,args.config) if args.action=='use' else registry.register(args.name,args.base,args.config) if args.action=='register' else registry.status(args.config)
            for kind,c in s['current'].items():print(f"{kind:9} {c['profile']:12} {c['asset_name']:30} {'OK' if c['ok'] else 'ПРОБЛЕМА: '+c['problem']}")
            for u in s['unregistered']:print('не зарегистрирован:',u['path'])
            if not s['ok']:raise SystemExit(1)
        elif args.command=='clean':
            from .clean import clean
            clean(args.runs_root,args.apply,args.delete)
        elif args.command=='bundle':read_bundle(args.path,args.profile_hash);print('Valid AnimationBundle:',bundle_hash(args.path))
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
            print(render(args.bundle,out,load_config(args.config,render_only=True),args.fps,str(out)+'.log',indices,view=args.view,quality=args.quality,engine=args.engine))
        else:
            from .pipeline import run
            import signal
            def interrupt(*_):raise KeyboardInterrupt
            # Terminal close (SIGHUP) or kill (SIGTERM) end the run like Ctrl+C, so its manifest says cancelled.
            for sig in (signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
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
            from .render.camera import parse_resolution
            resolution=parse_resolution(args.resolution) if args.resolution else None
            try:result=run(args.input,args.start,args.end,load_config(args.config),args.runs_root,args.channels,args.solver,args.quality,args.output,args.face_closeup,args.audio,args.surface_qa,args.camera,not args.no_refine,args.allow_degraded_final,resolution,args.keep_frames,args.engine)
            except KeyboardInterrupt:print('RUN CANCELLED',flush=True);raise SystemExit(130)
            print('RUN MANIFEST',result/'manifest.json');raise SystemExit(json.loads((result/'manifest.json').read_text()).get('result_exit_code',1))
    except (ValueError,FileNotFoundError,KeyError) as error:p.error(str(error))


if __name__=='__main__':main()
