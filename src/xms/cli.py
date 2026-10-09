import argparse,json
from pathlib import Path
from .animation.io import read_bundle,bundle_hash
from .config import load_config


def main():
    p=argparse.ArgumentParser(prog='xms');sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('bundle');bs=b.add_subparsers(dest='action',required=True)
    v=bs.add_parser('validate');v.add_argument('path');v.add_argument('--profile-hash')
    o=sub.add_parser('observe');o.add_argument('input');o.add_argument('--start',type=float,required=True);o.add_argument('--end',type=float,required=True);o.add_argument('--out',required=True);o.add_argument('--config',required=True)
    r=sub.add_parser('run');r.add_argument('input',nargs='?');r.add_argument('--start',type=float);r.add_argument('--end',type=float);r.add_argument('--config');r.add_argument('--job');r.add_argument('--runs-root',default='runs')
    r.add_argument('--channels',choices=['body','full'],default='body')
    for name,value in [('character','xandra'),('scene','bedroom'),('quality','preview'),('solver','baseline')]:r.add_argument('--'+name,choices=[value],default=value)
    rr=sub.add_parser('render');rr.add_argument('bundle');rr.add_argument('--config',required=True);rr.add_argument('--out',required=True);rr.add_argument('--fps',required=True);rr.add_argument('--frame-indices')
    rr.add_argument('--view',default='main',choices=['main','face','hand_l','hand_r'])
    rep=sub.add_parser('report');rep.add_argument('run');rep.add_argument('--out',required=True);rep.add_argument('--config',required=True)
    args=p.parse_args()
    if args.command=='bundle':
        read_bundle(args.path,args.profile_hash);print('Valid AnimationBundle:',bundle_hash(args.path))
    elif args.command=='observe':
        from .observations.pose_mediapipe import observe
        c=load_config(args.config);print(observe(args.input,args.start,args.end,args.out,c['pose_model'],c['ffmpeg'],c['ffprobe']))
    elif args.command=='report':
        from .assembly.compare import compare
        from .qa.run_report import write_report
        runpath=Path(args.run);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
        m=json.loads((runpath/'manifest.json').read_text());c=load_config(args.config,render_only=True)
        comparison=compare(m['parameters']['input'],runpath,out/'comparison',c)
        q=write_report(runpath,out,comparison);print(out);raise SystemExit(q['exit_code'])
    elif args.command=='render':
        from .render.process import render
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        indices=None if args.frame_indices is None else [int(x) for x in args.frame_indices.split(',')]
        print(render(args.bundle,out,load_config(args.config,render_only=True),args.fps,str(out)+'.log',indices,view=args.view))
    else:
        from .pipeline import run
        if args.job:
            job=json.loads(Path(args.job).read_text())
            if job['schema_version']!='xms.baseline_job.v1':raise ValueError('Unsupported job version')
            for k,v in [('character','xandra'),('scene','bedroom'),('quality','preview'),('solver','baseline')]:
                if job[k]!=v:raise ValueError('Unsupported Stage 4 job option: '+k)
            args.input=job['input'];args.start=job['start_s'];args.end=job['end_s'];args.config=job['config'];args.runs_root=job.get('runs_root','runs')
        if args.input is None or args.start is None or args.end is None or args.config is None:p.error('run requires INPUT, --start, --end and --config (or --job)')
        result=run(args.input,args.start,args.end,load_config(args.config),args.runs_root,channels=args.channels)
        print(result);raise SystemExit(json.loads((result/'manifest.json').read_text()).get('result_exit_code',0))


if __name__=='__main__':main()
