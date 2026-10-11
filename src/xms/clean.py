"""Remove regenerable PNG frame sequences from run folders; results stay.

A finished run keeps its MP4s, AnimationBundle, GLB, observations, manifests,
logs and report. The frame folders are encoder input only (about 7/8 of the
space of a typical runs/ directory).
"""
import json,shutil,subprocess,time
from pathlib import Path

FRAME_DIRS=('frames','preview-frames','face-frames','comparison/frames')


def frame_files(run):
    for name in FRAME_DIRS:
        folder=Path(run)/name
        if folder.is_dir():yield from sorted(folder.glob('*.png'))


def prune_frames(run):
    """Pipeline hook after a successful encode: delete frame PNGs, keep frame manifests."""
    removed=0
    for f in frame_files(run):removed+=f.stat().st_size;f.unlink()
    return removed


def active_runs():
    out=subprocess.run(['pgrep','-fl','xms.cli run'],capture_output=True,text=True).stdout
    return bool(out.strip())


def runs(root):
    for manifest in sorted(Path(root).rglob('manifest.json')):
        run=manifest.parent
        try:m=json.loads(manifest.read_text())
        except (OSError,json.JSONDecodeError):continue
        if m.get('schema_version')=='xms.run.v1':yield run,m


def clean(root='runs',apply=False,delete=False,log=print):
    root=Path(root);busy=active_runs();total=0;trash=Path.home()/'.Trash'/f'xms-frames-{time.strftime("%Y%m%d-%H%M%S")}'
    for run,m in runs(root):
        if m.get('status')=='running' and busy:log(f'skip (may be rendering) {run}');continue
        files=list(frame_files(run));size=sum(f.stat().st_size for f in files)
        if not files:continue
        total+=size;log(f'{size/1e9:6.2f} GB  {len(files):5d} frames  {m.get("status")}  {run.relative_to(root)}')
        if apply:
            for f in files:
                if delete:f.unlink()
                else:
                    target=trash/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(f),target)
    action='deleted' if apply and delete else f'moved to {trash}' if apply else 'would be freed (dry run; --apply moves them to the Trash)'
    log(f'{total/1e9:.2f} GB of frame PNGs {action}')
    return total
