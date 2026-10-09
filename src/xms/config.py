"""Local registered paths; explicit config overrides persisted defaults."""
import json,os
from pathlib import Path


def default_config_path():
    root=Path(__file__).resolve().parents[2]
    candidates=[Path(os.environ['XMS_CONFIG'])] if os.environ.get('XMS_CONFIG') else [root/'.xms/config.json',Path.home()/'.config/xms/config.json',root/'runs/local-config.json']
    for p in candidates:
        if p.is_file():return p
    raise ValueError('No registered asset configuration. Set XMS_CONFIG or pass --config PATH; see README quick start.')


def load_config(path=None,render_only=False):
    path=default_config_path() if path is None else Path(path);p=json.loads(path.read_text())
    required={'blender','ffmpeg','ffprobe','pose_model','character_asset','scene_asset','character_profile','scene_profile'}
    if not required<=set(p):raise ValueError('Config missing keys: '+', '.join(sorted(required-set(p))))
    for key in required|({'face_model','hand_model'}&set(p)):
        if render_only and key in ('pose_model','face_model','hand_model'):continue
        if not Path(p[key]).is_absolute() or not Path(p[key]).exists():raise ValueError(f'Missing/nonabsolute {key}: {p[key]}. Update {path}.')
    p['_config_path']=str(path.resolve());return p
