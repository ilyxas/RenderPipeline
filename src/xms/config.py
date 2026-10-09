"""Explicit local paths only. No registry, scheduler, cache, or discovery."""
import json
from pathlib import Path


def load_config(path,render_only=False):
    p=json.loads(Path(path).read_text())
    required={'blender','ffmpeg','ffprobe','pose_model','character_asset','scene_asset','character_profile','scene_profile'}
    if set(p)!=required:raise ValueError('Config requires explicit paths: '+str(sorted(required)))
    for key,value in p.items():
        if render_only and key=='pose_model':continue
        if not Path(value).is_absolute() or not Path(value).exists():raise ValueError('Missing/nonabsolute configured path: '+key)
    return p
