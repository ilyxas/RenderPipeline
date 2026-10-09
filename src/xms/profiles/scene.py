import json
from pathlib import Path
from .character import profile_digest


def load_scene(path):
    p=json.loads(Path(path).read_text())
    if p['schema_version']!='xms.scene.v1' or p['profile_hash']!=profile_digest(p): raise ValueError('Scene profile mismatch')
    return p
