import hashlib
import json
from pathlib import Path
import numpy as np
from xms.animation.io import file_hash


def profile_digest(profile):
    return hashlib.sha256(json.dumps({k:v for k,v in profile.items() if k!='profile_hash'},sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_character(path):
    path=Path(path); p=json.loads((path/'character.json').read_text())
    if p['schema_version']!='xms.character.v1' or p['profile_hash']!=profile_digest(p): raise ValueError('Character profile version/hash mismatch')
    for filename,expected in p['files'].items():
        if file_hash(path/filename)!=expected: raise ValueError('Profile component hash mismatch: '+filename)
    with np.load(path/'rig.npz',allow_pickle=False) as z: rig={k:z[k] for k in z.files}
    return p,rig,json.loads((path/'face_map.json').read_text())
