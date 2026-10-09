import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .bundle import AnimationBundle
from .validate import validate


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def bundle_hash(path):
    p=Path(path)
    return hashlib.sha256((file_hash(p/'metadata.json')+file_hash(p/'arrays.npz')).encode()).hexdigest()


def write_bundle(bundle,path):
    validate(bundle)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError('Published bundle is immutable: '+str(path))
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        stage=Path(temp)/'bundle'; stage.mkdir()
        (stage/'metadata.json').write_text(json.dumps(bundle.metadata,indent=2,allow_nan=False))
        np.savez_compressed(stage/'arrays.npz',**bundle.arrays)
        os.rename(stage,path)
    return bundle_hash(path)


def read_bundle(path,profile_hash=None):
    path=Path(path)
    with np.load(path/'arrays.npz',allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    return validate(AnimationBundle(json.loads((path/'metadata.json').read_text()),arrays),profile_hash)
