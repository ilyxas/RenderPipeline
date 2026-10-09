import json
from pathlib import Path
import numpy as np
from .contract import validate


def write_observations(path,metadata,arrays):
    validate(metadata,arrays);p=Path(path);p.mkdir(parents=True,exist_ok=False)
    (p/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False));np.savez_compressed(p/'arrays.npz',**arrays)


def read_observations(path):
    p=Path(path)
    with np.load(p/'arrays.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
    return validate(json.loads((p/'metadata.json').read_text()),a)
