import json,time
from pathlib import Path
from xms.animation.io import file_hash
from xms.ingest.decode import decode
from xms.observations.io import read_observations
from .face_mediapipe import observe_face
from .hands_mediapipe import observe_hands
from xms.qa.contract import write_json,self_peak_bytes


def observe_details(video,out,config):
    out=Path(out);timeline=json.loads((out/'timeline.json').read_text());probe=json.loads((out/'probe.json').read_text());meta,body=read_observations(out/'body')
    if file_hash(video)!=meta['source_sha256']:raise ValueError('Detail observation source mismatch')
    frames,transform=decode(video,probe,timeline,config['ffmpeg'],out/'details-decode.log')
    start=time.monotonic();face=observe_face(frames,timeline,transform,config['face_model'],out/'face',meta['source_sha256'],body);face_seconds=time.monotonic()-start
    # The face detector is closed before constructing the hand detector.
    start=time.monotonic();hands=observe_hands(frames,timeline,transform,body,config['hand_model'],out/'hands',meta['source_sha256']);hand_seconds=time.monotonic()-start
    write_json(out/'details-manifest.json',{'schema_version':'xms.details.v1','face_seconds':face_seconds,'hand_seconds':hand_seconds,'python_peak_rss_bytes':self_peak_bytes(),'face_valid_frames':int(face[1]['validity'].sum()),'hand_valid_frames_by_side':hands[1]['validity'].sum(axis=0).tolist(),'face_arrays_sha256':file_hash(out/'face/arrays.npz'),'hand_arrays_sha256':file_hash(out/'hands/arrays.npz'),'face_model_sha256':face[0]['model_sha256'],'hand_model_sha256':hands[0]['model_sha256']})
    return face,hands
