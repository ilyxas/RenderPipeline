"""Viggle Motion 3D client: one paid extraction per source video, cached by content hash.

The API key is read only from VIGGLE_API_KEY. Uploading sends the video to an
external paid service, so a cache miss fails unless XMS_VIGGLE_ALLOW_UPLOAD=1.
A started job is recorded before polling and resumed instead of re-submitted.
"""
import json,os,time,uuid,urllib.error,urllib.request
from pathlib import Path
from xms.animation.io import file_hash

API='https://apis.viggle.ai/v1'
# Viggle's CDN rejects the default Python-urllib agent (Cloudflare error 1010).
AGENT='xms-renderpipeline/0.1'
RIGS=('metahuman_body','metahuman','mixamo')


def cache_root():
    return Path(os.environ.get('XMS_VIGGLE_CACHE',Path.home()/'.cache/xms/viggle'))


def cache_dir(video,fps):
    return cache_root()/f'{file_hash(video)}-fps{fps:g}'


def request(method,path,key,body=None,headers=None):
    req=urllib.request.Request(API+path,method=method,data=body,headers={'Authorization':'Bearer '+key,'User-Agent':AGENT,**(headers or {})})
    try:
        with urllib.request.urlopen(req,timeout=120) as r:return json.loads(r.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Viggle {method} {path}: HTTP {error.code} {error.read()[:300]!r}') from None


def task_id(video,fps):
    # Viggle stores task_id (and the upload filename) in 50-character columns.
    return f'xms-{file_hash(video)[:32]}-{fps:g}'


def multipart(fields,files):
    boundary=uuid.uuid4().hex;parts=[]
    for name,value in fields.items():parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name,path in files.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="source{Path(path).suffix}"\r\nContent-Type: video/mp4\r\n\r\n'.encode()+Path(path).read_bytes()+b'\r\n')
    return b''.join(parts)+f'--{boundary}--\r\n'.encode(),{'Content-Type':'multipart/form-data; boundary='+boundary}


def fetch(video,fps,log=print,poll_s=5,timeout_s=1800):
    """Path of the cached MetaHuman-body GLB for this video and fps; extracts it on a cache miss."""
    folder=cache_dir(video,fps);target=folder/'metahuman_body.glb'
    if target.exists():log(f'VIGGLE cache hit {folder}');return target
    if os.environ.get('XMS_VIGGLE_ALLOW_UPLOAD')!='1':
        raise RuntimeError(f'Viggle result not cached ({folder}). Uploading is paid and sends the video externally; set XMS_VIGGLE_ALLOW_UPLOAD=1 to allow it.')
    key=os.environ.get('VIGGLE_API_KEY')
    if not key:raise RuntimeError('VIGGLE_API_KEY is not set')
    folder.mkdir(parents=True,exist_ok=True);state=folder/'motion.json'
    if state.exists():motion=json.loads(state.read_text());log(f"VIGGLE resume {motion['id']}")
    else:
        body,headers=multipart({'type':'glb','target_fps':f'{fps:g}','task_id':task_id(video,fps)},{'motion_video':video})
        motion=request('POST','/motions',key,body,headers);state.write_text(json.dumps(motion,indent=2));log(f"VIGGLE submitted {motion['id']}")
    deadline=time.monotonic()+timeout_s
    while motion['status'] not in ('ready','failed'):
        if time.monotonic()>deadline:raise RuntimeError(f"Viggle job {motion['id']} not ready after {timeout_s}s; rerun to resume")
        time.sleep(poll_s);motion=request('GET','/motions/'+motion['id'],key)
    (folder/'motion_final.json').write_text(json.dumps(motion,indent=2))
    if motion['status']=='failed':raise RuntimeError(f"Viggle job {motion['id']} failed: {motion.get('error')}")
    export=request('GET',f"/motions/{motion['id']}/export?download_type=glb",key)
    for rig in RIGS:
        url=export.get(rig+'_url')
        if url:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':AGENT}),timeout=300) as r:(folder/f'{rig}.glb.part').write_bytes(r.read())
            (folder/f'{rig}.glb.part').rename(folder/f'{rig}.glb')
    if not target.exists():raise RuntimeError(f"Viggle export for {motion['id']} has no MetaHuman-body GLB")
    (folder/'source.json').write_text(json.dumps({'video':str(Path(video).resolve()),'video_sha256':file_hash(video),'target_fps':fps,'motion_id':motion['id']},indent=2))
    log(f'VIGGLE downloaded {folder}');return target
