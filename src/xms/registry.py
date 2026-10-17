"""Registered characters/scenes: which profile versions exist, which .blend each one belongs to.

A profile records only the sha256 of its asset. The registry finds the file on disk that has
that hash, so the Studio (and `xms assets`) can only switch to pairs that match, explain a
mismatch in words, and register a changed .blend as the next profile version.
Registration never edits an existing profile: it copies the base version, binds the new hash
and checks in Blender that the asset still fits (same rig / same scene collection and world).
"""
import json,os,shutil,subprocess,threading,time
from pathlib import Path
from .animation.io import file_hash
from .profiles.character import profile_digest

ROOT=Path(__file__).resolve().parents[2]
PROFILES=ROOT/'profiles'
ASSET_DIRS=(ROOT/'blender',ROOT/'asset_store/scenes')
CACHE=ROOT/'.xms/hash-cache.json'


def _cache():
    try:return json.loads(CACHE.read_text())
    except (OSError,ValueError):return {}


def hash_of(path,cache=None):
    """sha256 with a (size, mtime) cache: .blend files are hundreds of MB."""
    path=Path(path);st=path.stat();key=str(path.resolve());own=cache is None;cache=_cache() if own else cache
    row=cache.get(key)
    if row and row['size']==st.st_size and row['mtime']==st.st_mtime:return row['sha256']
    digest=file_hash(path);cache[key]={'size':st.st_size,'mtime':st.st_mtime,'sha256':digest}
    if own:_save_cache(cache)
    return digest


def _save_cache(cache):
    CACHE.parent.mkdir(parents=True,exist_ok=True);tmp=CACHE.with_name(f'{CACHE.name}.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_text(json.dumps(cache,indent=1));tmp.replace(CACHE)


def asset_files():
    return sorted(p for d in ASSET_DIRS if d.is_dir() for p in d.glob('*.blend'))


def _version_key(path):
    try:return int(path.name.lstrip('v'))
    except ValueError:return -1


def profiles():
    """All profile versions: kind, name ('xandra/v9', 'loft/v4'), path for the config, asset hash, validity."""
    rows=[]
    for kind,folder,filename in (('character','characters','character.json'),('scene','scenes','scene.json')):
        for owner in sorted(p for p in (PROFILES/folder).glob('*') if p.is_dir()):
            for version in sorted((p for p in owner.glob('v*') if (p/filename).is_file()),key=_version_key):
                data=json.loads((version/filename).read_text());valid=data.get('profile_hash')==profile_digest(data)
                rows.append({'kind':kind,'name':f'{owner.name}/{version.name}','path':str(version if kind=='character' else version/filename),
                             'asset_sha256':data.get('asset_sha256'),'valid':valid,'note':data.get('asset_note',''),'revision':data.get('profile_revision')})
    return rows


def status(config_path=None):
    """Current config pair with a plain-language verdict, every profile with its matching file, unregistered files."""
    from .config import default_config_path
    path=Path(config_path) if config_path else default_config_path();config=json.loads(path.read_text())
    cache=_cache();files={str(f):hash_of(f,cache) for f in asset_files()}
    for key in ('character_asset','scene_asset'):
        if config.get(key) and Path(config[key]).is_file() and config[key] not in files:files[config[key]]=hash_of(config[key],cache)
    _save_cache(cache);by_hash={}
    for f,h in files.items():by_hash.setdefault(h,[]).append(f)
    rows=profiles()
    for row in rows:row['files']=by_hash.get(row['asset_sha256'],[])
    registered={r['asset_sha256'] for r in rows}
    current={}
    for kind in ('character','scene'):
        profile_path=config.get(kind+'_profile','');asset=config.get(kind+'_asset','')
        row=next((r for r in rows if r['kind']==kind and Path(r['path']).resolve()==Path(profile_path).resolve()),None) if profile_path else None
        exists=bool(asset) and Path(asset).is_file();digest=files.get(asset) if exists else None
        if row is None:problem='Профиль не найден: '+profile_path
        elif not row['valid']:problem=f'Профиль {row["name"]} повреждён (хеш профиля не сходится)'
        elif not exists:problem='Файл не найден: '+asset
        elif digest!=row['asset_sha256']:
            owners=[r['name'] for r in rows if r['kind']==kind and r['asset_sha256']==digest]
            problem=f'{Path(asset).name} не тот файл, для которого записан профиль {row["name"]}'+(f'; этот файл зарегистрирован как {", ".join(owners)}' if owners else '; файл изменён после регистрации — его можно зарегистрировать как новую версию')
        else:problem=None
        current[kind]={'profile':row['name'] if row else profile_path,'asset':asset,'asset_name':Path(asset).name if asset else '','ok':problem is None,'problem':problem,'suggest':[r['name'] for r in rows if r['kind']==kind and r['valid'] and r['asset_sha256']==digest] if digest else []}
    return {'config':str(path.resolve()),'ok':all(c['ok'] for c in current.values()),'current':current,'profiles':rows,
            'unregistered':[{'path':f,'name':Path(f).name} for f,h in sorted(files.items()) if h not in registered]}


def check(config):
    """Raise a readable error when the configured assets are not the registered ones (pipeline preflight)."""
    result=status(config['_config_path'])
    if not result['ok']:
        lines=[f'{k}: {c["problem"]}'+(f' (подходит профиль {", ".join(c["suggest"])})' if c['suggest'] else '') for k,c in result['current'].items() if not c['ok']]
        raise ValueError('Registered character/scene asset hash mismatch — '+'; '.join(lines)+'. Fix it in Studio → «Персонаж и сцена» or `xms assets`.')


def _write_config(path,config):
    path=Path(path);backups=path.parent/'backups';backups.mkdir(exist_ok=True)
    shutil.copy2(path,backups/f'{path.stem}.{time.strftime("%Y%m%d-%H%M%S")}.json')
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(config,indent=2)+'\n');tmp.replace(path)


def use(name,config_path=None):
    """Point the config at profile `name` and the file that has its hash (never a mismatching pair)."""
    from .config import default_config_path
    path=Path(config_path) if config_path else default_config_path();result=status(path)
    row=next((r for r in result['profiles'] if r['name']==name),None)
    if row is None:raise ValueError('Нет такого профиля: '+name)
    if not row['valid']:raise ValueError(f'Профиль {name} повреждён')
    if not row['files']:raise ValueError(f'Для профиля {name} нет файла с его хешем в blender/ или asset_store/scenes/')
    config=json.loads(path.read_text());config[row['kind']+'_profile']=row['path'];config[row['kind']+'_asset']=row['files'][0]
    _write_config(path,config);return status(path)


def _blender_check(blender,asset,kind,base):
    script=ROOT/'blender/check_registration.py'
    r=subprocess.run([blender,'-b',str(asset),'--python-exit-code','1','--python',str(script),'--','--kind',kind,'--profile',str(base)],capture_output=True,text=True,timeout=600)
    line=next((l for l in r.stdout.splitlines() if l.startswith('XMS_CHECK ')),None)
    if r.returncode or line is None:raise ValueError('Проверка в Blender не прошла: '+(r.stdout+r.stderr).strip().splitlines()[-1][:300])
    verdict=json.loads(line[10:])
    if not verdict['ok']:raise ValueError('Файл не подходит к профилю '+str(base.name if hasattr(base,'name') else base)+': '+verdict['reason'])
    return verdict


def register(asset,base,config_path=None,activate=True):
    """New profile version for `asset`, copied from profile `base` (e.g. 'loft/v4'), checked in Blender."""
    from .config import default_config_path
    # Raw config: registering is how a broken pair gets repaired, so the current assets may be missing.
    path=Path(config_path) if config_path else default_config_path();config=json.loads(path.read_text())
    if not Path(config.get('blender','')).is_file():raise ValueError('В конфигурации не указан Blender: '+str(config.get('blender')))
    asset=Path(asset).resolve()
    if not asset.is_file() or asset.suffix!='.blend':raise ValueError('Нужен существующий .blend: '+str(asset))
    row=next((r for r in profiles() if r['name']==base),None)
    if row is None or not row['valid']:raise ValueError('Нет исправного профиля '+base)
    digest=hash_of(asset);same=[r['name'] for r in profiles() if r['kind']==row['kind'] and r['asset_sha256']==digest]
    if same:raise ValueError(f'{asset.name} уже зарегистрирован как {", ".join(same)}')
    source=Path(row['path']) if row['kind']=='character' else Path(row['path']).parent
    _blender_check(config['blender'],asset,row['kind'],source)
    owner=source.parent;target=owner/f'v{max(_version_key(p) for p in owner.glob("v*"))+1}'
    staging=owner/('.'+target.name+'.tmp');shutil.rmtree(staging,ignore_errors=True);shutil.copytree(source,staging)
    filename='character.json' if row['kind']=='character' else 'scene.json';data=json.loads((staging/filename).read_text())
    data['asset_sha256']=digest;data['profile_revision']=data.get('profile_revision',0)+1
    data['asset_note']=f'{asset.name} registered {time.strftime("%Y-%m-%d %H:%M")} from {row["name"]} (same rig/scene contents, checked in Blender). '+data.get('asset_note','')
    if row['kind']=='character':data['files']={n:file_hash(staging/n) for n in data['files']}
    data['profile_hash']=profile_digest(data);(staging/filename).write_text(json.dumps(data,indent=2));staging.rename(target)
    name=f'{owner.name}/{target.name}'
    return use(name,path) if activate else status(path)
