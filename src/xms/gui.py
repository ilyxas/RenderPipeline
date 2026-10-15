"""Browser launcher for the existing CLI; one local process group.

Listens on loopback by default; `--tailscale` / `--host` add explicit addresses (never all
interfaces implicitly). Remote browsers upload the source video instead of using Finder.
"""
import hashlib
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from fractions import Fraction
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs,unquote

ROOT=Path(__file__).resolve().parents[2]
ASSETS=Path(__file__).with_name('gui_assets')
UPLOAD_SUFFIXES=('.mp4','.mov','.m4v','.3gp','.webm','.mkv','.avi')
MEDIA_TYPES={'.mov':'video/quicktime','.webm':'video/webm','.mkv':'video/x-matroska','.avi':'video/x-msvideo','.3gp':'video/3gpp'}
TAILSCALE_V4=ipaddress.ip_network('100.64.0.0/10')
TAILSCALE_V6=ipaddress.ip_network('fd7a:115c:a1e0::/48')
REMOTE_PORT=8765
PHASES={'prepare':'Подготовка','pose':'Ориентиры тела','face':'Мимика','hands':'Ориентиры кистей','solve':'Реконструкция движения','surface':'Проверка пересечений','preview':'Предварительный рендер','render':'Рендер видео','encode':'Сборка MP4 и проверка','closeup':'Рендер лица','compare':'Сравнение и отчёт'}


def video_info(path,config_path=None):
    from .config import load_config
    path=Path(path).expanduser().resolve()
    if not path.is_file():raise ValueError('Видео не найдено: '+str(path))
    config=load_config(config_path)
    result=subprocess.run([config['ffprobe'],'-v','error','-show_streams','-show_format','-of','json',str(path)],capture_output=True,text=True,timeout=30)
    if result.returncode:raise ValueError('Не удалось прочитать видео: '+result.stderr.strip())
    data=json.loads(result.stdout);videos=[s for s in data['streams'] if s['codec_type']=='video' and not s.get('disposition',{}).get('attached_pic')]
    if len(videos)!=1:raise ValueError('Нужен файл с одним видеопотоком')
    v=videos[0];duration=float(v.get('duration',data['format'].get('duration',0)))
    if not math.isfinite(duration) or duration<=0:raise ValueError('Не удалось определить длительность видео')
    rate=v.get('avg_frame_rate','0/1');fps=float(Fraction(rate)) if rate!='0/0' else 0
    from .observations.viggle import cache_dir
    viggle_cached=bool(fps) and (cache_dir(path,fps)/'metahuman_body.glb').exists()
    # Phones store portrait video as landscape plus a rotation; show the size as it plays.
    rotation=next((int(d['rotation']) for d in v.get('side_data_list',[]) if 'rotation' in d),int(v.get('tags',{}).get('rotate',0)))
    width,height=(v['height'],v['width']) if rotation%180 else (v['width'],v['height'])
    return {'path':str(path),'name':path.name,'duration':duration,'width':width,'height':height,'fps':fps,'audio':any(s['codec_type']=='audio' for s in data['streams']),'viggle_cached':viggle_cached,'viggle_key':bool(os.environ.get('VIGGLE_API_KEY'))}


def env_for(settings,base):
    """Child environment; only an explicit tick allows the paid Viggle upload for this run."""
    env=dict(base);env.pop('XMS_VIGGLE_ALLOW_UPLOAD',None)
    if settings.get('solver')=='viggle' and settings.get('viggle_upload') is True:env['XMS_VIGGLE_ALLOW_UPLOAD']='1'
    return env


def command_for(settings,runs_root,config_path=None):
    from .render.camera import parse_resolution
    path=Path(settings['input']).expanduser().resolve()
    if not path.is_file():raise ValueError('Выберите существующее видео')
    start=float(settings.get('start',0));duration=float(settings['duration'])
    if not math.isfinite(start) or not math.isfinite(duration) or start<0 or duration<=0:raise ValueError('Проверьте начало и длительность фрагмента')
    resolution=settings.get('resolution','540x960');parse_resolution(resolution)
    quality=settings.get('quality','preview');solver=settings.get('solver','video');audio=settings.get('audio','envelope');surface=settings.get('surface_qa','sampled')
    for value,choices in [(quality,('preview','final')),(solver,('video','baseline','temporal','viggle')),(audio,('off','envelope','speech','singing')),(surface,('off','sampled','all'))]:
        if value not in choices:raise ValueError('Неизвестная настройка: '+str(value))
    cmd=[sys.executable,'-u','-m','xms.cli','run',str(path),'--start',str(start),'--runs-root',str(runs_root),'--quality',quality,'--resolution',resolution,'--solver',solver,'--audio',audio,'--surface-qa',surface,'--channels','body' if settings.get('body_only') else 'full','--camera','fixed' if settings.get('fixed_camera') else 'fit']
    if not settings.get('to_end'):cmd+=['--end',str(start+duration)]
    if settings.get('face_closeup') and not settings.get('body_only'):cmd+=['--face-closeup']
    if settings.get('no_refine'):cmd+=['--no-refine']
    if config_path:cmd+=['--config',str(config_path)]
    output=settings.get('output','').strip()
    if output:
        output=Path(output).expanduser().resolve()
        if output.exists():raise ValueError('Выходной файл уже существует. Выберите другое имя.')
        if output.suffix.lower()!='.mp4':raise ValueError('Выходной файл должен иметь расширение .mp4')
        cmd+=['--output',str(output)]
    return cmd


class Runner:
    def __init__(self,runs_root=None,command_prefix=None):
        self.runs_root=Path(runs_root or ROOT/'runs').resolve();self.command_prefix=command_prefix
        self.lock=threading.RLock();self.process=None;self.worker=None;self.logs=deque(maxlen=160);self.run_dir=None;self.selected_input=None;self.cancelled=False;self.phase='prepare';self.completed=None;self.total=None;self.log_offsets={};self.state={'status':'idle','error':None,'outputs':{}}

    def start(self,settings):
        with self.lock:
            if self.state['status'] in ('running','stopping'):raise ValueError('Запуск уже выполняется')
            config_path=settings.get('config','').strip() or None
            cmd=command_for(settings,self.runs_root,config_path)
            if self.command_prefix:cmd=self.command_prefix+cmd[4:]
            env=env_for(settings,os.environ);env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','');env['PYTHONUNBUFFERED']='1'
            self.runs_root.mkdir(parents=True,exist_ok=True)
            self.process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            self.cancelled=False;self.run_dir=None;self.logs.clear();self.log_offsets={};self.phase='prepare';self.completed=None;self.total=None
            self.state={'status':'running','error':None,'outputs':{},'started_at':time.time(),'settings':settings}
            self.worker=threading.Thread(target=self._read,daemon=True);self.worker.start()

    def _phase(self,phase):
        if self.phase!=phase:self.phase=phase;self.completed=None;self.total=None

    def consume(self,line):
        with self.lock:
            line=line.rstrip()
            if line.startswith('XMS_PROGRESS '):
                _,phase,done,total=line.split();self._phase(phase);self.completed=int(done);self.total=int(total);return
            self.logs.append(line)
            if self.run_dir:
                with (self.run_dir/'gui.log').open('a') as f:f.write(line+'\n')
            if line.startswith('RUN '):
                candidate=Path(line[4:]).resolve()
                if candidate.is_relative_to(self.runs_root) and candidate.is_dir():self.run_dir=candidate
            for prefix,phase in [('OBSERVE ','pose'),('SOLVE ','solve'),('SURFACE QA','surface'),('FINAL preflight','preview'),('RENDER face','closeup'),('RENDER ','render'),('ENCODE ','encode'),('COMPARE ','compare')]:
                if line.startswith(prefix):self._phase(phase);break
            if line.startswith('ERROR '):self.state['error']=line[6:]

    def _read(self):
        process=self.process
        try:
            for line in process.stdout:self.consume(line)
            code=process.wait()
            with self.lock:
                self.state['exit_code']=code
                if self.cancelled:
                    self.state['status']='stopping' if self.state['status']=='stopping' else 'cancelled'
                elif code<0 or code==130:
                    # Killed outside the Stop button (signal or interrupt): stopped, not failed.
                    self.state['status']='cancelled';self.state['error']=f'Запуск остановлен извне (код {code}).';self._mark_cancelled('Stopped outside XMS GUI')
                else:
                    self.state['status']='complete' if code==0 else 'failed'
                    if code and not self.state['error']:self.state['error']='Запуск завершился с ошибкой. Подробности в журнале.'
                self.state['finished_at']=time.time()
        finally:process.stdout.close()

    def stop(self):
        with self.lock:
            if self.state['status']!='running':return
            self.cancelled=True;self.state['status']='stopping';process=self.process
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:pass
        # Kill remaining descendants even when the CLI exited before Blender.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        self.worker.join(timeout=3)
        with self.lock:
            self.state['status']='cancelled';self.state['finished_at']=time.time();self._mark_cancelled('Stopped from XMS GUI')

    def _mark_cancelled(self,reason):
        if not self.run_dir:return
        manifest=self.run_dir/'manifest.json'
        try:
            m=json.loads(manifest.read_text())
            if m.get('status')=='running':m.update(status='cancelled',result_exit_code=130,error=reason);manifest.write_text(json.dumps(m,indent=2))
        except (FileNotFoundError,json.JSONDecodeError):pass

    def snapshot(self):
        with self.lock:
            if self.run_dir:
                try:
                    m=json.loads((self.run_dir/'manifest.json').read_text());self.state['outputs']=m.get('outputs',{});self.state['quality_status']=m.get('quality_status')
                    if m.get('error') and self.state['status']=='failed':self.state['error']=m['error']
                except (FileNotFoundError,json.JSONDecodeError):pass
                if self.phase in ('preview','render','closeup') and self.state['status']=='running':
                    name={'preview':'preview-blender.log','render':'blender.log','closeup':'face-blender.log'}[self.phase];path=self.run_dir/'logs'/name
                    try:
                        with path.open('rb') as f:
                            f.seek(self.log_offsets.get(name,0));text=f.read().decode(errors='replace');self.log_offsets[name]=f.tell()
                        for line in text.splitlines():
                            if line.startswith('XMS_FRAME '):self.completed=int(line.split()[1])+1
                        t=json.loads((self.run_dir/'observations/timeline.json').read_text());self.total=t['frame_count']
                    except (FileNotFoundError,json.JSONDecodeError,ValueError):pass
            now=self.state.get('finished_at',time.time());elapsed=now-self.state.get('started_at',now)
            return {**self.state,'phase':self.phase,'phase_label':PHASES.get(self.phase,self.phase),'completed':self.completed,'total':self.total,'elapsed':elapsed,'run_dir':str(self.run_dir) if self.run_dir else None,'logs':list(self.logs)}

    def open_output(self,kind):
        state=self.snapshot();paths={**state['outputs'],'folder':state['run_dir']}
        path=paths.get(kind)
        if not path or not Path(path).exists():raise ValueError('Этот результат ещё не готов')
        subprocess.Popen(['open' if sys.platform=='darwin' else 'xdg-open',path],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def choose_file(kind):
    if sys.platform!='darwin':raise ValueError('Системный выбор файлов доступен на macOS. Введите полный путь к файлу.')
    script='POSIX path of (choose file with prompt "Выберите видео для Xandra")' if kind=='input' else 'POSIX path of (choose file name with prompt "Сохранить Xandra MP4" default name "xandra.mp4")'
    # Finder hosts the dialog visibly, rather than an invisible background
    # osascript process, and returns the original path without copying the video.
    script='tell application "Finder"\nactivate\n'+script+'\nend tell'
    result=subprocess.run(['osascript','-e',script],capture_output=True,text=True,timeout=300)
    if result.returncode:
        if '-128' in result.stderr:return None
        raise ValueError('Не удалось открыть системное окно выбора файла: '+result.stderr.strip())
    return result.stdout.strip()


def _tailscale_cli():
    for candidate in (shutil.which('tailscale'),'/Applications/Tailscale.app/Contents/MacOS/Tailscale','/opt/homebrew/bin/tailscale','/usr/local/bin/tailscale'):
        if candidate and Path(candidate).is_file():return candidate
    return None


def _run_text(cmd):
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
        return result.stdout if result.returncode==0 else ''
    except (OSError,subprocess.SubprocessError):return ''


def is_tailscale_address(text):
    try:address=ipaddress.ip_address(text.split('%')[0])
    except ValueError:return False
    return address in (TAILSCALE_V4 if address.version==4 else TAILSCALE_V6)


def tailscale_addresses():
    """This machine's Tailscale IPv4 addresses: `tailscale ip -4`, else 100.64/10 addresses on the interfaces."""
    cli=_tailscale_cli();text=_run_text([cli,'ip','-4']) if cli else ''
    if not text.strip():text=_run_text(['ifconfig']) or _run_text(['ip','-4','addr'])
    found=[]
    for token in re.findall(r'(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])',text):
        if is_tailscale_address(token) and token not in found:found.append(token)
    return found


def tailscale_dns_name():
    cli=_tailscale_cli()
    try:return json.loads(_run_text([cli,'status','--json']) or '{}').get('Self',{}).get('DNSName','').rstrip('.') if cli else ''
    except (json.JSONDecodeError,AttributeError):return ''


def resolve_hosts(hosts=(),tailscale=False):
    """Addresses to bind: loopback always, plus explicit --host values and the Tailscale IPs."""
    result=['127.0.0.1']
    for host in hosts:
        if host in ('0.0.0.0','::'):raise ValueError('Открытие на всех интерфейсах не поддерживается: укажите конкретный адрес или --tailscale')
        try:ipaddress.ip_address(host)
        except ValueError:raise ValueError('--host ожидает IP-адрес: '+host) from None
        if host not in result:result.append(host)
    if tailscale:
        found=tailscale_addresses()
        if not found:raise ValueError('Tailscale-адрес не найден. Проверьте, что Tailscale запущен и подключён (tailscale ip -4).')
        result+=[a for a in found if a not in result]
    return result


def safe_upload_name(name):
    name=Path(unquote(name or '').replace('\\','/')).name
    suffix=Path(name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:raise ValueError('Поддерживаются видео: '+', '.join(UPLOAD_SUFFIXES))
    stem=re.sub(r'[^\w.\- ]+','_',Path(name).stem).strip(' ._')[:80] or 'video'
    return stem+suffix


def receive_upload(stream,size,name,folder,max_bytes):
    """Stream an uploaded video into `folder`, named by content hash so a repeated upload reuses the file."""
    name=safe_upload_name(name)
    if size<=0:raise ValueError('Пустой файл')
    if size>max_bytes:raise ValueError(f'Файл больше лимита {max_bytes//2**20} МБ (XMS_GUI_MAX_UPLOAD_MB)')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(folder).free<size+2**30:raise ValueError('Недостаточно места на диске для загрузки')
    digest=hashlib.sha256();partial=folder/f'.upload-{secrets.token_hex(8)}.part';remaining=size
    try:
        with partial.open('wb') as f:
            while remaining:
                chunk=stream.read(min(2**20,remaining))
                if not chunk:raise ValueError('Загрузка прервана')
                f.write(chunk);digest.update(chunk);remaining-=len(chunk)
        target=folder/f'{digest.hexdigest()[:12]}-{name}'
        if target.exists():partial.unlink()
        else:partial.replace(target)
        return target
    finally:
        if partial.exists():partial.unlink()


def serve(port=0,open_browser=True,hosts=('127.0.0.1',),uploads_root=None,max_upload_mb=None):
    runner=Runner();token=secrets.token_urlsafe(32)
    uploads=Path(uploads_root or runner.runs_root/'uploads')
    max_bytes=int(max_upload_mb or os.environ.get('XMS_GUI_MAX_UPLOAD_MB') or 8192)*2**20
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,data,status=200,content_type='application/json'):
            body=json.dumps(data,ensure_ascii=False).encode() if content_type=='application/json' else data
            self.send_response(status);self.send_header('Content-Type',content_type+'; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def remote(self):
            try:return not ipaddress.ip_address(self.client_address[0].split('%')[0]).is_loopback
            except ValueError:return True
        def do_GET(self):
            parsed=urlparse(self.path);path=parsed.path
            if path=='/':self.send((ASSETS/'index.html').read_text().replace('__TOKEN__',token).replace('__REMOTE__','1' if self.remote() else '0').encode(),content_type='text/html')
            elif path in ('/app.js','/style.css'):self.send((ASSETS/path[1:]).read_bytes(),content_type='application/javascript' if path.endswith('.js') else 'text/css')
            elif path=='/api/state' and self.headers.get('X-XMS-Token')==token:self.send(runner.snapshot())
            elif path=='/api/media' and parse_qs(parsed.query).get('token')==[token]:
                kind=parse_qs(parsed.query).get('kind',['input'])[0]
                media=runner.selected_input if kind=='input' else runner.snapshot()['outputs'].get(kind)
                if media and Path(media).suffix.lower() not in UPLOAD_SUFFIXES:self.send({'error':'Not a video'},404);return
                if not media or not Path(media).is_file():self.send({'error':'Video not ready'},404);return
                size=Path(media).stat().st_size;start=0;end=size-1;partial=False
                try:
                    header=self.headers.get('Range')
                    if header:
                        a,b=header.removeprefix('bytes=').split('-');start=int(a) if a else max(0,size-int(b));end=min(int(b),size-1) if b and a else size-1;partial=True
                        if start<0 or start>end:raise ValueError
                except ValueError:self.send({'error':'Invalid range'},416);return
                self.send_response(206 if partial else 200);self.send_header('Content-Type',MEDIA_TYPES.get(Path(media).suffix.lower(),'video/mp4'));self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
                if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
                self.end_headers()
                try:
                    with Path(media).open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(1024*1024,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
            else:self.send({'error':'Not found'},404)
        def do_POST(self):
            if self.headers.get('X-XMS-Token')!=token:self.send({'error':'Invalid session'},403);return
            path=urlparse(self.path).path
            try:
                if path=='/api/upload':
                    if 'Content-Length' not in self.headers:self.send({'error':'Нужен Content-Length'},411);return
                    try:target=receive_upload(self.rfile,int(self.headers['Content-Length']),self.headers.get('X-XMS-Filename',''),uploads,max_bytes)
                    except ValueError:
                        self.close_connection=True;raise
                    try:info=video_info(target,self.headers.get('X-XMS-Config') or None)
                    except ValueError:
                        target.unlink(missing_ok=True);raise
                    runner.selected_input=info['path'];self.send(info);return
                size=int(self.headers.get('Content-Length','0'))
                if size>65536:raise ValueError('Request too large')
                data=json.loads(self.rfile.read(size) or b'{}')
                if path in ('/api/pick','/api/open') and self.remote():raise ValueError('Это действие открывает окно на Маке и недоступно с другого устройства')
                if path=='/api/probe':
                    info=video_info(data['path'],data.get('config') or None);runner.selected_input=info['path'];self.send(info)
                elif path=='/api/pick':self.send({'path':choose_file(data.get('kind','input'))})
                elif path=='/api/run':
                    info=video_info(data['input'],data.get('config') or None)
                    if float(data['start'])+float(data['duration'])>info['duration']+.001:raise ValueError('Фрагмент выходит за длительность видео')
                    runner.start(data);self.send(runner.snapshot())
                elif path=='/api/stop':runner.stop();self.send(runner.snapshot())
                elif path=='/api/open':runner.open_output(data['kind']);self.send({'ok':True})
                else:self.send({'error':'Not found'},404)
            except (ValueError,KeyError,OSError,subprocess.SubprocessError) as error:self.send({'error':str(error)},400)
    hosts=list(hosts) or ['127.0.0.1']
    if not port and any(h!='127.0.0.1' for h in hosts):port=REMOTE_PORT  # stable address for a phone bookmark
    servers=[]
    try:
        for host in hosts:
            servers.append(ThreadingHTTPServer((host,port),Handler) if ':' not in host else _v6_server(host,port,Handler));port=servers[0].server_port
    except OSError as error:
        for s in servers:s.server_close()
        raise ValueError(f'Не удалось открыть {host}:{port}: {error}') from None
    def url_for(host):return f'http://[{host}]:{port}/' if ':' in host else f'http://{host}:{port}/'
    url=url_for(hosts[0]);print('XMS GUI '+url,flush=True)
    for host in hosts[1:]:print(('XMS GUI Tailscale ' if is_tailscale_address(host) else 'XMS GUI ')+url_for(host),flush=True)
    name=tailscale_dns_name() if any(is_tailscale_address(h) for h in hosts) else ''
    if name:print(f'XMS GUI Tailscale http://{name}:{port}/',flush=True)
    server=servers[0]
    for extra in servers[1:]:threading.Thread(target=extra.serve_forever,kwargs={'poll_interval':.3},daemon=True).start()
    if open_browser:webbrowser.open(url)
    def terminate(*_):raise KeyboardInterrupt
    # SIGHUP: the launcher's Terminal window was closed; the run lives in its own session and must be stopped here.
    old_handlers={s:signal.signal(s,terminate) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:pass
    finally:
        runner.stop()
        for extra in servers[1:]:extra.shutdown()
        for s in servers:s.server_close()
        for s,h in old_handlers.items():signal.signal(s,h)


def _v6_server(host,port,handler):
    import socket
    class V6(ThreadingHTTPServer):address_family=socket.AF_INET6
    return V6((host,port),handler)


if __name__=='__main__':serve()
