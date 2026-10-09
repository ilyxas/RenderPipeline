"""Development QA outcomes; unavailable is never a passing measurement."""
import json,re,resource,subprocess,sys,time
from pathlib import Path

EXIT_CODES={'success':0,'failed':1,'needs_input':2,'needs_review':3,'warnings':4}

class NeedsInput(ValueError):
    """Input is ambiguous and needs an explicit user selection."""


def metric(value=None,unit=None,reason=None,passed=None):
    if value is None:return {'status':'unavailable','value':None,'unit':unit,'reason':reason or 'No evidence'}
    return {'status':'measured' if passed is None else 'pass' if passed else 'fail','value':value,'unit':unit}


def self_peak_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def measured_command(command,**kwargs):
    """One subprocess measurement; no scheduling or retries."""
    prefix=['/usr/bin/time','-l'] if sys.platform=='darwin' else []
    started=time.monotonic();r=subprocess.run(prefix+[str(v) for v in command],**kwargs)
    stderr=r.stderr or b''
    text=stderr.decode(errors='replace') if isinstance(stderr,bytes) else stderr
    match=re.search(r'(\d+)\s+maximum resident set size',text)
    return r,{'wall_s':time.monotonic()-started,'peak_rss_bytes':int(match.group(1)) if match else None,'peak_rss_reason':None if match else 'Per-process RSS unavailable on this platform','exit_status':r.returncode}


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
