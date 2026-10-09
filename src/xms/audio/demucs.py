"""Optional separate-runtime adapter. Missing runtime is a video-only fallback."""
import subprocess
from pathlib import Path


def separate(wav,python,out):
    # Never invokes the tracking environment or downloads weights implicitly.
    if not python:raise RuntimeError('Demucs runtime and preinstalled weights not configured')
    r=subprocess.run([python,'-m','demucs','--two-stems','vocals','--device','cpu','-o',str(out),str(wav)],capture_output=True,text=True,timeout=120)
    if r.returncode:raise RuntimeError('Demucs failed: '+r.stderr[-500:])
    matches=list(Path(out).glob('*/'+Path(wav).stem+'/vocals.wav'))
    if len(matches)!=1:raise RuntimeError('Demucs vocals output missing')
    return matches[0]
