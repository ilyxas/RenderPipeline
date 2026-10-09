"""Optional user-installed speech cue adapter; no downloads."""
import json,subprocess


def cues(wav,binary='rhubarb'):
    result=subprocess.run([binary,'-f','json',str(wav)],capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError('Rhubarb failed: '+result.stderr[-500:])
    return json.loads(result.stdout)['mouthCues']
