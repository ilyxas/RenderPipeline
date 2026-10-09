import html,json,os
from pathlib import Path
from xms.qa.contract import write_json


def report(out,quality,video=None,comparison=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);write_json(out/'quality.json',quality)
    esc=html.escape
    rows=''.join(f'<tr><td>{esc(k)}</td><td class="{esc(v["status"])}">{esc(v["status"])}</td><td>{esc(str(v.get("value")))}</td><td>{esc(v.get("unit") or "")}</td><td>{esc(v.get("reason") or "")}</td></tr>' for k,v in quality.get('metrics',{}).items())
    videos=''
    for title,path in [('Rendered preview',video),('Source / Xandra comparison',comparison)]:
        if path:videos+=f'<h2>{title}</h2><video controls preload="metadata" src="{esc(os.path.relpath(path,out))}"></video>'
    limitations=''.join('<li>'+esc(v)+'</li>' for v in quality.get('limitations',[]))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>XMS development report</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#15191f;color:#eee}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #555;text-align:left}.unavailable,.measured{color:#e9bf6c}.fail{color:#ff8a80}.pass{color:#80d5ae}video{max-width:90%;max-height:650px}pre{white-space:pre-wrap}</style><h1>XMS baseline preview</h1><p>Status: '+esc(quality['status'])+' — development evidence, not final quality acceptance.</p>'+videos+'<h2>Measurements</h2><table><tr><th>Metric</th><th>Status</th><th>Value</th><th>Unit</th><th>Reason</th></tr>'+rows+'</table><h2>Known limitations</h2><ul>'+limitations+'</ul><h2>Provenance</h2><pre>'+esc(json.dumps(quality.get('provenance',{}),indent=2))+'</pre>')
