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
    overrides=[r for r in quality.get('look_overrides',[]) if r.get('kind')=='override'];conversions=[r for r in quality.get('look_overrides',[]) if r.get('kind')!='override']
    def change(r):return esc(f"{r.get('stage')}: {r.get('material')} {r.get('input') or r.get('change','')}: {r.get('before','')} → {r.get('after','')}"+(' (ignored: '+r['ignored']+')' if r.get('ignored') else ''))
    banner=('<div class="alert"><h2>Внешний вид изменён пайплайном</h2><p>The pipeline changed the asset\'s look; this render/GLB is not the asset as authored.</p><ul>'+''.join('<li>'+change(r)+'</li>' for r in overrides)+'</ul></div>') if overrides else ''
    warnings=''.join('<li>'+esc(v)+'</li>' for v in quality.get('warnings',[]))
    glb=quality.get('glb') or {}
    export=('<h2>GLB export</h2><pre>'+esc(json.dumps(glb,indent=2))+'</pre>'+('<h3>Real-time conversions requested by the profile</h3><ul>'+''.join('<li>'+change(r)+'</li>' for r in conversions)+'</ul>' if conversions else '')) if any(glb.values()) else ''
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>XMS development report</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;background:#15191f;color:#eee}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #555;text-align:left}.unavailable,.measured{color:#e9bf6c}.fail{color:#ff8a80}.pass{color:#80d5ae}.alert{border:2px solid #ff8a80;background:#3a1f22;padding:4px 16px;margin:16px 0}video{max-width:90%;max-height:650px}pre{white-space:pre-wrap}</style><h1>XMS baseline preview</h1><p>Status: '+esc(quality['status'])+' — development evidence, not final quality acceptance.</p>'+banner+('<h2>Warnings</h2><ul>'+warnings+'</ul>' if warnings else '')+videos+'<h2>Measurements</h2><table><tr><th>Metric</th><th>Status</th><th>Value</th><th>Unit</th><th>Reason</th></tr>'+rows+'</table>'+export+'<h2>Known limitations</h2><ul>'+limitations+'</ul><h2>Provenance</h2><pre>'+esc(json.dumps(quality.get('provenance',{}),indent=2))+'</pre>')
