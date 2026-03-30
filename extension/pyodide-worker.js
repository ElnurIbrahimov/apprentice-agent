(function(){"use strict";const h="https://cdn.jsdelivr.net/pyodide/v0.27.4/full/";let o=null,y=!1;const _=new Set(["json","csv","re","math","statistics","collections","itertools","functools","datetime","random","hashlib","base64","io","os","sys","time","typing","dataclasses","sqlite3","urllib","html","xml","pathlib","textwrap","decimal","fractions","operator","string","struct","copy","pprint","difflib","enum","abc","contextlib"]),d=new Set(["numpy","pandas","matplotlib","scipy","scikit-learn","sklearn","sympy","pillow","PIL","seaborn","networkx","statsmodels"]);[..._,...d];function a(t){self.postMessage(t)}async function x(){a({type:"loading",stage:"Downloading Python runtime (~25MB, cached after first load)..."}),importScripts(h+"pyodide.js"),a({type:"loading",stage:"Initializing Python interpreter..."}),o=await loadPyodide({indexURL:h,stdout:t=>{a({type:"output",id:"_init",block:{type:"stdout",text:t+`
`}})},stderr:t=>{a({type:"output",id:"_init",block:{type:"stderr",text:t+`
`}})}}),a({type:"loading",stage:"Setting up package manager..."}),await o.loadPackage("micropip"),await o.runPythonAsync(`
import sys
import io

# Pre-configure matplotlib for non-interactive use
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
`),y=!0,a({type:"ready"})}function v(t){const r=new Set,c=t.split(`
`);for(const e of c){const i=e.trim(),s=i.match(/^import\s+([\w.]+)/);s&&r.add(s[1].split(".")[0]);const n=i.match(/^from\s+([\w.]+)\s+import/);n&&r.add(n[1].split(".")[0])}return Array.from(r)}const w=new Set;async function k(t){const r=t.filter(e=>!_.has(e)&&d.has(e)&&!w.has(e));if(r.length===0)return;const c=o.pyimport("micropip");for(const e of r)try{const i=e==="sklearn"?"scikit-learn":e==="PIL"?"Pillow":e;await c.install(i),w.add(e)}catch(i){console.warn(`[Pyodide] Failed to install ${e}:`,i.message)}}async function P(t,r){if(!y||!o){a({type:"error",id:t,message:"Pyodide is not initialized yet."});return}const c=performance.now();let e=!0;o.setStdout({batched:s=>{a({type:"output",id:t,block:{type:"stdout",text:s+`
`}})}}),o.setStderr({batched:s=>{a({type:"output",id:t,block:{type:"stderr",text:s+`
`}})}});try{const n=v(r).filter(p=>!_.has(p)&&d.has(p));n.length>0&&(a({type:"output",id:t,block:{type:"stdout",text:`Installing packages: ${n.join(", ")}...
`}}),await k(n));const u=`
import sys as _sys
import io as _io

# Capture matplotlib plots
_aura_images = []
try:
    import matplotlib.pyplot as _plt
    _orig_show = _plt.show
    def _aura_show(*args, **kwargs):
        _buf = _io.BytesIO()
        _plt.savefig(_buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0f', edgecolor='none')
        _buf.seek(0)
        import base64
        _aura_images.append(base64.b64encode(_buf.read()).decode())
        _plt.close('all')
    _plt.show = _aura_show
except ImportError:
    pass

# Execute user code
${r}

# Capture any unsaved matplotlib figures
try:
    import matplotlib.pyplot as _plt
    if _plt.get_fignums():
        _aura_show()
except (ImportError, NameError):
    pass
`;await o.runPythonAsync(u);const m=o.globals.get("_aura_images");if(m){const p=m.toJs();for(const b of p)a({type:"output",id:t,block:{type:"image",mime:"image/png",data:b}})}const l=await o.runPythonAsync(`
import json as _json
_aura_vars = []
for _name, _val in list(globals().items()):
    if _name.startswith('_'):
        continue
    try:
        _repr = repr(_val)
        if len(_repr) > 200:
            _repr = _repr[:200] + '...'
        _aura_vars.append({'name': _name, 'type_name': type(_val).__name__, 'repr': _repr})
    except:
        pass
_json.dumps(_aura_vars)
`);if(l!=null&&l!=="")try{const p=JSON.parse(l);Array.isArray(p)&&a({type:"variables",id:t,variables:p})}catch{}const g=await o.runPythonAsync(`
_aura_tables = []
try:
    import pandas as _pd
    for _name, _val in list(globals().items()):
        if _name.startswith('_') or not isinstance(_val, _pd.DataFrame):
            continue
        if len(_val) <= 50:
            _aura_tables.append(_val.to_html(classes='aura-df', max_rows=50))
except ImportError:
    pass
import json as _json
_json.dumps(_aura_tables)
`);if(g!=null&&g!=="")try{const p=JSON.parse(g);if(Array.isArray(p))for(const b of p)a({type:"output",id:t,block:{type:"html",content:b}})}catch{}}catch(s){e=!1;const n=s.message||String(s),u=n.match(/Traceback[\s\S]*/),m=u?u[0]:n,f=m.trim().split(`
`).pop()||"",l=f.match(/^(\w+Error|Exception):\s*(.*)/);a({type:"output",id:t,block:{type:"error",ename:l?l[1]:"Error",evalue:l?l[2]:f,traceback:m}})}const i=(performance.now()-c)/1e3;a({type:"done",id:t,success:e,executionTime:i})}async function I(){o&&(await o.runPythonAsync(`
_skip = set(dir()) | {'_skip'}
for _name in list(globals().keys()):
    if not _name.startswith('_') and _name not in _skip:
        try:
            del globals()[_name]
        except:
            pass
del _skip, _name
`),a({type:"ready"}))}self.onmessage=async t=>{const r=t.data;if(!r||!r.type)return;const{type:c,id:e,code:i,packages:s}=r;try{switch(c){case"init":y||await x();break;case"execute":if(!e||typeof i!="string"){a({type:"error",id:e||"_unknown",message:"Missing id or code in execute message"});return}await P(e,i);break;case"reset":await I();break;case"install":s!=null&&s.length&&await k(s);break}}catch(n){a({type:"error",id:e||"_unknown",message:`Worker error: ${n.message||String(n)}`})}}})();
