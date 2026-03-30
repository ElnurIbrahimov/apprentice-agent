/**
 * Pyodide Web Worker — Runs Python in-browser via WebAssembly.
 *
 * Messages IN:
 *   { type: 'execute', id: string, code: string }
 *   { type: 'reset' }
 *   { type: 'install', packages: string[] }
 *
 * Messages OUT:
 *   { type: 'ready' }
 *   { type: 'loading', stage: string }
 *   { type: 'output', id: string, block: OutputBlock }
 *   { type: 'variables', id: string, variables: VariableInfo[] }
 *   { type: 'done', id: string, success: boolean, executionTime: number }
 *   { type: 'error', id: string, message: string }
 */

declare const self: Worker & typeof globalThis;
declare function importScripts(...urls: string[]): void;
declare const loadPyodide: any;

const PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.4/full/';

let pyodide: any = null;
let isReady = false;

// Standard library modules — don't need micropip install
const STDLIB = new Set([
  'json', 'csv', 're', 'math', 'statistics', 'collections',
  'itertools', 'functools', 'datetime', 'random', 'hashlib',
  'base64', 'io', 'os', 'sys', 'time', 'typing', 'dataclasses',
  'sqlite3', 'urllib', 'html', 'xml', 'pathlib', 'textwrap',
  'decimal', 'fractions', 'operator', 'string', 'struct',
  'copy', 'pprint', 'difflib', 'enum', 'abc', 'contextlib',
]);

// Third-party packages available via micropip in Pyodide
const MICROPIP_PACKAGES = new Set([
  'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'sklearn',
  'sympy', 'pillow', 'PIL', 'seaborn', 'networkx', 'statsmodels',
]);

// All packages Pyodide can handle (stdlib + micropip)
const PYODIDE_PACKAGES = new Set([...STDLIB, ...MICROPIP_PACKAGES]);

function post(msg: any) {
  self.postMessage(msg);
}

async function initPyodide() {
  post({ type: 'loading', stage: 'Downloading Python runtime (~25MB, cached after first load)...' });

  // Load Pyodide from CDN
  importScripts(PYODIDE_CDN + 'pyodide.js');

  post({ type: 'loading', stage: 'Initializing Python interpreter...' });
  pyodide = await loadPyodide({
    indexURL: PYODIDE_CDN,
    stdout: (text: string) => {
      // Will be overridden per-execution, but this is the default
      post({ type: 'output', id: '_init', block: { type: 'stdout', text: text + '\n' } });
    },
    stderr: (text: string) => {
      post({ type: 'output', id: '_init', block: { type: 'stderr', text: text + '\n' } });
    },
  });

  // Load micropip for package management
  post({ type: 'loading', stage: 'Setting up package manager...' });
  await pyodide.loadPackage('micropip');

  // Set up matplotlib backend for non-interactive (Agg) rendering
  await pyodide.runPythonAsync(`
import sys
import io

# Pre-configure matplotlib for non-interactive use
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
`);

  isReady = true;
  post({ type: 'ready' });
}

/**
 * Parse import statements from Python code and return package names.
 */
function parseImports(code: string): string[] {
  const imports = new Set<string>();
  const lines = code.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    // import foo, import foo.bar, import foo as bar
    const importMatch = trimmed.match(/^import\s+([\w.]+)/);
    if (importMatch) {
      imports.add(importMatch[1].split('.')[0]);
    }
    // from foo import bar, from foo.bar import baz
    const fromMatch = trimmed.match(/^from\s+([\w.]+)\s+import/);
    if (fromMatch) {
      imports.add(fromMatch[1].split('.')[0]);
    }
  }
  return Array.from(imports);
}

/**
 * Install required packages via micropip.
 * Tracks installed packages in-session to skip redundant installs.
 * Browser HTTP cache handles cross-session wheel caching automatically.
 */
const installedPackages = new Set<string>();

async function installPackages(packages: string[]): Promise<void> {
  const toInstall = packages.filter(p => !STDLIB.has(p) && MICROPIP_PACKAGES.has(p) && !installedPackages.has(p));
  if (toInstall.length === 0) return;

  const micropip = pyodide.pyimport('micropip');
  for (const pkg of toInstall) {
    try {
      // Map package names to PyPI names
      const pypiName = pkg === 'sklearn' ? 'scikit-learn' : pkg === 'PIL' ? 'Pillow' : pkg;
      await micropip.install(pypiName);
      installedPackages.add(pkg);
    } catch (e: any) {
      // Non-fatal — code will error on actual import if needed
      console.warn(`[Pyodide] Failed to install ${pkg}:`, e.message);
    }
  }
}

/**
 * Execute Python code and stream output blocks.
 */
async function executeCode(id: string, code: string) {
  if (!isReady || !pyodide) {
    post({ type: 'error', id, message: 'Pyodide is not initialized yet.' });
    return;
  }

  const startTime = performance.now();
  let success = true;

  // Override stdout/stderr for this execution
  pyodide.setStdout({
    batched: (text: string) => {
      post({ type: 'output', id, block: { type: 'stdout', text: text + '\n' } });
    },
  });
  pyodide.setStderr({
    batched: (text: string) => {
      post({ type: 'output', id, block: { type: 'stderr', text: text + '\n' } });
    },
  });

  try {
    // Parse and install required packages
    const imports = parseImports(code);
    const needsInstall = imports.filter(p => !STDLIB.has(p) && MICROPIP_PACKAGES.has(p));
    if (needsInstall.length > 0) {
      post({ type: 'output', id, block: { type: 'stdout', text: `Installing packages: ${needsInstall.join(', ')}...\n` } });
      await installPackages(needsInstall);
    }

    // Wrap code to capture matplotlib figures and last expression value
    const wrappedCode = `
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
${code}

# Capture any unsaved matplotlib figures
try:
    import matplotlib.pyplot as _plt
    if _plt.get_fignums():
        _aura_show()
except (ImportError, NameError):
    pass
`;

    await pyodide.runPythonAsync(wrappedCode);

    // Extract captured images
    const images = pyodide.globals.get('_aura_images');
    if (images) {
      const imgList = images.toJs();
      for (const imgData of imgList) {
        post({ type: 'output', id, block: { type: 'image', mime: 'image/png', data: imgData } });
      }
    }

    // Extract variable state (user-defined only, skip _prefixed internals)
    const varsCode = `
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
`;
    const varsJson = await pyodide.runPythonAsync(varsCode);
    if (varsJson != null && varsJson !== '') {
      try {
        const variables = JSON.parse(varsJson);
        if (Array.isArray(variables)) post({ type: 'variables', id, variables });
      } catch { /* ignore parse errors */ }
    }

    // Check for DataFrame results — detect and send as HTML tables
    const dfCheck = `
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
`;
    const tablesJson = await pyodide.runPythonAsync(dfCheck);
    if (tablesJson != null && tablesJson !== '') {
      try {
        const tables = JSON.parse(tablesJson);
        if (Array.isArray(tables)) {
          for (const tableHtml of tables) {
            post({ type: 'output', id, block: { type: 'html', content: tableHtml } });
          }
        }
      } catch { /* ignore */ }
    }

  } catch (e: any) {
    success = false;
    // Parse Python traceback
    const message = e.message || String(e);
    // Pyodide wraps Python errors — extract the traceback
    const tbMatch = message.match(/Traceback[\s\S]*/);
    const traceback = tbMatch ? tbMatch[0] : message;

    // Try to extract error type and value
    const lastLine = traceback.trim().split('\n').pop() || '';
    const errMatch = lastLine.match(/^(\w+Error|Exception):\s*(.*)/);

    post({
      type: 'output', id,
      block: {
        type: 'error',
        ename: errMatch ? errMatch[1] : 'Error',
        evalue: errMatch ? errMatch[2] : lastLine,
        traceback,
      },
    });
  }

  const executionTime = (performance.now() - startTime) / 1000;
  post({ type: 'done', id, success, executionTime });
}

/**
 * Reset the Python runtime (clear all variables).
 */
async function resetRuntime() {
  if (!pyodide) return;
  // Clear user-defined globals
  await pyodide.runPythonAsync(`
_skip = set(dir()) | {'_skip'}
for _name in list(globals().keys()):
    if not _name.startswith('_') and _name not in _skip:
        try:
            del globals()[_name]
        except:
            pass
del _skip, _name
`);
  post({ type: 'ready' });
}

// Message handler
self.onmessage = async (e: MessageEvent) => {
  const data = e.data;
  if (!data || !data.type) return;

  const { type, id, code, packages } = data;

  try {
    switch (type) {
      case 'init':
        if (!isReady) {
          await initPyodide();
        }
        break;

      case 'execute':
        if (!id || typeof code !== 'string') {
          post({ type: 'error', id: id || '_unknown', message: 'Missing id or code in execute message' });
          return;
        }
        await executeCode(id, code);
        break;

      case 'reset':
        await resetRuntime();
        break;

      case 'install':
        if (packages?.length) {
          await installPackages(packages);
        }
        break;
    }
  } catch (err: any) {
    post({ type: 'error', id: id || '_unknown', message: `Worker error: ${err.message || String(err)}` });
  }
};
