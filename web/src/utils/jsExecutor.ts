import type { OutputBlock } from './pyodideExecutor';

/**
 * Execute JavaScript code in a sandboxed iframe to prevent access to the
 * parent page's DOM, localStorage, cookies, and API keys.
 *
 * The iframe uses `sandbox="allow-scripts"` (no allow-same-origin),
 * which isolates the code from the parent origin entirely.
 */
export function executeJS(
  code: string,
  callbacks: {
    onOutput: (block: OutputBlock) => void;
    onDone: (success: boolean, time: number) => void;
  }
): void {
  const start = performance.now();
  const { onOutput, onDone } = callbacks;

  // Build an HTML page that runs the code and posts results back
  const html = `<!DOCTYPE html><html><head><script>
    const _output = [];
    const console = {
      log: (...a) => parent.postMessage({ t: 'out', d: { type: 'stdout', text: a.map(String).join(' ') } }, '*'),
      warn: (...a) => parent.postMessage({ t: 'out', d: { type: 'stderr', text: '⚠ ' + a.map(String).join(' ') } }, '*'),
      error: (...a) => parent.postMessage({ t: 'out', d: { type: 'stderr', text: '✗ ' + a.map(String).join(' ') } }, '*'),
      info: (...a) => parent.postMessage({ t: 'out', d: { type: 'stdout', text: a.map(String).join(' ') } }, '*'),
      table: (d) => parent.postMessage({ t: 'out', d: { type: 'stdout', text: JSON.stringify(d, null, 2) } }, '*'),
    };
    try {
      const _r = (function() { ${code} })();
      if (_r && typeof _r.then === 'function') {
        _r.then(v => {
          if (v !== undefined) parent.postMessage({ t: 'out', d: { type: 'stdout', text: String(v) } }, '*');
          parent.postMessage({ t: 'done', ok: true }, '*');
        }).catch(e => {
          parent.postMessage({ t: 'out', d: { type: 'error', ename: 'AsyncError', evalue: e.message, traceback: e.stack } }, '*');
          parent.postMessage({ t: 'done', ok: false }, '*');
        });
      } else {
        if (_r !== undefined) parent.postMessage({ t: 'out', d: { type: 'stdout', text: String(_r) } }, '*');
        parent.postMessage({ t: 'done', ok: true }, '*');
      }
    } catch(e) {
      parent.postMessage({ t: 'out', d: { type: 'error', ename: e.name||'Error', evalue: e.message||String(e), traceback: e.stack } }, '*');
      parent.postMessage({ t: 'done', ok: false }, '*');
    }
  <\/script></head><body></body></html>`;

  const iframe = document.createElement('iframe');
  iframe.sandbox.add('allow-scripts'); // No allow-same-origin — fully isolated
  iframe.style.display = 'none';
  iframe.srcdoc = html;

  let settled = false;
  const timeout = setTimeout(() => {
    if (!settled) {
      settled = true;
      onOutput({ type: 'error', ename: 'TimeoutError', evalue: 'Execution timed out (10s)', traceback: undefined });
      onDone(false, performance.now() - start);
      cleanup();
    }
  }, 10000);

  const handler = (e: MessageEvent) => {
    // Only accept messages from our sandbox iframe
    if (e.source !== iframe.contentWindow) return;
    if (e.data?.t === 'out') {
      onOutput(e.data.d);
    } else if (e.data?.t === 'done') {
      if (!settled) {
        settled = true;
        onDone(e.data.ok, performance.now() - start);
        cleanup();
      }
    }
  };

  function cleanup() {
    clearTimeout(timeout);
    window.removeEventListener('message', handler);
    iframe.remove();
  }

  window.addEventListener('message', handler);
  document.body.appendChild(iframe);
}
