/**
 * Build srcdoc HTML for rendering artifacts in an iframe.
 * Ported from extension's ArtifactsPanel buildSrcdoc.
 */

export type ArtifactType = 'html' | 'react' | 'svg' | 'mermaid' | 'chart' | 'markdown';

const PRE_BUNDLED = new Set([
  'react', 'react-dom', 'react-dom/client', 'recharts', 'lucide-react',
  'framer-motion', 'three', 'd3', '@tanstack/react-query', 'zustand',
  'clsx', 'date-fns',
]);

function extractImports(code: string): string[] {
  const imports = new Set<string>();
  const regex = /import\s+(?:[\s\S]*?\s+from\s+)?['"]([^'"./][^'"]*)['"]/g;
  let match;
  while ((match = regex.exec(code)) !== null) {
    let pkg = match[1];
    if (pkg.startsWith('@')) {
      const parts = pkg.split('/');
      pkg = parts.slice(0, 2).join('/');
    } else {
      pkg = pkg.split('/')[0];
    }
    imports.add(pkg);
  }
  return Array.from(imports);
}

function generateDynamicImports(code: string): Record<string, string> {
  const detected = extractImports(code);
  const dynamic: Record<string, string> = {};
  for (const pkg of detected) {
    if (PRE_BUNDLED.has(pkg)) continue;
    dynamic[pkg] = `https://esm.sh/${pkg}?external=react,react-dom`;
  }
  return dynamic;
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function markdownToHtml(md: string): string {
  let html = escHtml(md);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, text, url) => {
    // Sanitize URLs — block javascript: and data: schemes
    const safeUrl = /^(https?:|mailto:|#|\/)/i.test(url) ? url : '#';
    return `<a href="${safeUrl}" target="_blank">${text}</a>`;
  });
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  html = html.replace(/\n\n/g, '</p><p>');
  return `<p>${html}</p>`;
}

const RESIZE_SCRIPT = `<script>
(function(){
  function notifyHeight(){
    try{parent.postMessage({type:'artifact-resize',height:document.documentElement.scrollHeight},'*')}catch(e){}
  }
  window.addEventListener('load',notifyHeight);
  if(window.ResizeObserver){new ResizeObserver(notifyHeight).observe(document.body)}
})();
<\/script>`;

const ERROR_SCRIPT = `<script>
window.onerror = function(msg, src, line, col, err) {
  parent.postMessage({ type: 'artifact-error', msg: String(msg), line: line, col: col, stack: err ? err.stack : '' }, '*');
};
window.addEventListener('unhandledrejection', function(e) {
  parent.postMessage({ type: 'artifact-error', msg: String(e.reason), line: 0 }, '*');
});
['log','warn','error','info'].forEach(function(method) {
  var orig = console[method];
  console[method] = function() {
    var args = Array.prototype.slice.call(arguments);
    orig.apply(console, arguments);
    try {
      parent.postMessage({ type: 'console', level: method, args: args.map(function(a) { try { return JSON.stringify(a); } catch(e) { return String(a); } }), timestamp: Date.now() }, '*');
    } catch(e) {}
  };
});
<\/script>`;

export function buildSrcdoc(type: ArtifactType, code: string): string {
  const hasTailwind = /\bclass(?:Name)?=["'][^"']*(?:flex|grid|p-|m-|text-|bg-|rounded|shadow|border|w-|h-|gap-|items-|justify-)/.test(code);
  const tailwindCdn = hasTailwind ? '<script src="https://cdn.tailwindcss.com"><\/script>' : '';

  if (type === 'html') {
    if (code.includes('</head>')) {
      return code.replace('</head>', tailwindCdn + ERROR_SCRIPT + RESIZE_SCRIPT + '</head>');
    }
    return tailwindCdn + ERROR_SCRIPT + RESIZE_SCRIPT + code;
  }

  if (type === 'svg') {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0a0a0f;overflow:hidden}</style></head><body>${code}</body></html>`;
  }

  if (type === 'markdown') {
    const htmlContent = markdownToHtml(code);
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{font-family:system-ui,-apple-system,sans-serif;padding:24px;line-height:1.7;max-width:720px;margin:0 auto;color:#e8e6f0;background:#0a0a0f}h1,h2,h3{color:#a78bfa;margin:20px 0 8px}a{color:#818cf8}code{background:rgba(167,139,250,0.12);padding:2px 6px;border-radius:4px;font-size:0.9em}pre{background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;overflow-x:auto}blockquote{border-left:3px solid #7c3aed;margin:12px 0;padding:8px 16px;color:#9ca3af}</style></head><body>${htmlContent}</body></html>`;
  }

  if (type === 'react') {
    const hasImports = /\bimport\s+/.test(code);
    if (hasImports) {
      const dynamicImports = generateDynamicImports(code);
      const importMap = {
        imports: {
          'react': 'https://esm.sh/react@19',
          'react-dom': 'https://esm.sh/react-dom@19',
          'react-dom/client': 'https://esm.sh/react-dom@19/client',
          'recharts': 'https://esm.sh/recharts?external=react,react-dom',
          'lucide-react': 'https://esm.sh/lucide-react?external=react',
          'framer-motion': 'https://esm.sh/framer-motion?external=react,react-dom',
          'three': 'https://esm.sh/three',
          'd3': 'https://esm.sh/d3',
          '@tanstack/react-query': 'https://esm.sh/@tanstack/react-query?external=react',
          'zustand': 'https://esm.sh/zustand?external=react',
          'clsx': 'https://esm.sh/clsx',
          'date-fns': 'https://esm.sh/date-fns',
          ...dynamicImports,
        },
      };
      return `<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">${ERROR_SCRIPT}${RESIZE_SCRIPT}${tailwindCdn}<script type="importmap">${JSON.stringify(importMap)}<\/script><style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh}#root{padding:16px;min-height:100vh}</style></head><body><div id="root"></div><script type="module">${code}<\/script></body></html>`;
    }
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><script src="https://unpkg.com/react@18/umd/react.production.min.js"><\/script><script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"><\/script>${tailwindCdn}<style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh}#root{padding:16px;min-height:100vh}</style>${ERROR_SCRIPT}${RESIZE_SCRIPT}</head><body><div id="root"></div><script>try{${code}}catch(e){parent.postMessage({type:'artifact-error',msg:e.message,line:0,stack:e.stack},'*');}<\/script></body></html>`;
  }

  if (type === 'mermaid') {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"><\/script><style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui;background:#0a0a0f;color:#e8e6f0;min-height:100vh}.mermaid{display:flex;justify-content:center;padding:24px}.mermaid svg{max-width:100%}</style>${ERROR_SCRIPT}${RESIZE_SCRIPT}</head><body><pre class="mermaid">${escHtml(code)}</pre><script>mermaid.initialize({startOnLoad:true,theme:'dark',themeVariables:{primaryColor:'#7c3aed',primaryTextColor:'#e8e6f0',primaryBorderColor:'#5b21b6',lineColor:'#6d28d9',secondaryColor:'#1e1b4b',tertiaryColor:'#0f0a2e'}});<\/script></body></html>`;
  }

  if (type === 'chart') {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"><\/script><style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:system-ui;background:#0a0a0f;color:#e8e6f0;min-height:100vh;padding:16px}canvas{max-width:100%;height:auto}</style>${ERROR_SCRIPT}${RESIZE_SCRIPT}</head><body><div id="root"><canvas id="chart"></canvas></div><script>try{${code}}catch(e){parent.postMessage({type:'artifact-error',msg:e.message,line:0,stack:e.stack},'*');}<\/script></body></html>`;
  }

  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><pre>${escHtml(code)}</pre></body></html>`;
}

/**
 * Detect artifact type from code block language and content.
 */
export function detectArtifactType(code: string, language: string): ArtifactType | null {
  const lang = language.toLowerCase();
  if (lang === 'html' && (code.includes('<!DOCTYPE') || code.includes('<html') || code.includes('<body') || code.includes('<div'))) return 'html';
  if (lang === 'jsx' || lang === 'tsx') return 'react';
  if (lang === 'svg' || (code.trim().startsWith('<svg'))) return 'svg';
  if (lang === 'mermaid' || code.trim().startsWith('graph') || code.trim().startsWith('sequenceDiagram') || code.trim().startsWith('flowchart') || code.trim().startsWith('classDiagram')) return 'mermaid';
  if (code.includes('new Chart(') || code.includes('Chart.js')) return 'chart';
  return null;
}
