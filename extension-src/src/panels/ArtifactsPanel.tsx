import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Copy, Download, Maximize2, Minimize2, Code2, Eye, SplitSquareHorizontal,
  Sparkles, Wand2, Wrench, RotateCcw, Globe, BarChart3, GitBranch,
  Gamepad2, Presentation, FileCode, ChevronRight, Radio, X, Terminal, Undo2, Redo2,
  Save, FolderOpen, Search, Pencil, Trash2, GitFork, Package,
} from 'lucide-react';
import { useStore } from '../store';
import CodeEditor, { type CodeEditorDiagnostic, type CodeEditorLanguage } from '../components/CodeEditor';
import FileTree from '../components/FileTree';
import ModelPill from '../components/ModelPill';
import OverlayModal from '../components/OverlayModal';
import { HTTP, getAuthHeaders } from '../api';
import { streamRawGenerate } from '../utils/streamChat';
import { StreamingPreviewController } from '../utils/StreamingPreviewController';
import { useVersionHistory } from '../utils/useVersionHistory';
import { generateDynamicImports } from '../utils/importDetector';
import { clearArtifactsPanelState, loadArtifactsPanelState, saveArtifactsPanelState } from '../utils/artifactPersistence';
import { captureIframeThumbnailDataUrl } from '../utils/captureIframeThumbnail';
import { getDefaultCreationSettings, loadCreationSettings, type CreationSettings } from '../utils/creationSettings';
import { summarizeDiff } from '../utils/diffHeuristics';
import { galleryStore, type SavedArtifact } from '../utils/artifactGallery';

const DiffEditor = React.lazy(() => import('../components/DiffEditor'));

/* ─── Types ─── */
type ArtifactType = 'html' | 'react' | 'svg' | 'mermaid' | 'chart' | 'markdown' | 'css';
type ViewMode = 'preview' | 'code' | 'split';
type PanelMode = 'generate' | 'live';

interface QuickStart {
  label: string;
  icon: React.ReactNode;
  type: ArtifactType;
  template: string;
}

interface LiveFile {
  filename: string;
  code: string;
  type: ArtifactType;
  timestamp: number;
}

interface ConsoleEntry {
  id: string;
  level: 'log' | 'warn' | 'error' | 'info';
  args: string[];
  timestamp: number;
  stack?: string;
}

interface PendingArtifactDiff {
  modified: string;
  original: string;
  prompt: string;
  type: ArtifactType;
}

type ConsoleFilter = 'all' | 'error' | 'warn' | 'info' | 'log';

/* ─── Constants ─── */
const QUICK_STARTS: QuickStart[] = [
  { label: 'Webpage',   icon: <Globe size={14} />,        type: 'html',    template: 'Create a beautiful landing page with a hero section, feature cards, and a footer. Use modern CSS with gradients and smooth animations.' },
  { label: 'Chart',     icon: <BarChart3 size={14} />,    type: 'chart',   template: 'Create an interactive dashboard with 3 charts: a bar chart showing monthly revenue, a line chart showing user growth, and a doughnut chart showing traffic sources. Use vibrant colors.' },
  { label: 'Mind Map',  icon: <GitBranch size={14} />,    type: 'mermaid', template: 'Create a mermaid mindmap diagram about Machine Learning, with branches for Supervised Learning, Unsupervised Learning, Reinforcement Learning, and Deep Learning, each with 3-4 sub-topics.' },
  { label: 'Flowchart', icon: <GitBranch size={14} />,    type: 'mermaid', template: 'Create a mermaid flowchart diagram showing a CI/CD pipeline from code commit to production deployment, including build, test, staging, and approval steps.' },
  { label: 'Game',      icon: <Gamepad2 size={14} />,     type: 'html',    template: 'Create a playable Snake game with keyboard controls, score tracking, and a game-over screen with restart button. Use canvas for rendering. Make it look polished with a dark theme.' },
  { label: 'Slides',    icon: <Presentation size={14} />, type: 'html',    template: 'Create a 5-slide presentation about AI in 2025. Each slide should have a title, bullet points, and smooth left/right navigation with arrow keys and on-screen buttons. Use a sleek dark gradient theme.' },
];

const SYSTEM_PROMPTS: Record<string, string> = {
  html: 'You are an expert web developer. Respond with ONLY a complete, self-contained HTML document (including <!DOCTYPE html>, <html>, <head>, <body>). Include all CSS in a <style> tag and all JS in a <script> tag. No markdown fences, no explanation. Make it visually polished with modern CSS.',
  react: 'You are an expert React developer. Respond with ONLY the JavaScript code for a React component. You may use ES module import statements — React, ReactDOM, recharts, lucide-react, framer-motion, three, d3, @tanstack/react-query, zustand, clsx, date-fns, and any npm package are available via esm.sh. Start with imports like: import React from "react"; import { createRoot } from "react-dom/client"; Then render: createRoot(document.getElementById("root")).render(<App />);. No markdown fences, no explanation. Use inline styles or Tailwind CSS classes (Tailwind is auto-detected and loaded).',
  svg: 'You are an expert SVG artist. Respond with ONLY valid SVG markup. No markdown fences, no explanation.',
  mermaid: 'You are an expert at Mermaid.js diagrams. Respond with ONLY the mermaid diagram definition (e.g., starting with "graph TD", "mindmap", "sequenceDiagram", "flowchart LR", etc.). No markdown fences, no explanation, no HTML wrapping.',
  chart: 'You are an expert data visualization developer using Chart.js. Respond with ONLY JavaScript code that creates Chart.js charts. The canvas elements should be created in JS and appended to document.getElementById("root"). Chart.js is available globally as Chart. No markdown fences, no explanation. Create beautiful charts with good color schemes.',
  markdown: 'You are a technical writer. Respond with ONLY well-formatted Markdown content. No HTML, no fences around the whole thing, no explanation.',
};

const ARTIFACTS_WS_URL = HTTP.replace(/^http/, 'ws') + '/api/artifacts/stream';
const MAX_AUTO_FIX_ATTEMPTS = 3;
const CONSOLE_FILTERS: ConsoleFilter[] = ['all', 'error', 'warn', 'info', 'log'];

function isArtifactType(value: string): value is ArtifactType {
  return ['html', 'react', 'svg', 'mermaid', 'chart', 'markdown', 'css'].includes(value);
}

function getArtifactEditorLanguage(type: ArtifactType): CodeEditorLanguage {
  switch (type) {
    case 'react':
      return 'jsx';
    case 'chart':
      return 'javascript';
    case 'markdown':
    case 'mermaid':
      return 'markdown';
    case 'css':
      return 'css';
    case 'svg':
      return 'svg';
    case 'html':
    default:
      return 'html';
  }
}

function buildRuntimeDiagnostics(message: string, line?: number): CodeEditorDiagnostic[] {
  if (!line || line < 1) return [];
  return [{ line, message, severity: 'error' }];
}

function parseConsoleArg(value: string): string {
  try {
    const parsed = JSON.parse(value);
    if (typeof parsed === 'string') return parsed;
    return JSON.stringify(parsed);
  } catch {
    return value;
  }
}

function buildArtifactRepairPrompt(type: ArtifactType, currentCode: string, errorMessage: string, line?: number): string {
  return `The ${type.toUpperCase()} code you generated has a runtime error in the preview iframe.

ERROR: ${errorMessage}
LINE: ${line || 'unknown'}

Current code:
\`\`\`${type}
${currentCode}
\`\`\`

Fix the error and return the complete corrected code only. Preserve the existing design and intended functionality.`;
}

function buildArtifactName(prompt: string, type: ArtifactType): string {
  const trimmed = prompt.trim();
  if (!trimmed) return `${type.toUpperCase()} Artifact`;
  const compact = trimmed.replace(/\s+/g, ' ').slice(0, 36).trim();
  return compact || `${type.toUpperCase()} Artifact`;
}

function getArtifactTypePalette(type: ArtifactType): { start: string; end: string; accent: string } {
  switch (type) {
    case 'react':
      return { start: '#0f172a', end: '#0b3b5f', accent: '#61dafb' };
    case 'svg':
      return { start: '#1f2937', end: '#4c1d95', accent: '#c084fc' };
    case 'mermaid':
      return { start: '#111827', end: '#1d4ed8', accent: '#93c5fd' };
    case 'chart':
      return { start: '#172554', end: '#14532d', accent: '#34d399' };
    case 'markdown':
      return { start: '#292524', end: '#0f172a', accent: '#f5f5f4' };
    case 'css':
      return { start: '#1e1b4b', end: '#3730a3', accent: '#a5b4fc' };
    default:
      return { start: '#2e1065', end: '#1d4ed8', accent: '#e9d5ff' };
  }
}

function toDataUrl(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function buildArtifactThumbnailDataUrl(name: string, type: ArtifactType, prompt: string, code: string): string {
  const palette = getArtifactTypePalette(type);
  const title = name.slice(0, 26);
  const summary = (prompt.trim() || code.replace(/\s+/g, ' ').trim()).slice(0, 56);
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320" viewBox="0 0 640 320">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="${palette.start}"/>
      <stop offset="100%" stop-color="${palette.end}"/>
    </linearGradient>
  </defs>
  <rect width="640" height="320" rx="28" fill="url(#bg)"/>
  <circle cx="582" cy="58" r="48" fill="${palette.accent}" fill-opacity="0.16"/>
  <circle cx="78" cy="270" r="62" fill="${palette.accent}" fill-opacity="0.10"/>
  <rect x="34" y="34" width="112" height="28" rx="14" fill="rgba(15,23,42,0.35)"/>
  <text x="90" y="53" text-anchor="middle" fill="${palette.accent}" font-family="Inter, Arial, sans-serif" font-size="14" font-weight="700" letter-spacing="1.4">${type.toUpperCase()}</text>
  <text x="34" y="116" fill="#F8FAFC" font-family="Inter, Arial, sans-serif" font-size="30" font-weight="700">${escapeXml(title)}</text>
  <text x="34" y="154" fill="rgba(248,250,252,0.85)" font-family="Inter, Arial, sans-serif" font-size="16">${escapeXml(summary)}</text>
  <rect x="34" y="198" width="572" height="82" rx="18" fill="rgba(15,23,42,0.22)" stroke="rgba(255,255,255,0.08)"/>
  <text x="54" y="226" fill="rgba(248,250,252,0.68)" font-family="'JetBrains Mono', Consolas, monospace" font-size="13">${escapeXml(code.replace(/\s+/g, ' ').trim().slice(0, 72) || '<empty>')}</text>
  <text x="54" y="252" fill="rgba(248,250,252,0.52)" font-family="'JetBrains Mono', Consolas, monospace" font-size="13">${escapeXml(code.replace(/\s+/g, ' ').trim().slice(72, 144))}</text>
</svg>`;
  return toDataUrl(svg);
}

function escapeXml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function buildDirectoriesFromPaths(paths: string[]): string[] {
  const directories = new Set<string>();
  for (const path of paths) {
    const parts = path.replace(/\\/g, '/').split('/').filter(Boolean);
    parts.pop();
    let current = '';
    for (const part of parts) {
      current = current ? `${current}/${part}` : part;
      directories.add(current);
    }
  }
  return Array.from(directories).sort((a, b) => a.localeCompare(b));
}

/** Build the full srcdoc HTML for the iframe given the artifact type and user code. */
function buildSrcdoc(type: ArtifactType, code: string, includeErrorHandler = true): string {
  const errorScript = includeErrorHandler
    ? `<script>
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
      parent.postMessage({
        type: 'console',
        level: method,
        args: args.map(function(a) { try { return JSON.stringify(a); } catch(e) { return String(a); } }),
        timestamp: Date.now()
      }, '*');
    } catch(e) {}
  };
});
</script>`
    : '';

  // Auto-detect Tailwind usage for html and react types
  const hasTailwind = /\bclass(?:Name)?=["'][^"']*(?:flex|grid|p-|m-|text-|bg-|rounded|shadow|border|w-|h-|gap-|items-|justify-)/.test(code);
  const tailwindCdn = hasTailwind ? '<script src="https://cdn.tailwindcss.com"><\/script>' : '';

  if (type === 'html') {
    if (code.includes('</head>')) {
      return code.replace('</head>', tailwindCdn + errorScript + '</head>');
    }
    return tailwindCdn + errorScript + code;
  }

  if (type === 'svg') {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#0a0a0f;overflow:hidden}</style></head><body>${code}</body></html>`;
  }

  if (type === 'css') {
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>${code}</style><style>body{font-family:system-ui,-apple-system,sans-serif;padding:24px;background:#0a0a0f;color:#e8e6f0;min-height:100vh}</style>${errorScript}</head><body><div class="preview"><h1>CSS Preview</h1><p>Paragraph text for styling preview.</p><button>Button</button><a href="#">Link</a><ul><li>List item 1</li><li>List item 2</li></ul></div></body></html>`;
  }

  if (type === 'markdown') {
    const htmlContent = markdownToHtml(code);
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:system-ui,-apple-system,sans-serif;padding:24px;line-height:1.7;max-width:720px;margin:0 auto;color:#e8e6f0;background:#0a0a0f}h1,h2,h3{color:#a78bfa;margin:20px 0 8px}h1{font-size:1.8em;border-bottom:1px solid rgba(167,139,250,0.2);padding-bottom:8px}h2{font-size:1.4em}h3{font-size:1.15em}a{color:#818cf8}code{background:rgba(167,139,250,0.12);padding:2px 6px;border-radius:4px;font-size:0.9em}pre{background:rgba(0,0,0,0.4);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:16px;overflow-x:auto}pre code{background:none;padding:0}blockquote{border-left:3px solid #7c3aed;margin:12px 0;padding:8px 16px;color:#9ca3af;background:rgba(124,58,237,0.06);border-radius:0 6px 6px 0}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid rgba(255,255,255,0.1);padding:8px 12px;text-align:left}th{background:rgba(124,58,237,0.1)}</style></head><body>${htmlContent}</body></html>`;
  }

  // For react with ES module imports — use import map + esm.sh
  if (type === 'react') {
    const hasImports = /\bimport\s+/.test(code);

    if (hasImports) {
      // Module mode: build dynamic import map and use <script type="module">
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

      return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
${errorScript}
${tailwindCdn}
<script type="importmap">${JSON.stringify(importMap)}<\/script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh}
#root{padding:16px;min-height:100vh}
</style>
</head>
<body>
<div id="root"></div>
<script type="module">
${code}
<\/script>
</body>
</html>`;
    }

    // UMD fallback: legacy non-import React code (React/ReactDOM available globally)
    return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/react@18/umd/react.production.min.js"><\/script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"><\/script>
${tailwindCdn}
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh}
#root{padding:16px;min-height:100vh}
</style>
${errorScript}
</head>
<body>
<div id="root"></div>
<script>
try {
${code}
} catch(e) {
  parent.postMessage({ type: 'artifact-error', msg: e.message, line: 0, stack: e.stack }, '*');
}
<\/script>
</body>
</html>`;
  }

  // For chart, mermaid — use the full runtime template
  const cdnScripts: string[] = [];
  let bodyContent = '<div id="root"></div>';
  let userScript = code;

  if (type === 'chart') {
    cdnScripts.push(
      '<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"><\/script>',
    );
  } else if (type === 'mermaid') {
    cdnScripts.push(
      '<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"><\/script>',
    );
    bodyContent = `<pre class="mermaid">\n${escHtml(code)}\n</pre>`;
    userScript = `mermaid.initialize({ startOnLoad: true, theme: 'dark', themeVariables: { primaryColor: '#7c3aed', primaryTextColor: '#e8e6f0', primaryBorderColor: '#5b21b6', lineColor: '#6d28d9', secondaryColor: '#1e1b4b', tertiaryColor: '#0f0a2e' } });`;
  }

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
${cdnScripts.join('\n')}
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh}
#root{padding:16px;min-height:100vh}
canvas{max-width:100%;height:auto}
.mermaid{display:flex;justify-content:center;padding:24px}
.mermaid svg{max-width:100%}
</style>
${errorScript}
</head>
<body>
${bodyContent}
<script>
try {
${userScript}
} catch(e) {
  parent.postMessage({ type: 'artifact-error', msg: e.message, line: 0, stack: e.stack }, '*');
}
<\/script>
</body>
</html>`;
}

/** Minimal markdown to HTML (no external dep for iframe content) */
function markdownToHtml(md: string): string {
  let html = escHtml(md);
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/((<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');
  html = html.replace(/^---$/gm, '<hr>');
  html = html.replace(/\n\n/g, '</p><p>');
  html = '<p>' + html + '</p>';
  html = html.replace(/<p>\s*<(h[1-3]|ul|ol|blockquote|pre|hr)/g, '<$1');
  html = html.replace(/<\/(h[1-3]|ul|ol|blockquote|pre)>\s*<\/p>/g, '</$1>');
  return html;
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function stripFences(s: string): string {
  return s.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
}

function timeAgo(ts: number): string {
  const diff = Math.floor((Date.now() / 1000) - ts);
  if (diff < 5) return 'just now';
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

/* ─── Shared button style ─── */
const btnBase: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 5,
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 'var(--r-md)',
  color: 'var(--mu)',
  padding: '6px 10px',
  cursor: 'pointer',
  fontSize: '11.5px',
  fontFamily: 'inherit',
  transition: 'all 0.15s ease',
  whiteSpace: 'nowrap',
};

const btnHover: React.CSSProperties = {
  background: 'var(--pg)',
  borderColor: 'rgba(124,58,237,0.2)',
  color: 'var(--pl)',
};

/* ─────────────────────────────────────────────────────────────────────────
   useArtifactsWS — dedicated hook for the live-preview WebSocket
   ───────────────────────────────────────────────────────────────────────── */
function useArtifactsWS(enabled: boolean) {
  const [files, setFiles] = useState<Record<string, LiveFile>>({});
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  const connect = useCallback(() => {
    if (!enabledRef.current) return;
    if (wsRef.current && wsRef.current.readyState <= WebSocket.OPEN) return;

    const socket = new WebSocket(ARTIFACTS_WS_URL);

    socket.onopen = () => {
      setConnected(true);
    };

    socket.onmessage = (ev) => {
      let d: any;
      try { d = JSON.parse(ev.data); } catch { return; }

      if (d.type === 'snapshot' && d.previews) {
        // Hydrate from snapshot on connect
        const hydrated: Record<string, LiveFile> = {};
        for (const [key, val] of Object.entries(d.previews as Record<string, any>)) {
          hydrated[key] = {
            filename: val.filename,
            code: val.code,
            type: val.type as ArtifactType,
            timestamp: val.timestamp,
          };
        }
        setFiles(hydrated);
      } else if (d.type === 'artifact_update') {
        setFiles(prev => ({
          ...prev,
          [d.filename]: {
            filename: d.filename,
            code: d.code,
            type: (d.artifact_type || 'html') as ArtifactType,
            timestamp: d.timestamp,
          },
        }));
      } else if (d.type === 'pong') {
        // keepalive response — nothing to do
      }
    };

    socket.onclose = () => {
      setConnected(false);
      wsRef.current = null;
      // Reconnect with backoff
      if (enabledRef.current) {
        retryRef.current = setTimeout(connect, 3000);
      }
    };

    socket.onerror = () => {
      setConnected(false);
    };

    wsRef.current = socket;
  }, []);

  // Connect/disconnect based on enabled flag
  useEffect(() => {
    if (enabled) {
      connect();
    } else {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
    }
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [enabled, connect]);

  // Keepalive ping every 30s
  useEffect(() => {
    if (!enabled || !connected) return;
    const iv = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30_000);
    return () => clearInterval(iv);
  }, [enabled, connected]);

  const clearFile = useCallback((filename: string) => {
    setFiles(prev => {
      const next = { ...prev };
      delete next[filename];
      return next;
    });
  }, []);

  const clearAll = useCallback(() => setFiles({}), []);

  return { files, connected, clearFile, clearAll };
}


/* ─── Action button helper (outside component to avoid re-creation on each render) ─── */
function ActionBtn({ id, icon, label, onClick, accent, hoveredBtn, setHoveredBtn }: {
  id: string; icon: React.ReactNode; label: string; onClick: () => void; accent?: boolean;
  hoveredBtn: string | null; setHoveredBtn: (v: string | null) => void;
}) {
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHoveredBtn(id)}
      onMouseLeave={() => setHoveredBtn(null)}
      style={{
        ...btnBase,
        ...(hoveredBtn === id ? btnHover : {}),
        ...(accent ? { background: 'var(--pg)', borderColor: 'rgba(124,58,237,0.2)', color: 'var(--pl)' } : {}),
      }}
    >
      {icon} {label}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Component
   ═══════════════════════════════════════════════════════════════════════════ */
export default function ArtifactsPanel() {
  const { getModel } = useStore();
  const [creationSettings, setCreationSettings] = useState<CreationSettings>(getDefaultCreationSettings());

  // Panel mode: "generate" (manual) or "live" (agent loop)
  const [panelMode, setPanelMode] = useState<PanelMode>('generate');

  // --- Generate mode state (unchanged from original) ---
  const [artifactType, setArtifactType] = useState<ArtifactType>('html');
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [code, setCode] = useState('');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [iframeError, setIframeError] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<ConsoleEntry[]>([]);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [consoleFilter, setConsoleFilter] = useState<ConsoleFilter>('all');
  const [expandedConsoleIds, setExpandedConsoleIds] = useState<string[]>([]);
  const [unreadErrorCount, setUnreadErrorCount] = useState(0);
  const [editorDiagnostics, setEditorDiagnostics] = useState<CodeEditorDiagnostic[]>([]);
  const [autoFixAttempts, setAutoFixAttempts] = useState(0);
  const [isAutoFixing, setIsAutoFixing] = useState(false);
  const [galleryItems, setGalleryItems] = useState<SavedArtifact[]>([]);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [renameDialogArtifact, setRenameDialogArtifact] = useState<SavedArtifact | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteDialogArtifact, setDeleteDialogArtifact] = useState<SavedArtifact | null>(null);
  const [galleryQuery, setGalleryQuery] = useState('');
  const [galleryTypeFilter, setGalleryTypeFilter] = useState<'all' | ArtifactType>('all');
  const [pendingDiff, setPendingDiff] = useState<PendingArtifactDiff | null>(null);
  const { versions, currentIdx, pushVersion, goToVersion, undo, redo, canUndo, canRedo, clear: clearVersions } = useVersionHistory(20, 'aura_artifacts_versions');

  // --- Live mode state ---
  const [liveActiveFile, setLiveActiveFile] = useState<string | null>(null);
  const [liveViewMode, setLiveViewMode] = useState<ViewMode>('preview');
  const { files: liveFiles, connected: liveConnected, clearFile: liveClearFile, clearAll: liveClearAll } = useArtifactsWS(panelMode === 'live');

  // Auto-select latest file in live mode
  const liveFileNames = Object.keys(liveFiles);
  const liveDirectories = useMemo(() => buildDirectoriesFromPaths(liveFileNames), [liveFileNames]);
  const activeLiveFile = (liveActiveFile && liveFiles[liveActiveFile]) ? liveFiles[liveActiveFile] : (liveFileNames.length > 0 ? liveFiles[liveFileNames[liveFileNames.length - 1]] : null);

  // Auto-switch to newest file when a new one arrives
  const prevFileCountRef = useRef(0);
  useEffect(() => {
    if (liveFileNames.length > prevFileCountRef.current && liveFileNames.length > 0) {
      // A new file arrived — switch to it
      setLiveActiveFile(liveFileNames[liveFileNames.length - 1]);
    }
    prevFileCountRef.current = liveFileNames.length;
  }, [liveFileNames.length]);

  // Time-ago ticker for live mode
  const [, setTick] = useState(0);
  useEffect(() => {
    if (panelMode !== 'live') return;
    const iv = setInterval(() => setTick(t => t + 1), 5000);
    return () => clearInterval(iv);
  }, [panelMode]);

  // Refs
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const liveIframeRef = useRef<HTMLIFrameElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const codeRef = useRef('');
  const abortRef = useRef<AbortController | null>(null);
  const loadingRef = useRef(false);
  const autoFixAttemptsRef = useRef(0);
  const autoFixingRef = useRef(false);
  const lastAutoFixSignatureRef = useRef<string | null>(null);
  const restoredStateRef = useRef(false);
  const persistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualPreviewTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    loadingRef.current = loading;
  }, [loading]);

  useEffect(() => {
    loadCreationSettings().then(setCreationSettings).catch(() => {});
  }, []);

  const setTimedStatus = useCallback((nextStatus: string, duration = 2200) => {
    setStatus(nextStatus);
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    if (!nextStatus) return;
    statusTimerRef.current = setTimeout(() => setStatus(''), duration);
  }, []);

  const refreshGallery = useCallback(async () => {
    const items = await galleryStore.list();
    const missingThumbnails = items.filter((item) => !item.thumbnail);
    if (missingThumbnails.length > 0) {
      await Promise.all(
        missingThumbnails.map((item) => galleryStore.update(item.id, {
          thumbnail: buildArtifactThumbnailDataUrl(
            item.name,
            item.type as ArtifactType,
            item.prompt || '',
            item.code,
          ),
        })),
      );
      setGalleryItems(await galleryStore.list());
      return;
    }
    setGalleryItems(items);
  }, []);

  useEffect(() => {
    void refreshGallery();
  }, [refreshGallery]);

  /* ─── Abort fetch on unmount ─── */
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);
    };
  }, []);

  /* ─── Restore persisted generate state ─── */
  useEffect(() => {
    if (restoredStateRef.current) return;
    restoredStateRef.current = true;

    // Check for panel handoff first (e.g., from CodePanel)
    const { consumePanelHandoff } = useStore.getState();
    const handoff = consumePanelHandoff();
    if (handoff?.code) {
      const hType = isArtifactType(handoff.type) ? handoff.type : 'html';
      setArtifactType(hType);
      setCode(handoff.code);
      codeRef.current = handoff.code;
      setPanelMode('generate');
      if (iframeRef.current) {
        iframeRef.current.srcdoc = buildSrcdoc(hType, handoff.code, true);
      }
      return;
    }

    loadArtifactsPanelState().then((saved) => {
      if (!saved) return;
      const nextType = isArtifactType(saved.type) ? saved.type : 'html';
      setArtifactType(nextType);
      setPrompt(saved.prompt);
      setCode(saved.code);
      codeRef.current = saved.code;
      if (saved.activeFile) setLiveActiveFile(saved.activeFile);
      if (saved.code && iframeRef.current) {
        iframeRef.current.srcdoc = buildSrcdoc(nextType, saved.code, true);
      }
    }).catch(() => {});
  }, []);

  /* ─── Persist generate state ─── */
  useEffect(() => {
    if (panelMode !== 'generate') return;
    if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    persistTimerRef.current = setTimeout(() => {
      void saveArtifactsPanelState({
        code,
        prompt,
        type: artifactType,
        timestamp: Date.now(),
        activeFile: liveActiveFile || undefined,
      });
    }, 1000);

    return () => {
      if (persistTimerRef.current) clearTimeout(persistTimerRef.current);
    };
  }, [artifactType, code, liveActiveFile, panelMode, prompt]);

  useEffect(() => {
    if (!consoleOpen) return;
    setUnreadErrorCount(0);
  }, [consoleOpen]);

  const attemptArtifactAutoFix = useCallback(async (errorMessage: string, line?: number) => {
    if (autoFixingRef.current || !codeRef.current.trim()) return;
    if (autoFixAttemptsRef.current >= MAX_AUTO_FIX_ATTEMPTS) return;

    const attempt = autoFixAttemptsRef.current + 1;
    const type = artifactType;
    const systemPrompt = SYSTEM_PROMPTS[type] || SYSTEM_PROMPTS.html;
    const model = getModel('artifacts') || undefined;

    setIsAutoFixing(true);
    autoFixingRef.current = true;
    setStatus(`Auto-fixing error (${attempt}/${MAX_AUTO_FIX_ATTEMPTS})...`);

    try {
      let repairedCode = '';
      for await (const chunk of streamRawGenerate(
        buildArtifactRepairPrompt(type, codeRef.current, errorMessage, line),
        { systemPrompt, model },
      )) {
        repairedCode += chunk;
      }

      const finalCode = stripFences(repairedCode);
      if (!finalCode.trim()) {
        throw new Error('Auto-fix returned empty code');
      }

      autoFixAttemptsRef.current = attempt;
      setAutoFixAttempts(attempt);
      setIframeError(null);
      setEditorDiagnostics([]);
      setCode(finalCode);
      codeRef.current = finalCode;
      pushVersion(`Auto-fix attempt ${attempt}: ${errorMessage}`, finalCode, `Fix ${attempt}`);
      if (iframeRef.current) {
        iframeRef.current.srcdoc = buildSrcdoc(type, finalCode, true);
      }
      setTimedStatus(`Auto-fix applied (${attempt}/${MAX_AUTO_FIX_ATTEMPTS})`);
    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        setTimedStatus(err?.message || 'Auto-fix failed', 3000);
      }
    } finally {
      setIsAutoFixing(false);
      autoFixingRef.current = false;
    }
  }, [artifactType, getModel, pushVersion, setTimedStatus]);

  /* ─── iframe error listener ─── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (!e.data || typeof e.data !== 'object') return;
      const targetIframe = panelMode === 'live' ? liveIframeRef.current : iframeRef.current;
      if (e.source !== targetIframe?.contentWindow) return;

      if (e.data?.type === 'artifact-error') {
        const msg = e.data.msg || 'Unknown error';
        const lineNumber = typeof e.data.line === 'number' ? e.data.line : 0;
        const line = lineNumber ? ` (line ${lineNumber})` : '';
        const entry: ConsoleEntry = {
          id: crypto.randomUUID(),
          level: 'error',
          args: [`${msg}${line}`],
          timestamp: e.data.timestamp || Date.now(),
          stack: typeof e.data.stack === 'string' ? e.data.stack : undefined,
        };

        setIframeError(`${msg}${line}`);
        setEditorDiagnostics(buildRuntimeDiagnostics(msg, lineNumber));
        setConsoleLogs((prev) => [...prev.slice(-99), entry]);
        if (!consoleOpen) setUnreadErrorCount((prev) => prev + 1);
        if (creationSettings.autoOpenConsoleOnError) setConsoleOpen(true);

        const shouldAutoFix =
          panelMode === 'generate' &&
          creationSettings.autoFixErrors &&
          !loadingRef.current &&
          !!codeRef.current.trim();

        if (shouldAutoFix) {
          const signature = `${msg}|${lineNumber}|${codeRef.current}`;
          if (lastAutoFixSignatureRef.current !== signature) {
            lastAutoFixSignatureRef.current = signature;
            void attemptArtifactAutoFix(msg, lineNumber);
          }
        }
      }

      if (e.data?.type === 'console') {
        const level = (['log', 'warn', 'error', 'info'].includes(e.data.level) ? e.data.level : 'log') as ConsoleEntry['level'];
        const entry: ConsoleEntry = {
          id: crypto.randomUUID(),
          level,
          args: Array.isArray(e.data.args) ? e.data.args : [String(e.data.args ?? '')],
          timestamp: e.data.timestamp || Date.now(),
        };
        setConsoleLogs((prev) => [...prev.slice(-99), entry]);
        if (level === 'error' && !consoleOpen) setUnreadErrorCount((prev) => prev + 1);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [attemptArtifactAutoFix, consoleOpen, creationSettings.autoFixErrors, creationSettings.autoOpenConsoleOnError, panelMode]);

  /* ─── Update live preview when active file changes ─── */
  useEffect(() => {
    if (panelMode !== 'live' || !activeLiveFile || !liveIframeRef.current) return;
    setIframeError(null);
    setEditorDiagnostics([]);
    liveIframeRef.current.srcdoc = buildSrcdoc(activeLiveFile.type, activeLiveFile.code);
  }, [panelMode, activeLiveFile?.code, activeLiveFile?.type, activeLiveFile?.filename]);

  /* ─── Generate artifact ─── */
  const generate = useCallback(async (overridePrompt?: string, overrideType?: ArtifactType) => {
    const text = (overridePrompt ?? prompt).trim();
    if (!text) return;

    const type = overrideType ?? artifactType;
    const previousCode = codeRef.current;
    const shouldReviewDiff = creationSettings.showDiffBeforeApply && !!previousCode.trim();
    setArtifactType(type);
    setAutoFixAttempts(0);
    autoFixAttemptsRef.current = 0;
    setIsAutoFixing(false);
    autoFixingRef.current = false;
    lastAutoFixSignatureRef.current = null;
    setPendingDiff(null);
    setLoading(true);
    setStatus('Generating...');
    setIframeError(null);
    setEditorDiagnostics([]);
    setConsoleLogs([]);
    setExpandedConsoleIds([]);
    setConsoleFilter('all');
    setUnreadErrorCount(0);
    if (!shouldReviewDiff) {
      setCode('');
      codeRef.current = '';
      if (iframeRef.current) iframeRef.current.srcdoc = '';
    }

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const systemPrompt = SYSTEM_PROMPTS[type] || SYSTEM_PROMPTS.html;
    const model = getModel('artifacts') || undefined;

    const previewCtrl = shouldReviewDiff
      ? null
      : new StreamingPreviewController((html) => {
          if (iframeRef.current) {
            iframeRef.current.srcdoc = buildSrcdoc(type, html, true);
          }
        });

    try {
      let streamedCode = '';
      let lastCodeUpdate = 0;
      for await (const chunk of streamRawGenerate(text, {
        systemPrompt,
        model,
        signal: ctrl.signal,
      })) {
        streamedCode += chunk;
        // Throttle code display to every 200ms to avoid render thrashing
        const now = Date.now();
        if (!shouldReviewDiff && now - lastCodeUpdate > 200) {
          setCode(streamedCode);
          lastCodeUpdate = now;
        }
        // Update preview via debounced controller
        previewCtrl?.append(chunk);
      }
      // Ensure final code is shown
      if (!shouldReviewDiff) {
        setCode(streamedCode);
      }

      // Cancel any pending timer without rendering to avoid fenced content flash
      previewCtrl?.reset();
      const finalCode = stripFences(streamedCode);
      const diffMetrics = shouldReviewDiff ? summarizeDiff(previousCode, finalCode) : null;
      const shouldAutoAcceptDiff =
        !!diffMetrics &&
        diffMetrics.changedLineCount > 0 &&
        diffMetrics.changedLineCount <= creationSettings.autoAcceptDiffLineThreshold;
      const shouldReviewProposal =
        !!diffMetrics &&
        diffMetrics.changedLineCount > 0 &&
        !shouldAutoAcceptDiff;

      if (shouldReviewProposal && finalCode.trim() && finalCode !== previousCode) {
        setPendingDiff({
          original: previousCode,
          modified: finalCode,
          prompt: text,
          type,
        });
        if (diffMetrics.changeRatio >= creationSettings.forceDiffReviewChangePercent / 100) {
          setStatus('Large update ready for review');
        } else {
          setStatus('Review changes');
        }
        setViewMode('code');
      } else {
        codeRef.current = finalCode;
        setCode(finalCode);
        pushVersion(text, finalCode);
        if (iframeRef.current) {
          iframeRef.current.srcdoc = buildSrcdoc(type, finalCode, true);
        }
        setStatus(shouldAutoAcceptDiff ? 'Applied small update automatically' : '');
        setViewMode('preview');
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
      previewCtrl?.dispose();
    }
  }, [
    prompt,
    artifactType,
    creationSettings.autoAcceptDiffLineThreshold,
    creationSettings.forceDiffReviewChangePercent,
    creationSettings.showDiffBeforeApply,
    getModel,
    pushVersion,
  ]);

  /* ─── Actions ─── */
  const activeCode = panelMode === 'live' ? (activeLiveFile?.code || '') : code;
  const activeType = panelMode === 'live' ? (activeLiveFile?.type || 'html') : artifactType;

  const copyCode = useCallback(() => {
    if (!activeCode) return;
    navigator.clipboard.writeText(activeCode).then(() => {
      setTimedStatus('Copied!', 1500);
    });
  }, [activeCode, setTimedStatus]);

  const downloadFile = useCallback(() => {
    if (!activeCode) return;
    const exts: Record<string, string> = {
      html: '.html', react: '.html', svg: '.svg',
      mermaid: '.html', chart: '.html', markdown: '.md', css: '.css',
    };
    const isRaw = activeType === 'html' || activeType === 'svg' || activeType === 'markdown' || activeType === 'css';
    const content = isRaw ? activeCode : buildSrcdoc(activeType as ArtifactType, activeCode, false);
    const ext = exts[activeType] || '.html';
    const blob = new Blob([content], { type: 'text/html' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (panelMode === 'live' && activeLiveFile) ? activeLiveFile.filename : `artifact${ext}`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, [activeCode, activeType, panelMode, activeLiveFile]);

  const fixError = useCallback(() => {
    if (!iframeError || !code) return;
    const fixPrompt = `The following code has an error: "${iframeError}"\n\nFix this code and return the corrected version:\n\n${code}`;
    setPrompt(fixPrompt);
    generate(fixPrompt, artifactType);
  }, [iframeError, code, artifactType, generate]);

  const remix = useCallback(() => {
    if (!code) return;
    const remixPrompt = `Take the following ${artifactType} code and create a significantly different variation of it. Change the visual style, colors, layout, and add creative enhancements while keeping the same core functionality:\n\n${code}`;
    setPrompt(remixPrompt);
    generate(remixPrompt, artifactType);
  }, [code, artifactType, generate]);

  const handleQuickStart = useCallback((qs: QuickStart) => {
    setPrompt(qs.template);
    generate(qs.template, qs.type);
  }, [generate]);

  const clearGeneratedArtifact = useCallback(() => {
    setCode('');
    setPrompt('');
    setStatus('');
    setIframeError(null);
    setEditorDiagnostics([]);
    setConsoleLogs([]);
    setConsoleOpen(false);
    setConsoleFilter('all');
    setExpandedConsoleIds([]);
    setUnreadErrorCount(0);
    setAutoFixAttempts(0);
    setIsAutoFixing(false);
    setPendingDiff(null);
    autoFixAttemptsRef.current = 0;
    autoFixingRef.current = false;
    lastAutoFixSignatureRef.current = null;
    codeRef.current = '';
    setViewMode('preview');
    clearVersions();
    void clearArtifactsPanelState();
    if (iframeRef.current) iframeRef.current.srcdoc = '';
  }, [clearVersions]);

  const toggleConsoleEntry = useCallback((id: string) => {
    setExpandedConsoleIds((prev) => (
      prev.includes(id) ? prev.filter((entryId) => entryId !== id) : [...prev, id]
    ));
  }, []);

  const clearConsole = useCallback(() => {
    setConsoleLogs([]);
    setExpandedConsoleIds([]);
    setUnreadErrorCount(0);
  }, []);

  const openSaveDialog = useCallback(() => {
    if (panelMode !== 'generate' || !code.trim()) return;
    setSaveName(buildArtifactName(prompt, artifactType));
    setSaveDialogOpen(true);
  }, [artifactType, code, panelMode, prompt]);

  const saveCurrentArtifact = useCallback(async () => {
    if (!code.trim()) return;
    const nextName = saveName.trim() || buildArtifactName(prompt, artifactType);
    const thumbnail = await captureIframeThumbnailDataUrl({
      iframe: iframeRef.current,
      fallback: { name: nextName, prompt, code, type: artifactType },
      buildFallback: ({ name, prompt: fallbackPrompt, code: fallbackCode, type }) =>
        buildArtifactThumbnailDataUrl(name, type, fallbackPrompt, fallbackCode),
    });
    await galleryStore.save({
      name: nextName,
      type: artifactType,
      code,
      prompt,
      thumbnail,
      tags: prompt.trim() ? prompt.trim().toLowerCase().split(/\s+/).filter(Boolean).slice(0, 6) : [],
    });
    setSaveDialogOpen(false);
    await refreshGallery();
    setTimedStatus(`Saved "${nextName}"`, 1800);
  }, [artifactType, code, prompt, refreshGallery, saveName, setTimedStatus]);

  const loadSavedArtifact = useCallback((artifact: SavedArtifact) => {
    setPanelMode('generate');
    setArtifactType(artifact.type as ArtifactType);
    setCode(artifact.code);
    codeRef.current = artifact.code;
    setPrompt(artifact.prompt || '');
    setViewMode('preview');
    setIframeError(null);
    setEditorDiagnostics([]);
    setConsoleLogs([]);
    setExpandedConsoleIds([]);
    setConsoleFilter('all');
    setUnreadErrorCount(0);
    setAutoFixAttempts(0);
    setIsAutoFixing(false);
    setPendingDiff(null);
    autoFixAttemptsRef.current = 0;
    autoFixingRef.current = false;
    lastAutoFixSignatureRef.current = null;
    if (iframeRef.current) {
      iframeRef.current.srcdoc = buildSrcdoc(artifact.type as ArtifactType, artifact.code, true);
    }
    setGalleryOpen(false);
    setTimedStatus(`Loaded "${artifact.name}"`, 1800);
  }, [setTimedStatus]);

  const forkSavedArtifact = useCallback(async (artifact: SavedArtifact) => {
    const forkName = `${artifact.name} Copy`;
    const id = await galleryStore.save({
      name: forkName,
      type: artifact.type,
      code: artifact.code,
      prompt: artifact.prompt,
      tags: artifact.tags,
      thumbnail: artifact.thumbnail || buildArtifactThumbnailDataUrl(forkName, artifact.type as ArtifactType, artifact.prompt || '', artifact.code),
    });
    await refreshGallery();
    const next = await galleryStore.get(id);
    if (next) loadSavedArtifact(next);
    setTimedStatus(`Forked "${artifact.name}"`, 1800);
  }, [loadSavedArtifact, refreshGallery, setTimedStatus]);

  const openRenameDialog = useCallback((artifact: SavedArtifact) => {
    setRenameDialogArtifact(artifact);
    setRenameValue(artifact.name);
  }, []);

  const submitRenameArtifact = useCallback(async () => {
    if (!renameDialogArtifact) return;
    const nextName = renameValue.trim() || renameDialogArtifact.name;
    await galleryStore.update(renameDialogArtifact.id, { name: nextName });
    setRenameDialogArtifact(null);
    setRenameValue('');
    await refreshGallery();
    setTimedStatus('Artifact renamed', 1500);
  }, [refreshGallery, renameDialogArtifact, renameValue, setTimedStatus]);

  const openDeleteDialog = useCallback((artifact: SavedArtifact) => {
    setDeleteDialogArtifact(artifact);
  }, []);

  const confirmDeleteArtifact = useCallback(async () => {
    if (!deleteDialogArtifact) return;
    await galleryStore.delete(deleteDialogArtifact.id);
    setDeleteDialogArtifact(null);
    await refreshGallery();
    setTimedStatus('Artifact deleted', 1500);
  }, [deleteDialogArtifact, refreshGallery, setTimedStatus]);

  const acceptPendingDiff = useCallback(() => {
    if (!pendingDiff) return;
    setArtifactType(pendingDiff.type);
    setCode(pendingDiff.modified);
    codeRef.current = pendingDiff.modified;
    setIframeError(null);
    setEditorDiagnostics([]);
    pushVersion(pendingDiff.prompt, pendingDiff.modified);
    if (iframeRef.current) {
      iframeRef.current.srcdoc = buildSrcdoc(pendingDiff.type, pendingDiff.modified, true);
    }
    setPendingDiff(null);
    setViewMode('preview');
    setTimedStatus('Changes applied', 1800);
  }, [pendingDiff, pushVersion, setTimedStatus]);

  const rejectPendingDiff = useCallback(() => {
    setPendingDiff(null);
    setTimedStatus('Kept current artifact', 1800);
  }, [setTimedStatus]);

  const handleEditorCodeChange = useCallback((nextCode: string) => {
    if (panelMode !== 'generate') return;
    setCode(nextCode);
    codeRef.current = nextCode;
    setIframeError(null);
    setEditorDiagnostics([]);
    lastAutoFixSignatureRef.current = null;
  }, [panelMode]);

  useEffect(() => {
    if (panelMode !== 'generate' || loading) return;
    if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);

    if (!code.trim()) {
      if (iframeRef.current) iframeRef.current.srcdoc = '';
      return;
    }

    manualPreviewTimerRef.current = setTimeout(() => {
      if (iframeRef.current) {
        iframeRef.current.srcdoc = buildSrcdoc(artifactType, code, true);
      }
    }, 250);

    return () => {
      if (manualPreviewTimerRef.current) clearTimeout(manualPreviewTimerRef.current);
    };
  }, [artifactType, code, loading, panelMode]);

  /* ─── View mode tabs ─── */
  const viewTabs: { mode: ViewMode; icon: React.ReactNode; label: string }[] = [
    { mode: 'preview', icon: <Eye size={13} />, label: 'Preview' },
    { mode: 'code', icon: <Code2 size={13} />, label: 'Code' },
    { mode: 'split', icon: <SplitSquareHorizontal size={13} />, label: 'Split' },
  ];

  /* ─── Current view mode based on panel mode ─── */
  const currentViewMode = panelMode === 'live' ? liveViewMode : viewMode;
  const setCurrentViewMode = panelMode === 'live' ? setLiveViewMode : setViewMode;
  const currentIframeRef = panelMode === 'live' ? liveIframeRef : iframeRef;
  const hasCode = panelMode === 'live' ? !!activeLiveFile?.code : !!code;
  const filteredConsoleLogs = consoleLogs.filter((log) => consoleFilter === 'all' ? true : log.level === consoleFilter);
  const errorLogCount = consoleLogs.filter((log) => log.level === 'error').length;
  const normalizedGalleryQuery = galleryQuery.trim().toLowerCase();
  const filteredGalleryItems = galleryItems.filter((item) => {
    const matchesType = galleryTypeFilter === 'all' || item.type === galleryTypeFilter;
    const haystack = `${item.name} ${item.prompt || ''} ${(item.tags || []).join(' ')}`.toLowerCase();
    const matchesQuery = !normalizedGalleryQuery || haystack.includes(normalizedGalleryQuery);
    return matchesType && matchesQuery;
  });

  /* ─── Render ─── */
  const panelStyle: React.CSSProperties = fullscreen
    ? { position: 'fixed', inset: 0, zIndex: 9999, background: 'var(--bg)', display: 'flex', flexDirection: 'column' }
    : { display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' };

  return (
    <div style={panelStyle}>
      {/* ═══ Top bar: mode toggle + type selector + model pill ═══ */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', flexShrink: 0,
        borderBottom: '1px solid var(--b1)',
      }}>
        <FileCode size={15} style={{ color: 'var(--pl)', flexShrink: 0 }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.02em' }}>
          Artifacts
        </span>

        {/* Mode toggle: Generate / Live */}
        <div style={{
          display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
          border: '1px solid var(--b1)', overflow: 'hidden',
        }}>
          {(['generate', 'live'] as PanelMode[]).map(m => (
            <button
              key={m}
              onClick={() => setPanelMode(m)}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '3px 10px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: panelMode === m ? 'var(--pg)' : 'transparent',
                color: panelMode === m ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s ease',
              }}
            >
              {m === 'live' && <Radio size={10} />}
              {m === 'generate' ? 'Generate' : 'Live'}
            </button>
          ))}
        </div>

        {/* Type selector (generate mode only) */}
        {panelMode === 'generate' && (
          <select
            value={artifactType}
            onChange={e => setArtifactType(e.target.value as ArtifactType)}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--tx)', fontSize: '11px', padding: '3px 8px', fontFamily: 'inherit',
              outline: 'none', cursor: 'pointer',
            }}
          >
            <option value="html">HTML / CSS / JS</option>
            <option value="react">React</option>
            <option value="svg">SVG</option>
            <option value="mermaid">Mermaid Diagram</option>
            <option value="chart">Chart.js</option>
            <option value="markdown">Markdown</option>
          </select>
        )}

        {panelMode === 'generate' && (
          <button
            onClick={() => setGalleryOpen(true)}
            style={{
              ...btnBase,
              padding: '4px 10px',
              fontSize: '10.5px',
            }}
          >
            <FolderOpen size={12} /> Gallery
            {galleryItems.length > 0 && (
              <span style={{
                minWidth: 16,
                height: 16,
                padding: '0 4px',
                borderRadius: 999,
                background: 'rgba(124,58,237,0.18)',
                color: 'var(--pl)',
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '9px',
                fontWeight: 700,
              }}>
                {galleryItems.length}
              </span>
            )}
          </button>
        )}

        {/* Live mode: connection indicator */}
        {panelMode === 'live' && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 5, fontSize: '10.5px',
            color: liveConnected ? 'var(--gr)' : 'var(--mu)',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: liveConnected ? '#10b981' : '#6b7280',
              boxShadow: liveConnected ? '0 0 6px rgba(16,185,129,0.5)' : 'none',
              animation: liveConnected ? 'pulse 2s ease-in-out infinite' : 'none',
            }} />
            {liveConnected ? 'Connected' : 'Connecting...'}
          </div>
        )}

        <div style={{ flex: 1 }} />
        {fullscreen && (
          <button onClick={() => setFullscreen(false)} style={{ ...btnBase, padding: '4px 8px' }}>
            <Minimize2 size={13} /> Exit
          </button>
        )}
        {panelMode === 'generate' && <ModelPill featureKey="artifacts" />}
      </div>

      {/* ═══ Generate mode: prompt input ═══ */}
      {panelMode === 'generate' && (
        <div style={{
          padding: '10px 12px', flexShrink: 0, borderBottom: '1px solid var(--b1)',
          display: 'flex', flexDirection: 'column', gap: 8,
        }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {QUICK_STARTS.map(qs => (
              <button
                key={qs.label}
                onClick={() => handleQuickStart(qs)}
                disabled={loading}
                onMouseEnter={() => setHoveredBtn(`qs-${qs.label}`)}
                onMouseLeave={() => setHoveredBtn(null)}
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4,
                  padding: '4px 10px', fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                  background: hoveredBtn === `qs-${qs.label}` ? 'var(--pg)' : 'transparent',
                  border: '1px solid var(--b1)',
                  borderColor: hoveredBtn === `qs-${qs.label}` ? 'rgba(124,58,237,0.2)' : 'var(--b1)',
                  borderRadius: 'var(--r-pill)', color: hoveredBtn === `qs-${qs.label}` ? 'var(--pl)' : 'var(--mu)',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s ease',
                  opacity: loading ? 0.5 : 1,
                }}
              >
                {qs.icon} {qs.label}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <textarea
              ref={textareaRef}
              value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe what to create... (Ctrl+Enter to generate)"
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  generate();
                }
              }}
              style={{
                flex: 1, background: 'var(--s2)', border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
                padding: '8px 10px', resize: 'none', height: 52, outline: 'none',
                fontFamily: 'inherit', lineHeight: 1.5,
                transition: 'border-color 0.2s ease',
              }}
              onFocus={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)'; }}
              onBlur={e => { e.currentTarget.style.borderColor = 'var(--b1)'; }}
            />
            <button
              onClick={() => generate()}
              disabled={loading || !prompt.trim()}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                background: loading ? 'var(--s3)' : 'var(--p)',
                border: 'none', borderRadius: 'var(--r-md)', color: '#fff',
                padding: '0 16px', cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer',
                fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                alignSelf: 'stretch', minWidth: 44,
                opacity: !prompt.trim() && !loading ? 0.5 : 1,
                transition: 'all 0.2s ease',
                boxShadow: loading ? 'none' : '0 2px 10px rgba(124,58,237,0.3)',
              }}
            >
              {loading ? (
                <div style={{ display: 'flex', gap: 3 }}>
                  <span style={{ width: 4, height: 4, borderRadius: '50%', background: '#fff', animation: 'dotPulse 1.2s ease-in-out infinite' }} />
                  <span style={{ width: 4, height: 4, borderRadius: '50%', background: '#fff', animation: 'dotPulse 1.2s ease-in-out infinite 0.2s' }} />
                  <span style={{ width: 4, height: 4, borderRadius: '50%', background: '#fff', animation: 'dotPulse 1.2s ease-in-out infinite 0.4s' }} />
                </div>
              ) : (
                <>
                  <Sparkles size={14} />
                  <ChevronRight size={13} />
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* ═══ Live mode: "Last updated" indicator ═══ */}
      {panelMode === 'live' && activeLiveFile && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '4px 12px', fontSize: '10.5px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: 'rgba(16,185,129,0.04)',
          color: 'var(--mu)',
        }}>
          <Radio size={10} style={{ color: '#10b981' }} />
          <span>
            <strong style={{ color: 'var(--tx)' }}>{activeLiveFile.filename}</strong>
            {' '}&middot;{' '}
            <span style={{ fontVariantNumeric: 'tabular-nums' }}>
              {activeLiveFile.type}
            </span>
            {' '}&middot;{' '}
            Last updated: {timeAgo(activeLiveFile.timestamp)}
            {' '}&middot;{' '}
            {activeLiveFile.code.length.toLocaleString()} chars
          </span>
          <div style={{ flex: 1 }} />
          {liveFileNames.length > 1 && (
            <button
              onClick={liveClearAll}
              style={{ ...btnBase, padding: '3px 8px', fontSize: '9.5px', flexShrink: 0 }}
            >
              Clear all
            </button>
          )}
        </div>
      )}

      {/* ═══ Status bar ═══ */}
      {(status || iframeError || isAutoFixing) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: isAutoFixing
            ? 'rgba(59,130,246,0.08)'
            : iframeError
              ? 'rgba(239,68,68,0.06)'
              : status === 'Copied!'
                ? 'rgba(16,185,129,0.06)'
                : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: isAutoFixing
              ? '#60a5fa'
              : iframeError
                ? 'var(--rd)'
                : status === 'Copied!'
                  ? 'var(--gr)'
                  : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {isAutoFixing
              ? `Auto-fixing preview error (${Math.min(autoFixAttempts + 1, MAX_AUTO_FIX_ATTEMPTS)}/${MAX_AUTO_FIX_ATTEMPTS})`
              : iframeError
                ? `Error: ${iframeError}`
                : status}
          </span>
          {iframeError && panelMode === 'generate' && (
            <button
              onClick={fixError}
              disabled={isAutoFixing}
              style={{
                ...btnBase, padding: '3px 10px', fontSize: '10.5px',
                background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.2)',
                color: 'var(--rd)', flexShrink: 0,
                opacity: isAutoFixing ? 0.6 : 1,
                cursor: isAutoFixing ? 'wait' : 'pointer',
              }}
            >
              <Wand2 size={11} /> {isAutoFixing ? 'Fixing...' : 'Fix with AI'}
            </button>
          )}
        </div>
      )}

      {/* ═══ View mode tabs (when code exists) ═══ */}
      {hasCode && (
        <div style={{
          display: 'flex', alignItems: 'center', flexShrink: 0,
          borderBottom: '1px solid var(--b1)', padding: '0 8px',
        }}>
          {viewTabs.map(t => (
            <button
              key={t.mode}
              onClick={() => setCurrentViewMode(t.mode)}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '7px 12px', background: 'none', border: 'none',
                borderBottom: currentViewMode === t.mode ? '2px solid var(--p)' : '2px solid transparent',
                color: currentViewMode === t.mode ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11.5px', cursor: 'pointer', fontFamily: 'inherit',
                transition: 'all 0.15s ease',
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
          <div style={{ flex: 1 }} />
          <button
            onClick={() => setFullscreen(f => !f)}
            style={{ ...btnBase, padding: '3px 6px', border: 'none', background: 'none' }}
            title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {fullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        </div>
      )}

      {/* ═══ Main content area ═══ */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden', display: 'flex' }}>
        {panelMode === 'live' && liveFileNames.length > 0 && (
          <div style={{
            width: 220,
            borderRight: '1px solid var(--b1)',
            background: 'rgba(255,255,255,0.02)',
            flexShrink: 0,
            minHeight: 0,
          }}>
            <FileTree
              files={liveFileNames}
              directories={liveDirectories}
              activeFile={activeLiveFile?.filename || ''}
              onFileSelect={setLiveActiveFile}
              onFileDelete={liveClearFile}
              title="Live Files"
            />
          </div>
        )}
        {/* ── Code editor pane ── */}
        {hasCode && (currentViewMode === 'code' || currentViewMode === 'split') && (
          <div style={{
            width: currentViewMode === 'split' ? '50%' : '100%',
            height: '100%',
            background: '#0d0d14',
            borderRight: currentViewMode === 'split' ? '1px solid var(--b1)' : 'none',
            display: 'flex',
            flexDirection: 'column',
          }}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px',
              background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
              flexShrink: 0,
            }}>
              <span style={{
                fontSize: '9.5px', fontWeight: 600, letterSpacing: '0.06em',
                textTransform: 'uppercase', color: '#a78bfa',
                background: 'rgba(167,139,250,0.1)', padding: '2px 8px', borderRadius: 3,
              }}>
                {activeType}
              </span>
              <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.3)', fontVariantNumeric: 'tabular-nums' }}>
                {activeCode.length.toLocaleString()} chars
              </span>
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <CodeEditor
                code={activeCode}
                diagnostics={editorDiagnostics}
                language={getArtifactEditorLanguage(activeType)}
                onChange={panelMode === 'generate' ? handleEditorCodeChange : undefined}
                readOnly={panelMode === 'live' || loading}
              />
            </div>
          </div>
        )}

        {/* ── Preview pane (iframe) ── */}
        {hasCode && (currentViewMode === 'preview' || currentViewMode === 'split') && (
          <div style={{
            width: currentViewMode === 'split' ? '50%' : '100%',
            height: '100%', position: 'relative',
          }}>
            <iframe
              ref={currentIframeRef}
              sandbox="allow-scripts"
              style={{
                width: '100%', height: '100%', border: 'none',
                background: '#0a0a0f',
              }}
            />
            {loading && panelMode === 'generate' && !code && (
              <div style={{
                position: 'absolute', inset: 0, display: 'flex',
                flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
                background: 'rgba(3,3,3,0.6)', backdropFilter: 'blur(4px)',
                zIndex: 5,
              }}>
                <div className="aura-thinking">
                  <span /><span /><span />
                </div>
                <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                  Generating...
                </span>
              </div>
            )}
            {loading && panelMode === 'generate' && code && (
              <div style={{
                position: 'absolute', bottom: 12, right: 12, zIndex: 5,
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '6px 14px', borderRadius: 'var(--r-pill)',
                background: 'rgba(124,58,237,0.15)', border: '1px solid rgba(124,58,237,0.25)',
                backdropFilter: 'blur(8px)',
              }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: 'var(--pl)',
                  animation: 'pulse 1.5s ease-in-out infinite',
                }} />
                <span style={{ fontSize: '11px', color: 'var(--pl)', fontWeight: 500 }}>
                  Streaming... {code.length.toLocaleString()} chars
                </span>
              </div>
            )}
            {isAutoFixing && panelMode === 'generate' && (
              <div style={{
                position: 'absolute', top: 12, right: 12, zIndex: 6,
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '6px 12px', borderRadius: 'var(--r-pill)',
                background: 'rgba(59,130,246,0.14)', border: '1px solid rgba(59,130,246,0.28)',
                backdropFilter: 'blur(8px)',
              }}>
                <span style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: '#60a5fa',
                  animation: 'pulse 1.25s ease-in-out infinite',
                }} />
                <span style={{ fontSize: '10.5px', color: '#bfdbfe', fontWeight: 600 }}>
                  Fixing... {Math.min(autoFixAttempts + 1, MAX_AUTO_FIX_ATTEMPTS)}/{MAX_AUTO_FIX_ATTEMPTS}
                </span>
              </div>
            )}
            {/* Console Drawer */}
            {consoleOpen && (
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0, maxHeight: 200,
                background: '#1e1e1e', borderTop: '1px solid #333', overflow: 'auto',
                fontFamily: 'monospace', fontSize: '11px', zIndex: 10,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px', background: '#252526', borderBottom: '1px solid #333', flexWrap: 'wrap' }}>
                  <span style={{ color: '#ccc', fontSize: '10px', fontWeight: 600 }}>Console</span>
                  {CONSOLE_FILTERS.map((filter) => (
                    <button
                      key={filter}
                      onClick={() => setConsoleFilter(filter)}
                      style={{
                        background: consoleFilter === filter ? 'rgba(55,148,255,0.18)' : 'transparent',
                        border: '1px solid ' + (consoleFilter === filter ? 'rgba(55,148,255,0.35)' : 'transparent'),
                        color: consoleFilter === filter ? '#8cc6ff' : '#888',
                        cursor: 'pointer',
                        fontSize: '9px',
                        padding: '2px 6px',
                        borderRadius: 999,
                        textTransform: 'uppercase',
                      }}
                    >
                      {filter}
                    </button>
                  ))}
                  <span style={{ flex: 1 }} />
                  <button onClick={clearConsole} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '10px', padding: '2px 6px' }}>Clear</button>
                  <button onClick={() => setConsoleOpen(false)} style={{ background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: '14px', padding: '2px 6px' }}>&times;</button>
                </div>
                {filteredConsoleLogs.map((log) => {
                  const expanded = expandedConsoleIds.includes(log.id);
                  const text = log.args.map(parseConsoleArg).join(' ');
                  return (
                    <div
                      key={log.id}
                      onClick={() => {
                        navigator.clipboard.writeText([text, log.stack].filter(Boolean).join('\n\n')).catch(() => {});
                        setTimedStatus('Console entry copied', 1200);
                      }}
                      style={{
                        padding: '5px 8px', borderBottom: '1px solid #2a2a2a',
                        color: log.level === 'error' ? '#f44747' : log.level === 'warn' ? '#cca700' : log.level === 'info' ? '#3794ff' : '#d4d4d4',
                        cursor: 'pointer',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                        <span style={{ opacity: 0.7, minWidth: 42 }}>{log.level.toUpperCase()}</span>
                        <span style={{ flex: 1, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>{text}</span>
                        {log.stack && (
                          <button
                            onClick={(event) => {
                              event.stopPropagation();
                              toggleConsoleEntry(log.id);
                            }}
                            style={{ background: 'none', border: 'none', color: '#8cc6ff', cursor: 'pointer', fontSize: '10px', padding: 0 }}
                          >
                            {expanded ? 'Hide' : 'Stack'}
                          </button>
                        )}
                      </div>
                      {expanded && log.stack && (
                        <pre style={{
                          margin: '6px 0 0 50px',
                          padding: '6px 8px',
                          background: 'rgba(0,0,0,0.28)',
                          borderRadius: 6,
                          color: '#cbd5e1',
                          whiteSpace: 'pre-wrap',
                        }}>
                          {log.stack}
                        </pre>
                      )}
                    </div>
                  );
                })}
                {filteredConsoleLogs.length === 0 && (
                  <div style={{ padding: '8px', color: '#666', textAlign: 'center' }}>
                    {consoleLogs.length === 0 ? 'No console output' : `No ${consoleFilter} entries`}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── Empty state: Generate mode ── */}
        {panelMode === 'generate' && !code && !loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24,
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--pg)', border: '1px solid rgba(124,58,237,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Sparkles size={24} style={{ color: 'var(--pl)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                Create anything
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 260, lineHeight: 1.5 }}>
                Webpages, React components, charts, diagrams, games, presentations — all rendered live.
              </div>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', maxWidth: 320 }}>
              {QUICK_STARTS.map(qs => (
                <button
                  key={qs.label}
                  onClick={() => handleQuickStart(qs)}
                  onMouseEnter={() => setHoveredBtn(`empty-${qs.label}`)}
                  onMouseLeave={() => setHoveredBtn(null)}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5,
                    padding: '6px 14px', fontSize: '11px', fontFamily: 'inherit', fontWeight: 500,
                    background: hoveredBtn === `empty-${qs.label}` ? 'var(--pg)' : 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderColor: hoveredBtn === `empty-${qs.label}` ? 'rgba(124,58,237,0.2)' : 'var(--b1)',
                    borderRadius: 'var(--r-pill)',
                    color: hoveredBtn === `empty-${qs.label}` ? 'var(--pl)' : 'var(--mu)',
                    cursor: 'pointer', transition: 'all 0.15s ease',
                  }}
                >
                  {qs.icon} {qs.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Empty state: Live mode ── */}
        {panelMode === 'live' && liveFileNames.length === 0 && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24,
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: liveConnected ? 'rgba(16,185,129,0.08)' : 'var(--s2)',
              border: `1px solid ${liveConnected ? 'rgba(16,185,129,0.15)' : 'var(--b1)'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Radio size={24} style={{ color: liveConnected ? '#10b981' : 'var(--mu)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                Live Preview
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 280, lineHeight: 1.5 }}>
                {liveConnected
                  ? 'Waiting for the agent to write HTML, React, CSS, SVG, or Markdown files. Preview will appear automatically.'
                  : 'Connecting to the artifacts stream...'}
              </div>
            </div>
            {liveConnected && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 14px', borderRadius: 'var(--r-pill)',
                background: 'rgba(16,185,129,0.06)', border: '1px solid rgba(16,185,129,0.12)',
                fontSize: '10.5px', color: '#10b981',
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', background: '#10b981',
                  animation: 'pulse 2s ease-in-out infinite',
                }} />
                Listening for file changes
              </div>
            )}
          </div>
        )}

        {/* ── Loading state: Generate mode (no code yet) ── */}
        {panelMode === 'generate' && !code && loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
          }}>
            <div className="aura-thinking">
              <span /><span /><span />
            </div>
            <span style={{
              fontSize: '12px', color: 'var(--pl)', fontWeight: 500,
              animation: 'pulse 1.5s ease-in-out infinite',
            }}>
              Generating...
            </span>
          </div>
        )}
      </div>

      {saveDialogOpen && (
        <OverlayModal
          onClose={() => setSaveDialogOpen(false)}
          title="Save Artifact"
          icon={<Save size={16} style={{ color: 'var(--pl)' }} />}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              Give this artifact a name so you can reload it later from the gallery.
            </div>
            <input
              value={saveName}
              onChange={(event) => setSaveName(event.target.value)}
              placeholder="Artifact name"
              autoFocus
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void saveCurrentArtifact();
                }
              }}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setSaveDialogOpen(false)} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void saveCurrentArtifact()}
                style={{
                  background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Save
              </button>
            </div>
        </OverlayModal>
      )}

      {renameDialogArtifact && (
        <OverlayModal
          onClose={() => { setRenameDialogArtifact(null); setRenameValue(''); }}
          title="Rename Artifact"
          icon={<Pencil size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10011}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              Update how this artifact appears in the gallery.
            </div>
            <input
              value={renameValue}
              onChange={(event) => setRenameValue(event.target.value)}
              placeholder="Artifact name"
              autoFocus
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  event.preventDefault();
                  void submitRenameArtifact();
                }
              }}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                color: 'var(--tx)', fontSize: '12px', padding: '10px 12px', outline: 'none', fontFamily: 'inherit',
              }}
            />
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => { setRenameDialogArtifact(null); setRenameValue(''); }} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void submitRenameArtifact()}
                style={{
                  background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Save Name
              </button>
            </div>
        </OverlayModal>
      )}

      {deleteDialogArtifact && (
        <OverlayModal
          onClose={() => setDeleteDialogArtifact(null)}
          title="Delete Artifact"
          icon={<Trash2 size={16} style={{ color: '#fca5a5' }} />}
          zIndex={10012}
        >
            <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
              Permanently remove <span style={{ color: 'var(--tx)', fontWeight: 700 }}>{deleteDialogArtifact.name}</span> from the gallery?
            </div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={() => setDeleteDialogArtifact(null)} style={{ ...btnBase, padding: '8px 12px' }}>
                Cancel
              </button>
              <button
                onClick={() => void confirmDeleteArtifact()}
                style={{
                  background: '#dc2626', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                  padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                }}
              >
                Delete
              </button>
            </div>
        </OverlayModal>
      )}

      {pendingDiff && (
        <OverlayModal
          onClose={rejectPendingDiff}
          title="Review Artifact Changes"
          icon={<GitFork size={16} style={{ color: 'var(--pl)' }} />}
          zIndex={10013}
          contentStyle={{
            width: 'min(1120px, 100%)',
            height: 'min(82vh, 900px)',
            gap: 10,
          }}
        >
          <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
            Compare the current artifact with Aura&apos;s proposed update before it replaces the live version.
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: '10px', color: 'var(--mu)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            <div>Current</div>
            <div>AI Proposal</div>
          </div>
          <React.Suspense
            fallback={
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--mu)', fontSize: 12 }}>
                Loading diff...
              </div>
            }
          >
            <DiffEditor
              original={pendingDiff.original}
              modified={pendingDiff.modified}
              language={getArtifactEditorLanguage(pendingDiff.type)}
            />
          </React.Suspense>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button onClick={rejectPendingDiff} style={{ ...btnBase, padding: '8px 12px' }}>
              Keep Current
            </button>
            <button
              onClick={acceptPendingDiff}
              style={{
                background: 'var(--p)', color: '#fff', border: 'none', borderRadius: 'var(--r-md)',
                padding: '8px 14px', cursor: 'pointer', fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              }}
            >
              Apply Changes
            </button>
          </div>
        </OverlayModal>
      )}

      {galleryOpen && (
        <div
          onClick={() => setGalleryOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 10000,
            background: 'rgba(0,0,0,0.48)',
            display: 'flex', justifyContent: 'flex-end',
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: 'min(520px, 100vw)',
              height: '100%',
              background: '#10111a',
              borderLeft: '1px solid rgba(255,255,255,0.08)',
              boxShadow: '-20px 0 50px rgba(0,0,0,0.35)',
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '16px 16px 12px', borderBottom: '1px solid var(--b1)' }}>
              <FolderOpen size={16} style={{ color: 'var(--pl)' }} />
              <div>
                <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--tx)' }}>Artifact Gallery</div>
                <div style={{ fontSize: '10.5px', color: 'var(--mu)' }}>{galleryItems.length} saved items</div>
              </div>
              <div style={{ flex: 1 }} />
              <button onClick={() => setGalleryOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
                <X size={16} />
              </button>
            </div>

            <div style={{ padding: '12px 16px', display: 'flex', gap: 8, borderBottom: '1px solid var(--b1)' }}>
              <div style={{ position: 'relative', flex: 1 }}>
                <Search size={12} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--mu)' }} />
                <input
                  value={galleryQuery}
                  onChange={(event) => setGalleryQuery(event.target.value)}
                  placeholder="Search saved artifacts"
                  style={{
                    width: '100%',
                    background: 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-md)',
                    color: 'var(--tx)',
                    fontSize: '11.5px',
                    padding: '8px 10px 8px 30px',
                    outline: 'none',
                    fontFamily: 'inherit',
                  }}
                />
              </div>
              <select
                value={galleryTypeFilter}
                onChange={(event) => setGalleryTypeFilter(event.target.value as 'all' | ArtifactType)}
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  color: 'var(--tx)', fontSize: '11px', padding: '8px 10px', fontFamily: 'inherit',
                }}
              >
                <option value="all">All types</option>
                <option value="html">HTML</option>
                <option value="react">React</option>
                <option value="svg">SVG</option>
                <option value="mermaid">Mermaid</option>
                <option value="chart">Chart</option>
                <option value="markdown">Markdown</option>
                <option value="css">CSS</option>
              </select>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
              {filteredGalleryItems.map((item) => (
                <div
                  key={item.id}
                  style={{
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 14,
                    background: 'rgba(255,255,255,0.02)',
                    overflow: 'hidden',
                  }}
                >
                  <div style={{
                    height: 112,
                    borderBottom: '1px solid rgba(255,255,255,0.06)',
                    background: item.thumbnail
                      ? `center / cover no-repeat url("${item.thumbnail}")`
                      : 'linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.12))',
                    position: 'relative',
                    display: 'flex',
                    alignItems: 'flex-end',
                  }}>
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      background: 'linear-gradient(180deg, rgba(3,7,18,0.04), rgba(3,7,18,0.72))',
                    }} />
                    <div style={{
                      position: 'relative',
                      zIndex: 1,
                      width: '100%',
                      padding: 12,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 6,
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          fontSize: '9px', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase',
                          color: '#d8b4fe', background: 'rgba(15,23,42,0.48)', padding: '3px 7px', borderRadius: 999,
                        }}>
                          {item.type}
                        </span>
                        <span style={{ fontSize: '10px', color: '#e2e8f0' }}>
                          {new Date(item.updatedAt).toLocaleDateString()}
                        </span>
                      </div>
                      <div style={{ fontSize: '13px', fontWeight: 700, color: '#f8fafc' }}>{item.name}</div>
                    </div>
                  </div>
                  <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <div style={{ fontSize: '11px', color: 'var(--mu)', lineHeight: 1.5 }}>
                      {(item.prompt || item.code).slice(0, 140)}{(item.prompt || item.code).length > 140 ? '...' : ''}
                    </div>
                    {!!item.tags?.length && (
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {item.tags.slice(0, 4).map((tag) => (
                          <span
                            key={tag}
                            style={{
                              fontSize: '9px',
                              color: '#c4b5fd',
                              background: 'rgba(124,58,237,0.12)',
                              border: '1px solid rgba(124,58,237,0.18)',
                              padding: '3px 6px',
                              borderRadius: 999,
                            }}
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, fontSize: '10px', color: 'var(--mu)' }}>
                      <span>{item.code.length.toLocaleString()} chars</span>
                      <span>Created {new Date(item.createdAt).toLocaleDateString()}</span>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      <button onClick={() => loadSavedArtifact(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <FolderOpen size={12} /> Load
                      </button>
                      <button onClick={() => void forkSavedArtifact(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <GitFork size={12} /> Fork
                      </button>
                      <button onClick={() => openRenameDialog(item)} style={{ ...btnBase, padding: '6px 10px', fontSize: '10.5px' }}>
                        <Pencil size={12} /> Rename
                      </button>
                      <button
                        onClick={() => openDeleteDialog(item)}
                        style={{
                          ...btnBase, padding: '6px 10px', fontSize: '10.5px',
                          color: '#fca5a5', borderColor: 'rgba(239,68,68,0.18)',
                        }}
                      >
                        <Trash2 size={12} /> Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}

              {filteredGalleryItems.length === 0 && (
                <div style={{
                  flex: 1,
                  minHeight: 180,
                  border: '1px dashed rgba(255,255,255,0.12)',
                  borderRadius: 16,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  color: 'var(--mu)',
                  textAlign: 'center',
                  padding: 24,
                }}>
                  <FolderOpen size={22} style={{ color: 'var(--pl)' }} />
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)' }}>
                    {galleryItems.length === 0 ? 'No saved artifacts yet' : 'No matches for this filter'}
                  </div>
                  <div style={{ fontSize: '10.5px', lineHeight: 1.5, maxWidth: 280 }}>
                    {galleryItems.length === 0
                      ? 'Save generated work here so you can reload it later and use it as a starting point.'
                      : 'Try another search term or switch the type filter.'}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══ Footer action bar ═══ */}
      {hasCode && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          <ActionBtn id="copy" icon={<Copy size={13} />} label="Copy Code" onClick={copyCode} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <ActionBtn id="download" icon={<Download size={13} />} label="Download" onClick={downloadFile} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <ActionBtn
            id="savecomp"
            icon={<Package size={13} />}
            label="Save Component"
            onClick={async () => {
              const compName = window.prompt('Component name:');
              if (!compName) return;
              const { componentLibrary } = await import('../utils/componentLibrary');
              await componentLibrary.save({
                name: compName,
                description: 'Saved from Artifacts',
                category: 'other',
                html: activeCode,
                css: '',
                tags: [activeType],
                source: 'generated',
              });
              setTimedStatus('Component saved!');
            }}
            hoveredBtn={hoveredBtn}
            setHoveredBtn={setHoveredBtn}
          />
          {panelMode === 'generate' && (
            <ActionBtn id="save" icon={<Save size={13} />} label="Save" onClick={openSaveDialog} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          )}
          {panelMode === 'generate' && (
            <ActionBtn id="clear" icon={<X size={13} />} label="Clear" onClick={clearGeneratedArtifact} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          )}
          <button
            onClick={() => { const v = undo(); if (v) { setCode(v.code); codeRef.current = v.code; if (iframeRef.current) iframeRef.current.srcdoc = buildSrcdoc(artifactType, v.code, true); } }}
            disabled={!canUndo}
            title="Undo"
            style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', color: canUndo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canUndo ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', opacity: canUndo ? 1 : 0.4 }}
          >
            <Undo2 size={12} />
          </button>
          <button
            onClick={() => { const v = redo(); if (v) { setCode(v.code); codeRef.current = v.code; if (iframeRef.current) iframeRef.current.srcdoc = buildSrcdoc(artifactType, v.code, true); } }}
            disabled={!canRedo}
            title="Redo"
            style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', color: canRedo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canRedo ? 'pointer' : 'not-allowed', display: 'flex', alignItems: 'center', opacity: canRedo ? 1 : 0.4 }}
          >
            <Redo2 size={12} />
          </button>
          <button
            onClick={() => setConsoleOpen(!consoleOpen)}
            title="Console"
            style={{
              background: consoleOpen ? 'rgba(55,148,255,0.15)' : 'var(--s2)',
              border: `1px solid ${consoleOpen ? 'rgba(55,148,255,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: consoleOpen ? '#3794ff' : 'var(--mu)',
              padding: '3px 6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
              fontSize: '10px', fontFamily: 'inherit', position: 'relative',
            }}
          >
            <Terminal size={12} /> Console
            {(unreadErrorCount > 0 || errorLogCount > 0) && (
              <span style={{
                position: 'absolute', top: -4, right: -4, background: '#f44747',
                color: 'white', borderRadius: '50%', width: 14, height: 14,
                fontSize: '9px', display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {consoleOpen ? errorLogCount : unreadErrorCount || errorLogCount}
              </span>
            )}
          </button>
          {panelMode === 'generate' && (
            <ActionBtn id="remix" icon={<RotateCcw size={13} />} label="Remix" onClick={remix} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          )}
          {iframeError && panelMode === 'generate' && (
            <ActionBtn id="fix" icon={<Wrench size={13} />} label="Fix Error" onClick={fixError} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          )}
          <div style={{ flex: 1 }} />
          <ActionBtn
            id="fullscreen"
            icon={fullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            label={fullscreen ? 'Exit' : 'Full Screen'}
            onClick={() => setFullscreen(f => !f)}
            hoveredBtn={hoveredBtn}
            setHoveredBtn={setHoveredBtn}
          />
        </div>
      )}
      {/* Version timeline */}
      {versions.length > 1 && (
        <div style={{
          display: 'flex', gap: 4, padding: '4px 8px', borderTop: '1px solid var(--b1)',
          overflowX: 'auto', background: 'var(--s1)', flexShrink: 0,
        }}>
          {versions.map((v, i) => (
            <button
              key={v.id}
              onClick={() => { const ver = goToVersion(i); if (ver) { setCode(ver.code); codeRef.current = ver.code; if (iframeRef.current) iframeRef.current.srcdoc = buildSrcdoc(artifactType, ver.code, true); } }}
              title={v.prompt}
              style={{
                flexShrink: 0, padding: '2px 8px', fontSize: '10px',
                background: i === currentIdx ? 'var(--p)' : 'var(--s2)',
                color: i === currentIdx ? 'white' : 'var(--mu)',
                border: '1px solid ' + (i === currentIdx ? 'var(--p)' : 'var(--b1)'),
                borderRadius: 'var(--r-pill)', cursor: 'pointer', fontFamily: 'inherit',
                maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}
            >
              v{i + 1}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
