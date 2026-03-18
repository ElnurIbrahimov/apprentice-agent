import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Copy, Download, Maximize2, Minimize2, Code2, Eye, SplitSquareHorizontal,
  Sparkles, Wand2, Wrench, RotateCcw, Globe, BarChart3, GitBranch,
  Gamepad2, Presentation, FileCode, ChevronRight, Radio, X,
} from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';

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
  react: 'You are an expert React developer. Respond with ONLY the JavaScript code for a React component. Do NOT include import statements — React and ReactDOM are available globally. Render your root component using: ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App)). No markdown fences, no explanation. Use inline styles or a <style> tag injected via JS.',
  svg: 'You are an expert SVG artist. Respond with ONLY valid SVG markup. No markdown fences, no explanation.',
  mermaid: 'You are an expert at Mermaid.js diagrams. Respond with ONLY the mermaid diagram definition (e.g., starting with "graph TD", "mindmap", "sequenceDiagram", "flowchart LR", etc.). No markdown fences, no explanation, no HTML wrapping.',
  chart: 'You are an expert data visualization developer using Chart.js. Respond with ONLY JavaScript code that creates Chart.js charts. The canvas elements should be created in JS and appended to document.getElementById("root"). Chart.js is available globally as Chart. No markdown fences, no explanation. Create beautiful charts with good color schemes.',
  markdown: 'You are a technical writer. Respond with ONLY well-formatted Markdown content. No HTML, no fences around the whole thing, no explanation.',
};

const ARTIFACTS_WS_URL = `ws://localhost:8000/api/artifacts/stream`;

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
</script>`
    : '';

  if (type === 'html') {
    if (code.includes('</head>')) {
      return code.replace('</head>', errorScript + '</head>');
    }
    return errorScript + code;
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

  // For react, chart, mermaid — use the full runtime template
  const cdnScripts: string[] = [];
  let bodyContent = '<div id="root"></div>';
  let userScript = code;

  if (type === 'react') {
    cdnScripts.push(
      '<script src="https://unpkg.com/react@18/umd/react.production.min.js"><\/script>',
      '<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"><\/script>',
    );
  } else if (type === 'chart') {
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


/* ═══════════════════════════════════════════════════════════════════════════
   Component
   ═══════════════════════════════════════════════════════════════════════════ */
export default function ArtifactsPanel() {
  const { getModel } = useStore();

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

  // --- Live mode state ---
  const [liveActiveFile, setLiveActiveFile] = useState<string | null>(null);
  const [liveViewMode, setLiveViewMode] = useState<ViewMode>('preview');
  const { files: liveFiles, connected: liveConnected, clearFile: liveClearFile, clearAll: liveClearAll } = useArtifactsWS(panelMode === 'live');

  // Auto-select latest file in live mode
  const liveFileNames = Object.keys(liveFiles);
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
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /* ─── Abort fetch on unmount ─── */
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  /* ─── iframe error listener ─── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      const targetIframe = panelMode === 'live' ? liveIframeRef.current : iframeRef.current;
      if (e.source !== targetIframe?.contentWindow) return;
      if (e.data?.type === 'artifact-error') {
        const msg = e.data.msg || 'Unknown error';
        const line = e.data.line ? ` (line ${e.data.line})` : '';
        setIframeError(`${msg}${line}`);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [panelMode]);

  /* ─── Update live preview when active file changes ─── */
  useEffect(() => {
    if (panelMode !== 'live' || !activeLiveFile || !liveIframeRef.current) return;
    setIframeError(null);
    liveIframeRef.current.srcdoc = buildSrcdoc(activeLiveFile.type, activeLiveFile.code);
  }, [panelMode, activeLiveFile?.code, activeLiveFile?.type, activeLiveFile?.filename]);

  /* ─── Update iframe preview (generate mode) ─── */
  const updatePreview = useCallback((rawCode: string, type: ArtifactType) => {
    if (!iframeRef.current) return;
    setIframeError(null);
    iframeRef.current.srcdoc = buildSrcdoc(type, rawCode);
  }, []);

  /* ─── Generate artifact ─── */
  const generate = useCallback(async (overridePrompt?: string, overrideType?: ArtifactType) => {
    const text = (overridePrompt ?? prompt).trim();
    if (!text) return;

    const type = overrideType ?? artifactType;
    setArtifactType(type);
    setLoading(true);
    setStatus('Generating...');
    setCode('');
    setIframeError(null);
    codeRef.current = '';
    if (iframeRef.current) iframeRef.current.srcdoc = '';

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const systemPrompt = SYSTEM_PROMPTS[type] || SYSTEM_PROMPTS.html;

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `${systemPrompt}\n\nTask: ${text}`,
          model: getModel('artifacts') || undefined,
        }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus((d as any).detail || resp.statusText);
        setLoading(false);
        return;
      }

      const data = await resp.json();
      const responseText = data.response || data.text || data.content || data.reply || data.message || '';

      const finalCode = stripFences(responseText);
      codeRef.current = finalCode;
      setCode(finalCode);
      updatePreview(finalCode, type);
      setStatus('');
      setViewMode('preview');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [prompt, artifactType, getModel, updatePreview]);

  /* ─── Actions ─── */
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    };
  }, []);

  const activeCode = panelMode === 'live' ? (activeLiveFile?.code || '') : code;
  const activeType = panelMode === 'live' ? (activeLiveFile?.type || 'html') : artifactType;

  const copyCode = useCallback(() => {
    if (!activeCode) return;
    navigator.clipboard.writeText(activeCode).then(() => {
      setStatus('Copied!');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 1500);
    });
  }, [activeCode]);

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

  /* ─── Button helper ─── */
  const ActionBtn = ({ id, icon, label, onClick, accent }: {
    id: string; icon: React.ReactNode; label: string; onClick: () => void; accent?: boolean;
  }) => (
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

  /* ─── View mode tabs ─── */
  const viewTabs: { mode: ViewMode; icon: React.ReactNode; label: string }[] = [
    { mode: 'preview', icon: <Eye size={13} />, label: 'Preview' },
    { mode: 'code', icon: <Code2 size={13} />, label: 'Code' },
    { mode: 'split', icon: <SplitSquareHorizontal size={13} />, label: 'Split' },
  ];

  /* ─── Syntax-highlighted code display ─── */
  const highlightedCode = React.useMemo(() => {
    if (!activeCode) return '';
    let h = escHtml(activeCode);
    h = h.replace(/(&quot;|&#39;)(.*?)\1/g, '<span style="color:#a5d6ff">$1$2$1</span>');
    h = h.replace(/(&lt;\/?)([\w-]+)/g, '$1<span style="color:#ff7b72">$2</span>');
    h = h.replace(/\s([\w-]+)=/g, ' <span style="color:#d2a8ff">$1</span>=');
    h = h.replace(/(\/\/.*?)(\n|$)/g, '<span style="color:#6a737d">$1</span>$2');
    h = h.replace(/(&lt;!--[\s\S]*?--&gt;)/g, '<span style="color:#6a737d">$1</span>');
    const kw = ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'new', 'try', 'catch', 'async', 'await'];
    for (const k of kw) {
      h = h.replace(new RegExp(`\\b(${k})\\b`, 'g'), '<span style="color:#ff7b72">$1</span>');
    }
    h = h.replace(/\b(\d+\.?\d*)\b/g, '<span style="color:#79c0ff">$1</span>');
    return h;
  }, [activeCode]);

  /* ─── Current view mode based on panel mode ─── */
  const currentViewMode = panelMode === 'live' ? liveViewMode : viewMode;
  const setCurrentViewMode = panelMode === 'live' ? setLiveViewMode : setViewMode;
  const currentIframeRef = panelMode === 'live' ? liveIframeRef : iframeRef;
  const hasCode = panelMode === 'live' ? !!activeLiveFile?.code : !!code;

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

      {/* ═══ Live mode: file tab bar ═══ */}
      {panelMode === 'live' && liveFileNames.length > 0 && (
        <div style={{
          display: 'flex', alignItems: 'center', flexShrink: 0,
          borderBottom: '1px solid var(--b1)', padding: '0 4px',
          overflowX: 'auto', gap: 1,
        }}>
          {liveFileNames.map(fname => {
            const f = liveFiles[fname];
            const isActive = activeLiveFile?.filename === fname;
            return (
              <div
                key={fname}
                onClick={() => setLiveActiveFile(fname)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 5,
                  padding: '6px 10px', cursor: 'pointer',
                  borderBottom: isActive ? '2px solid var(--p)' : '2px solid transparent',
                  color: isActive ? 'var(--pl)' : 'var(--mu)',
                  fontSize: '11px', fontFamily: 'inherit',
                  transition: 'all 0.15s ease',
                  whiteSpace: 'nowrap', flexShrink: 0,
                }}
              >
                <span style={{ fontWeight: isActive ? 600 : 400 }}>{fname}</span>
                <span style={{
                  fontSize: '9px', color: 'rgba(255,255,255,0.3)',
                  fontVariantNumeric: 'tabular-nums',
                }}>
                  {timeAgo(f.timestamp)}
                </span>
                <button
                  onClick={e => { e.stopPropagation(); liveClearFile(fname); }}
                  style={{
                    background: 'none', border: 'none', padding: '1px', cursor: 'pointer',
                    color: 'rgba(255,255,255,0.25)', display: 'flex',
                  }}
                  title="Remove from preview"
                >
                  <X size={10} />
                </button>
              </div>
            );
          })}
          <div style={{ flex: 1 }} />
          {liveFileNames.length > 1 && (
            <button
              onClick={liveClearAll}
              style={{
                ...btnBase, padding: '3px 8px', fontSize: '9.5px', flexShrink: 0,
              }}
            >
              Clear all
            </button>
          )}
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
        </div>
      )}

      {/* ═══ Status bar ═══ */}
      {(status || iframeError) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: iframeError ? 'rgba(239,68,68,0.06)' : status === 'Copied!' ? 'rgba(16,185,129,0.06)' : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: iframeError ? 'var(--rd)' : status === 'Copied!' ? 'var(--gr)' : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {iframeError ? `Error: ${iframeError}` : status}
          </span>
          {iframeError && panelMode === 'generate' && (
            <button
              onClick={fixError}
              style={{
                ...btnBase, padding: '3px 10px', fontSize: '10.5px',
                background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.2)',
                color: 'var(--rd)', flexShrink: 0,
              }}
            >
              <Wand2 size={11} /> Fix with AI
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
        {/* ── Code editor pane ── */}
        {hasCode && (currentViewMode === 'code' || currentViewMode === 'split') && (
          <div style={{
            width: currentViewMode === 'split' ? '50%' : '100%',
            height: '100%', overflow: 'auto',
            background: '#0d0d14',
            borderRight: currentViewMode === 'split' ? '1px solid var(--b1)' : 'none',
          }}>
            <div style={{
              position: 'sticky', top: 0, zIndex: 2,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px',
              background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
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
            <pre
              style={{
                margin: 0, padding: '12px 14px', background: 'transparent', border: 'none',
                fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
                fontSize: '11.5px', lineHeight: 1.6, color: '#e2e0f0',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflow: 'visible',
              }}
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightedCode) }}
            />
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
            {loading && panelMode === 'generate' && (
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
            <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
              Creating your artifact...
            </span>
          </div>
        )}
      </div>

      {/* ═══ Footer action bar ═══ */}
      {hasCode && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          <ActionBtn id="copy" icon={<Copy size={13} />} label="Copy Code" onClick={copyCode} />
          <ActionBtn id="download" icon={<Download size={13} />} label="Download" onClick={downloadFile} />
          {panelMode === 'generate' && (
            <ActionBtn id="remix" icon={<RotateCcw size={13} />} label="Remix" onClick={remix} />
          )}
          {iframeError && panelMode === 'generate' && (
            <ActionBtn id="fix" icon={<Wrench size={13} />} label="Fix Error" onClick={fixError} accent />
          )}
          <div style={{ flex: 1 }} />
          <ActionBtn
            id="fullscreen"
            icon={fullscreen ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
            label={fullscreen ? 'Exit' : 'Full Screen'}
            onClick={() => setFullscreen(f => !f)}
          />
        </div>
      )}
    </div>
  );
}
