import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Crosshair, Copy, Eye, Code2, Wand2, Sparkles, X,
  Image as ImageIcon, Layers, SplitSquareHorizontal,
} from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import ext from '../ext';

/* ─── Types ─── */
interface CapturedComponent {
  html: string;
  css: Record<string, Record<string, string>>;
  screenshot_b64: string;
  dimensions: { width: number; height: number; padding: string; margin: string };
  textContent: string;
  tagName: string;
  className: string;
}

type ViewTab = 'preview' | 'html' | 'css' | 'generated';

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

function cssMapToString(cssMap: Record<string, Record<string, string>>): string {
  const lines: string[] = [];
  for (const [selector, props] of Object.entries(cssMap)) {
    lines.push(`/* ${selector} */`);
    const entries = Object.entries(props);
    if (entries.length === 0) continue;
    lines.push(`${selector} {`);
    for (const [prop, val] of entries) {
      lines.push(`  ${prop}: ${val};`);
    }
    lines.push('}');
    lines.push('');
  }
  return lines.join('\n');
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function stripFences(s: string): string {
  return s.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
}

/* ─── Component ─── */
export default function CapturePanel() {
  const { getModel } = useStore();

  const [capturing, setCapturing] = useState(false);
  const [captured, setCaptured] = useState<CapturedComponent | null>(null);
  const [activeTab, setActiveTab] = useState<ViewTab>('preview');
  const [generatedCode, setGeneratedCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);
  const [previewSrcdoc, setPreviewSrcdoc] = useState('');

  const abortRef = useRef<AbortController | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    };
  }, []);

  // Listen for capture results from content script via background
  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === 'COMPONENT_CAPTURED') {
        setCaptured(msg.data);
        setCapturing(false);
        setActiveTab('preview');
        setGeneratedCode('');
        setPreviewSrcdoc('');
        setStatus('Component captured!');
        if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
        statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
      }
      if (msg.type === 'CAPTURE_MODE_EXITED') {
        setCapturing(false);
      }
    };
    ext?.runtime?.onMessage?.addListener(handler);

    // Also listen for forwarded custom events from App.tsx
    const onCaptured = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.data) {
        setCaptured(detail.data);
        setCapturing(false);
        setActiveTab('preview');
        setGeneratedCode('');
        setPreviewSrcdoc('');
        setStatus('Component captured!');
        if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
        statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
      }
    };
    const onExited = () => setCapturing(false);
    window.addEventListener('component-captured', onCaptured);
    window.addEventListener('capture-mode-exited', onExited);

    return () => {
      ext?.runtime?.onMessage?.removeListener(handler);
      window.removeEventListener('component-captured', onCaptured);
      window.removeEventListener('capture-mode-exited', onExited);
    };
  }, []);

  /* ─── Start / Stop capture mode ─── */
  const startCapture = useCallback(() => {
    setCapturing(true);
    setStatus('Click on any element to capture...');
    ext?.runtime?.sendMessage({ type: 'START_CAPTURE_MODE' });
  }, []);

  const stopCapture = useCallback(() => {
    setCapturing(false);
    setStatus('');
    ext?.runtime?.sendMessage({ type: 'STOP_CAPTURE_MODE' });
  }, []);

  /* ─── Generate code with AI ─── */
  const recreateWithAI = useCallback(async () => {
    if (!captured) return;
    setLoading(true);
    setStatus('Generating code...');
    setGeneratedCode('');
    setActiveTab('generated');

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const cssText = cssMapToString(captured.css);
    const prompt = `You are an expert frontend developer. Recreate this UI component using React + Tailwind CSS.

Here is the HTML structure:
\`\`\`html
${captured.html}
\`\`\`

Here are the computed styles:
\`\`\`css
${cssText.slice(0, 8000)}
\`\`\`

Element dimensions: ${captured.dimensions.width}px x ${captured.dimensions.height}px
Padding: ${captured.dimensions.padding}
Margin: ${captured.dimensions.margin}

Match the design exactly. Return ONLY a single React functional component using Tailwind CSS classes. Use inline styles only when Tailwind can't achieve the exact value. No imports, no exports — just the function component rendered with ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App)).`;

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          model: getModel('capture') || getModel('artifacts') || undefined,
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
      setGeneratedCode(finalCode);
      setStatus('Code generated!');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [captured, getModel]);

  /* ─── Preview generated code ─── */
  const previewCode = useCallback(() => {
    if (!generatedCode) return;
    const srcdoc = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://unpkg.com/react@18/umd/react.production.min.js"><\/script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"><\/script>
<script src="https://cdn.tailwindcss.com"><\/script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:24px}
#root{width:100%}
</style>
</head>
<body>
<div id="root"></div>
<script>
try {
${generatedCode}
} catch(e) {
  document.getElementById('root').innerHTML = '<pre style="color:#ef4444;padding:16px">' + e.message + '</pre>';
}
<\/script>
</body>
</html>`;
    setPreviewSrcdoc(srcdoc);
    setActiveTab('preview');
  }, [generatedCode]);

  /* ─── Copy helpers ─── */
  const copyText = useCallback((text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setStatus(`${label} copied!`);
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 1500);
    });
  }, []);

  /* ─── Syntax highlight ─── */
  const highlightCode = (code: string): string => {
    let h = escHtml(code);
    h = h.replace(/(&quot;|&#39;)(.*?)\1/g, '<span style="color:#a5d6ff">$1$2$1</span>');
    h = h.replace(/(&lt;\/?)([\w-]+)/g, '$1<span style="color:#ff7b72">$2</span>');
    h = h.replace(/\s([\w-]+)=/g, ' <span style="color:#d2a8ff">$1</span>=');
    h = h.replace(/(\/\/.*?)(\n|$)/g, '<span style="color:#6a737d">$1</span>$2');
    const kw = ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'export', 'from', 'new', 'try', 'catch', 'async', 'await'];
    for (const k of kw) {
      h = h.replace(new RegExp(`\\b(${k})\\b`, 'g'), '<span style="color:#ff7b72">$1</span>');
    }
    h = h.replace(/\b(\d+\.?\d*)\b/g, '<span style="color:#79c0ff">$1</span>');
    return h;
  };

  /* ─── Button helper ─── */
  const ActionBtn = ({ id, icon, label, onClick, accent, disabled }: {
    id: string; icon: React.ReactNode; label: string; onClick: () => void; accent?: boolean; disabled?: boolean;
  }) => (
    <button
      onClick={onClick}
      disabled={disabled}
      onMouseEnter={() => setHoveredBtn(id)}
      onMouseLeave={() => setHoveredBtn(null)}
      style={{
        ...btnBase,
        ...(hoveredBtn === id && !disabled ? btnHover : {}),
        ...(accent ? { background: 'var(--pg)', borderColor: 'rgba(124,58,237,0.2)', color: 'var(--pl)' } : {}),
        ...(disabled ? { opacity: 0.4, cursor: 'not-allowed' } : {}),
      }}
    >
      {icon} {label}
    </button>
  );

  /* ─── Tab definitions ─── */
  const tabs: { key: ViewTab; icon: React.ReactNode; label: string }[] = [
    { key: 'preview', icon: <Eye size={13} />, label: 'Preview' },
    { key: 'html', icon: <Code2 size={13} />, label: 'HTML' },
    { key: 'css', icon: <Layers size={13} />, label: 'CSS' },
    { key: 'generated', icon: <Sparkles size={13} />, label: 'AI Code' },
  ];

  /* ─── Render ─── */
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* ═══ Top bar ═══ */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', flexShrink: 0,
        borderBottom: '1px solid var(--b1)',
      }}>
        <Crosshair size={15} style={{ color: 'var(--pl)', flexShrink: 0 }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.02em' }}>
          Component Capture
        </span>
        <div style={{ flex: 1 }} />
        <ModelPill featureKey="capture" />
      </div>

      {/* ═══ Capture controls ═══ */}
      <div style={{
        padding: '10px 12px', flexShrink: 0, borderBottom: '1px solid var(--b1)',
        display: 'flex', gap: 8, alignItems: 'center',
      }}>
        {!capturing ? (
          <button
            onClick={startCapture}
            onMouseEnter={() => setHoveredBtn('start')}
            onMouseLeave={() => setHoveredBtn(null)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: hoveredBtn === 'start' ? 'var(--p)' : 'var(--pg)',
              border: '1px solid rgba(124,58,237,0.3)',
              borderRadius: 'var(--r-md)', color: 'var(--pl)',
              padding: '8px 16px', cursor: 'pointer',
              fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              transition: 'all 0.2s ease',
              boxShadow: hoveredBtn === 'start' ? '0 2px 10px rgba(124,58,237,0.3)' : 'none',
            }}
          >
            <Crosshair size={14} /> Start Capture
          </button>
        ) : (
          <button
            onClick={stopCapture}
            onMouseEnter={() => setHoveredBtn('stop')}
            onMouseLeave={() => setHoveredBtn(null)}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--r-md)', color: 'var(--rd)',
              padding: '8px 16px', cursor: 'pointer',
              fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              transition: 'all 0.2s ease',
            }}
          >
            <X size={14} /> Stop Capture
          </button>
        )}

        {capturing && (
          <span style={{
            fontSize: '11px', color: 'var(--pl)',
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--p)', boxShadow: '0 0 8px var(--p)',
              animation: 'pulse 1.5s ease-in-out infinite',
            }} />
            Hover over any element, click to capture
          </span>
        )}
      </div>

      {/* ═══ Status bar ═══ */}
      {status && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: status.includes('failed') || status.includes('error')
            ? 'rgba(239,68,68,0.06)'
            : status.includes('copied') || status.includes('captured') || status.includes('generated')
              ? 'rgba(16,185,129,0.06)'
              : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: status.includes('failed') || status.includes('error') ? 'var(--rd)'
              : status.includes('copied') || status.includes('captured') || status.includes('generated') ? 'var(--gr)'
                : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {status}
          </span>
        </div>
      )}

      {/* ═══ Tab bar (when captured) ═══ */}
      {captured && (
        <div style={{
          display: 'flex', alignItems: 'center', flexShrink: 0,
          borderBottom: '1px solid var(--b1)', padding: '0 8px',
        }}>
          {tabs.map(t => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '7px 12px', background: 'none', border: 'none',
                borderBottom: activeTab === t.key ? '2px solid var(--p)' : '2px solid transparent',
                color: activeTab === t.key ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11.5px', cursor: 'pointer', fontFamily: 'inherit',
                transition: 'all 0.15s ease',
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      )}

      {/* ═══ Main content ═══ */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        {/* Empty state */}
        {!captured && !capturing && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24,
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--pg)', border: '1px solid rgba(124,58,237,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Crosshair size={24} style={{ color: 'var(--pl)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                Capture any component
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 260, lineHeight: 1.5 }}>
                Hover over any UI element on any website, click to capture its DOM, styles, and screenshot. Then recreate it with AI.
              </div>
            </div>
          </div>
        )}

        {/* Capturing state */}
        {capturing && !captured && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
          }}>
            <div className="aura-thinking">
              <span /><span /><span />
            </div>
            <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
              Waiting for capture...
            </span>
          </div>
        )}

        {/* Preview tab */}
        {captured && activeTab === 'preview' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 0 }}>
            {/* Screenshot preview */}
            {captured.screenshot_b64 && (
              <div style={{ padding: 12 }}>
                <div style={{
                  fontSize: '9.5px', fontWeight: 600, letterSpacing: '0.06em',
                  textTransform: 'uppercase', color: '#a78bfa',
                  background: 'rgba(167,139,250,0.1)', padding: '2px 8px', borderRadius: 3,
                  display: 'inline-block', marginBottom: 8,
                }}>
                  Screenshot
                </div>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', background: '#0d0d14',
                }}>
                  <img
                    src={`data:image/png;base64,${captured.screenshot_b64}`}
                    alt="Captured component"
                    style={{ width: '100%', display: 'block' }}
                  />
                </div>
              </div>
            )}
            {/* Element info */}
            <div style={{ padding: '0 12px 12px' }}>
              <div style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
                fontSize: '11px', color: 'var(--mu)',
              }}>
                <div style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)', padding: '6px 10px',
                }}>
                  <span style={{ color: 'var(--pl)', fontWeight: 600 }}>Tag: </span>
                  {captured.tagName}
                </div>
                <div style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)', padding: '6px 10px',
                }}>
                  <span style={{ color: 'var(--pl)', fontWeight: 600 }}>Size: </span>
                  {Math.round(captured.dimensions.width)} x {Math.round(captured.dimensions.height)}
                </div>
                {captured.className && (
                  <div style={{
                    gridColumn: '1 / -1',
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '6px 10px',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    <span style={{ color: 'var(--pl)', fontWeight: 600 }}>Class: </span>
                    {captured.className}
                  </div>
                )}
              </div>
            </div>
            {/* Generated code preview iframe */}
            {previewSrcdoc && (
              <div style={{ padding: '0 12px 12px' }}>
                <div style={{
                  fontSize: '9.5px', fontWeight: 600, letterSpacing: '0.06em',
                  textTransform: 'uppercase', color: '#a78bfa',
                  background: 'rgba(167,139,250,0.1)', padding: '2px 8px', borderRadius: 3,
                  display: 'inline-block', marginBottom: 8,
                }}>
                  AI Recreation
                </div>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', height: 300,
                }}>
                  <iframe
                    srcDoc={previewSrcdoc}
                    sandbox="allow-scripts"
                    style={{ width: '100%', height: '100%', border: 'none', background: '#0a0a0f' }}
                  />
                </div>
              </div>
            )}
          </div>
        )}

        {/* HTML tab */}
        {captured && activeTab === 'html' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
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
                HTML
              </span>
              <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.3)', fontVariantNumeric: 'tabular-nums' }}>
                {captured.html.length.toLocaleString()} chars
              </span>
            </div>
            <pre
              style={{
                margin: 0, padding: '12px 14px', background: 'transparent', border: 'none',
                fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
                fontSize: '11.5px', lineHeight: 1.6, color: '#e2e0f0',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflow: 'visible',
              }}
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightCode(captured.html)) }}
            />
          </div>
        )}

        {/* CSS tab */}
        {captured && activeTab === 'css' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
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
                CSS
              </span>
            </div>
            <pre
              style={{
                margin: 0, padding: '12px 14px', background: 'transparent', border: 'none',
                fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
                fontSize: '11.5px', lineHeight: 1.6, color: '#e2e0f0',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflow: 'visible',
              }}
              dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightCode(cssMapToString(captured.css))) }}
            />
          </div>
        )}

        {/* Generated AI code tab */}
        {captured && activeTab === 'generated' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
            {!generatedCode && !loading && (
              <div style={{
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                height: '100%', gap: 12, padding: 24,
              }}>
                <Wand2 size={24} style={{ color: 'var(--pl)', opacity: 0.5 }} />
                <span style={{ fontSize: '12px', color: 'var(--mu)', textAlign: 'center' }}>
                  Click "Recreate with AI" below to generate React + Tailwind code from the captured component.
                </span>
              </div>
            )}
            {loading && (
              <div style={{
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                height: '100%', gap: 12,
              }}>
                <div className="aura-thinking">
                  <span /><span /><span />
                </div>
                <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                  Generating React + Tailwind code...
                </span>
              </div>
            )}
            {generatedCode && !loading && (
              <>
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
                    React + Tailwind
                  </span>
                  <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.3)', fontVariantNumeric: 'tabular-nums' }}>
                    {generatedCode.length.toLocaleString()} chars
                  </span>
                </div>
                <pre
                  style={{
                    margin: 0, padding: '12px 14px', background: 'transparent', border: 'none',
                    fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
                    fontSize: '11.5px', lineHeight: 1.6, color: '#e2e0f0',
                    whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflow: 'visible',
                  }}
                  dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(highlightCode(generatedCode)) }}
                />
              </>
            )}
          </div>
        )}
      </div>

      {/* ═══ Footer action bar ═══ */}
      {captured && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          <ActionBtn
            id="ai"
            icon={<Wand2 size={13} />}
            label="Recreate with AI"
            onClick={recreateWithAI}
            accent
            disabled={loading}
          />
          {generatedCode && (
            <ActionBtn
              id="preview-gen"
              icon={<Eye size={13} />}
              label="Preview"
              onClick={previewCode}
            />
          )}
          <ActionBtn
            id="copy-html"
            icon={<Copy size={13} />}
            label="Copy HTML"
            onClick={() => copyText(captured.html, 'HTML')}
          />
          <ActionBtn
            id="copy-css"
            icon={<Copy size={13} />}
            label="Copy CSS"
            onClick={() => copyText(cssMapToString(captured.css), 'CSS')}
          />
          {generatedCode && (
            <ActionBtn
              id="copy-gen"
              icon={<Copy size={13} />}
              label="Copy AI Code"
              onClick={() => copyText(generatedCode, 'AI code')}
            />
          )}
        </div>
      )}
    </div>
  );
}
