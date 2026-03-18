import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Crosshair, Copy, Eye, Code2, Wand2, Sparkles, X,
  Image as ImageIcon, Layers, SplitSquareHorizontal,
  Globe, Send, Palette, Type, Check, RefreshCw, Trash2, FileText,
} from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
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

interface FullPageCapture {
  html: string;
  css: string;
  css_map: Record<string, Record<string, string>>;
  screenshot_b64: string;
  colors: string[];
  fonts: string[];
  metadata: {
    title: string;
    description: string;
    og_image: string;
    og_title: string;
    og_description: string;
    og_type: string;
    og_site_name: string;
    favicon: string;
  };
  source_url: string;
  viewport: { width: number; height: number };
  asset_urls: { images: string[]; stylesheets: string[] };
  responsive_info: { viewport_width: number; media_queries: string[] };
  element_count: number;
  timestamp: number;
}

type CaptureMode = 'component' | 'page';
type ViewTab = 'preview' | 'html' | 'css' | 'generated';
type PageViewTab = 'preview' | 'html' | 'colors' | 'fonts' | 'assets';

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

/* ─── Badge component ─── */
function Badge({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <span style={{
      fontSize: '9.5px', fontWeight: 600, letterSpacing: '0.06em',
      textTransform: 'uppercase',
      color: accent ? '#a78bfa' : 'var(--mu)',
      background: accent ? 'rgba(167,139,250,0.1)' : 'var(--s2)',
      padding: '2px 8px', borderRadius: 3,
      display: 'inline-block',
    }}>
      {children}
    </span>
  );
}

/* ─── Color swatch ─── */
function ColorSwatch({ color }: { color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 18, height: 18, borderRadius: 3, flexShrink: 0,
        background: color,
        border: '1px solid rgba(255,255,255,0.1)',
      }} />
      <span style={{ fontSize: '11px', color: 'var(--mu)', fontFamily: 'monospace' }}>
        {color}
      </span>
    </div>
  );
}

/* ─── Info row ─── */
function InfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{
      background: 'var(--s2)', border: '1px solid var(--b1)',
      borderRadius: 'var(--r-sm)', padding: '6px 10px',
      fontSize: '11px', color: 'var(--mu)',
    }}>
      <span style={{ color: 'var(--pl)', fontWeight: 600 }}>{label}: </span>
      {value}
    </div>
  );
}


/* ─── Component ─── */
export default function CapturePanel() {
  const { getModel } = useStore();

  const [captureMode, setCaptureMode] = useState<CaptureMode>('component');
  const [capturing, setCapturing] = useState(false);
  const [captured, setCaptured] = useState<CapturedComponent | null>(null);
  const [pageCaptured, setPageCaptured] = useState<FullPageCapture | null>(null);
  const [activeTab, setActiveTab] = useState<ViewTab>('preview');
  const [pageTab, setPageTab] = useState<PageViewTab>('preview');
  const [generatedCode, setGeneratedCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);
  const [previewSrcdoc, setPreviewSrcdoc] = useState('');
  const [sendingToCli, setSendingToCli] = useState(false);
  const [pageCapturing, setPageCapturing] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flash = useCallback((msg: string, dur = 2000) => {
    setStatus(msg);
    if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    statusTimerRef.current = setTimeout(() => setStatus(''), dur);
  }, []);

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
        setCaptureMode('component');
        setActiveTab('preview');
        setGeneratedCode('');
        setPreviewSrcdoc('');
        flash('Component captured!');
      }
      if (msg.type === 'CAPTURE_MODE_EXITED') {
        setCapturing(false);
      }
      if (msg.type === 'FULL_PAGE_CAPTURED') {
        setPageCaptured(msg.data);
        setPageCapturing(false);
        setCaptureMode('page');
        setPageTab('preview');
        flash('Full page captured!');
      }
      if (msg.type === 'FULL_PAGE_CAPTURE_ERROR') {
        setPageCapturing(false);
        flash(msg.error || 'Full page capture failed');
      }
    };
    ext?.runtime?.onMessage?.addListener(handler);

    // Also listen for forwarded custom events from App.tsx
    const onCaptured = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail?.data) {
        setCaptured(detail.data);
        setCapturing(false);
        setCaptureMode('component');
        setActiveTab('preview');
        setGeneratedCode('');
        setPreviewSrcdoc('');
        flash('Component captured!');
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
  }, [flash]);

  /* ─── Start / Stop component capture mode ─── */
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

  /* ─── Full page capture ─── */
  const captureFullPage = useCallback(() => {
    setPageCapturing(true);
    setStatus('Capturing full page...');
    ext?.runtime?.sendMessage({ type: 'FULL_PAGE_CAPTURE' });
  }, []);

  /* ─── Send to CLI ─── */
  const sendToCli = useCallback(async () => {
    setSendingToCli(true);
    setStatus('Sending to CLI...');

    try {
      let payload: any;

      if (captureMode === 'component' && captured) {
        payload = {
          type: 'component',
          html: captured.html,
          css_map: captured.css,
          css: cssMapToString(captured.css),
          screenshot_b64: captured.screenshot_b64,
          colors: [],
          fonts: [],
          metadata: { title: '', description: '', og_image: '', og_title: '', og_description: '', og_type: '', og_site_name: '', favicon: '' },
          source_url: '',
          tag_name: captured.tagName,
          class_name: captured.className,
          dimensions: captured.dimensions,
          text_content: captured.textContent,
          element_count: Object.keys(captured.css).length,
          timestamp: Date.now() / 1000,
        };
      } else if (captureMode === 'page' && pageCaptured) {
        payload = {
          type: 'page',
          html: pageCaptured.html,
          css: pageCaptured.css,
          css_map: pageCaptured.css_map,
          screenshot_b64: pageCaptured.screenshot_b64,
          colors: pageCaptured.colors,
          fonts: pageCaptured.fonts,
          metadata: pageCaptured.metadata,
          source_url: pageCaptured.source_url,
          viewport: pageCaptured.viewport,
          asset_urls: pageCaptured.asset_urls,
          responsive_info: pageCaptured.responsive_info,
          element_count: pageCaptured.element_count,
          timestamp: pageCaptured.timestamp,
        };
      } else {
        flash('Nothing to send');
        setSendingToCli(false);
        return;
      }

      const resp = await fetch(`${HTTP}/api/feed/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        throw new Error((d as any).detail || `HTTP ${resp.status}`);
      }

      const result = await resp.json();
      flash(`Sent to CLI! (${(result.size_bytes / 1024).toFixed(1)} KB)`);
    } catch (err: any) {
      flash(err.message || 'Send failed');
    } finally {
      setSendingToCli(false);
    }
  }, [captureMode, captured, pageCaptured, flash]);

  /* ─── Generate code with AI ─── */
  const recreateWithAI = useCallback(async () => {
    if (!captured && !pageCaptured) return;
    setLoading(true);
    setStatus('Generating code...');
    setGeneratedCode('');
    if (captureMode === 'component') setActiveTab('generated');

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let prompt: string;

    if (captureMode === 'component' && captured) {
      const cssText = cssMapToString(captured.css);
      prompt = `You are an expert frontend developer. Recreate this UI component using React + Tailwind CSS.

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
    } else if (captureMode === 'page' && pageCaptured) {
      prompt = `You are an expert frontend developer. Analyze and recreate this full page design.

Page: ${pageCaptured.metadata.title || pageCaptured.source_url}
Viewport: ${pageCaptured.viewport.width}x${pageCaptured.viewport.height}
Colors: ${pageCaptured.colors.slice(0, 15).join(', ')}
Fonts: ${pageCaptured.fonts.slice(0, 8).join(', ')}

HTML structure (abbreviated):
\`\`\`html
${pageCaptured.html.slice(0, 12000)}
\`\`\`

Key styles:
\`\`\`css
${pageCaptured.css.slice(0, 6000)}
\`\`\`

Recreate the page layout and visual design using React + Tailwind CSS. Focus on capturing the layout structure, color scheme, typography, and spacing. Return ONLY a single React functional component. No imports, no exports — just the function rendered with ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App)).`;
    } else {
      setLoading(false);
      return;
    }

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
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
      flash('Code generated!');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [captured, pageCaptured, captureMode, getModel, flash]);

  /* ─── AI Design Review ─── */
  const reviewDesign = useCallback(async () => {
    if (!pageCaptured) return;
    setLoading(true);
    setStatus('Reviewing design...');
    setGeneratedCode('');

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const prompt = `You are a senior UI/UX design reviewer. Analyze this captured website and provide a concise design review.

Page: ${pageCaptured.metadata.title || pageCaptured.source_url}
URL: ${pageCaptured.source_url}
Viewport: ${pageCaptured.viewport.width}x${pageCaptured.viewport.height}
Colors: ${pageCaptured.colors.slice(0, 15).join(', ')}
Fonts: ${pageCaptured.fonts.slice(0, 8).join(', ')}
Elements: ${pageCaptured.element_count}

HTML structure:
\`\`\`html
${pageCaptured.html.slice(0, 8000)}
\`\`\`

Provide a brief review covering:
1. Visual hierarchy & layout quality
2. Color palette assessment
3. Typography choices
4. Spacing & alignment consistency
5. Accessibility concerns
6. Top 3 specific improvement suggestions

Keep it concise and actionable.`;

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
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
      setGeneratedCode(responseText);
      flash('Review complete!');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Review failed');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [pageCaptured, getModel, flash]);

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
    if (captureMode === 'component') setActiveTab('preview');
  }, [generatedCode, captureMode]);

  /* ─── Copy helpers ─── */
  const copyText = useCallback((text: string, label: string) => {
    navigator.clipboard.writeText(text).then(() => {
      flash(`${label} copied!`);
    });
  }, [flash]);

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

  /* ─── Has anything captured? ─── */
  const hasCapture = captureMode === 'component' ? !!captured : !!pageCaptured;

  /* ─── Component tab definitions ─── */
  const componentTabs: { key: ViewTab; icon: React.ReactNode; label: string }[] = [
    { key: 'preview', icon: <Eye size={13} />, label: 'Preview' },
    { key: 'html', icon: <Code2 size={13} />, label: 'HTML' },
    { key: 'css', icon: <Layers size={13} />, label: 'CSS' },
    { key: 'generated', icon: <Sparkles size={13} />, label: 'AI Code' },
  ];

  /* ─── Page tab definitions ─── */
  const pageTabs: { key: PageViewTab; icon: React.ReactNode; label: string }[] = [
    { key: 'preview', icon: <Eye size={13} />, label: 'Preview' },
    { key: 'html', icon: <Code2 size={13} />, label: 'HTML' },
    { key: 'colors', icon: <Palette size={13} />, label: 'Colors' },
    { key: 'fonts', icon: <Type size={13} />, label: 'Fonts' },
    { key: 'assets', icon: <Layers size={13} />, label: 'Assets' },
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
          Capture
        </span>
        <div style={{ flex: 1 }} />
        <ModelPill featureKey="capture" />
      </div>

      {/* ═══ Mode tabs: Component | Full Page ═══ */}
      <div style={{
        display: 'flex', flexShrink: 0,
        borderBottom: '1px solid var(--b1)',
      }}>
        <button
          onClick={() => setCaptureMode('component')}
          style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
            padding: '7px 0', background: 'none', border: 'none',
            borderBottom: captureMode === 'component' ? '2px solid var(--p)' : '2px solid transparent',
            color: captureMode === 'component' ? 'var(--pl)' : 'var(--mu)',
            fontSize: '11.5px', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
            transition: 'all 0.15s ease',
          }}
        >
          <Crosshair size={13} /> Component
        </button>
        <button
          onClick={() => setCaptureMode('page')}
          style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
            padding: '7px 0', background: 'none', border: 'none',
            borderBottom: captureMode === 'page' ? '2px solid var(--p)' : '2px solid transparent',
            color: captureMode === 'page' ? 'var(--pl)' : 'var(--mu)',
            fontSize: '11.5px', fontWeight: 600, cursor: 'pointer', fontFamily: 'inherit',
            transition: 'all 0.15s ease',
          }}
        >
          <Globe size={13} /> Full Page
        </button>
      </div>

      {/* ═══ Capture controls ═══ */}
      <div style={{
        padding: '10px 12px', flexShrink: 0, borderBottom: '1px solid var(--b1)',
        display: 'flex', gap: 8, alignItems: 'center',
      }}>
        {captureMode === 'component' ? (
          <>
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
                Click any element
              </span>
            )}
          </>
        ) : (
          <>
            <button
              onClick={captureFullPage}
              disabled={pageCapturing}
              onMouseEnter={() => setHoveredBtn('page-cap')}
              onMouseLeave={() => setHoveredBtn(null)}
              style={{
                display: 'flex', alignItems: 'center', gap: 6,
                background: hoveredBtn === 'page-cap' && !pageCapturing ? 'var(--p)' : 'var(--pg)',
                border: '1px solid rgba(124,58,237,0.3)',
                borderRadius: 'var(--r-md)', color: 'var(--pl)',
                padding: '8px 16px', cursor: pageCapturing ? 'not-allowed' : 'pointer',
                fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
                transition: 'all 0.2s ease',
                opacity: pageCapturing ? 0.6 : 1,
                boxShadow: hoveredBtn === 'page-cap' && !pageCapturing ? '0 2px 10px rgba(124,58,237,0.3)' : 'none',
              }}
            >
              <Globe size={14} /> {pageCapturing ? 'Capturing...' : 'Capture Full Page'}
            </button>
            {pageCapturing && (
              <span style={{
                fontSize: '11px', color: 'var(--pl)',
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: 'var(--p)', boxShadow: '0 0 8px var(--p)',
                  animation: 'pulse 1.5s ease-in-out infinite',
                }} />
                Extracting page data...
              </span>
            )}
          </>
        )}
      </div>

      {/* ═══ Status bar ═══ */}
      {status && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: status.includes('failed') || status.includes('error') || status.includes('Error')
            ? 'rgba(239,68,68,0.06)'
            : status.includes('copied') || status.includes('captured') || status.includes('generated') || status.includes('Sent') || status.includes('complete')
              ? 'rgba(16,185,129,0.06)'
              : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: status.includes('failed') || status.includes('error') || status.includes('Error') ? 'var(--rd)'
              : status.includes('copied') || status.includes('captured') || status.includes('generated') || status.includes('Sent') || status.includes('complete') ? 'var(--gr)'
                : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {status}
          </span>
        </div>
      )}

      {/* ═══ COMPONENT MODE: Tab bar ═══ */}
      {captureMode === 'component' && captured && (
        <div style={{
          display: 'flex', alignItems: 'center', flexShrink: 0,
          borderBottom: '1px solid var(--b1)', padding: '0 8px',
        }}>
          {componentTabs.map(t => (
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

      {/* ═══ PAGE MODE: Tab bar ═══ */}
      {captureMode === 'page' && pageCaptured && (
        <div style={{
          display: 'flex', alignItems: 'center', flexShrink: 0,
          borderBottom: '1px solid var(--b1)', padding: '0 8px',
        }}>
          {pageTabs.map(t => (
            <button
              key={t.key}
              onClick={() => setPageTab(t.key)}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '7px 12px', background: 'none', border: 'none',
                borderBottom: pageTab === t.key ? '2px solid var(--p)' : '2px solid transparent',
                color: pageTab === t.key ? 'var(--pl)' : 'var(--mu)',
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

        {/* ── Empty state ── */}
        {!hasCapture && !capturing && !pageCapturing && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 16, padding: 24,
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--pg)', border: '1px solid rgba(124,58,237,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              {captureMode === 'component'
                ? <Crosshair size={24} style={{ color: 'var(--pl)' }} />
                : <Globe size={24} style={{ color: 'var(--pl)' }} />
              }
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                {captureMode === 'component' ? 'Capture any component' : 'Capture full page'}
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 260, lineHeight: 1.5 }}>
                {captureMode === 'component'
                  ? 'Hover over any UI element on any website, click to capture its DOM, styles, and screenshot. Then recreate it with AI.'
                  : 'Capture the entire page HTML, styles, color palette, font stack, and screenshot. Send it to the CLI for code generation.'
                }
              </div>
            </div>
          </div>
        )}

        {/* ── Capturing spinner ── */}
        {(capturing || pageCapturing) && !hasCapture && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
          }}>
            <div className="aura-thinking">
              <span /><span /><span />
            </div>
            <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
              {capturing ? 'Waiting for capture...' : 'Extracting page data...'}
            </span>
          </div>
        )}

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* ═══ COMPONENT MODE CONTENT ═══ */}
        {/* ═══════════════════════════════════════════════════════════════ */}

        {/* Preview tab */}
        {captureMode === 'component' && captured && activeTab === 'preview' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 0 }}>
            {captured.screenshot_b64 && (
              <div style={{ padding: 12 }}>
                <Badge accent>Screenshot</Badge>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', background: '#0d0d14', marginTop: 8,
                }}>
                  <img
                    src={`data:image/png;base64,${captured.screenshot_b64}`}
                    alt="Captured component"
                    style={{ width: '100%', display: 'block' }}
                  />
                </div>
              </div>
            )}
            <div style={{ padding: '0 12px 12px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <InfoRow label="Tag" value={captured.tagName} />
                <InfoRow label="Size" value={`${Math.round(captured.dimensions.width)} x ${Math.round(captured.dimensions.height)}`} />
                {captured.className && (
                  <div style={{
                    gridColumn: '1 / -1',
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '6px 10px',
                    fontSize: '11px', color: 'var(--mu)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    <span style={{ color: 'var(--pl)', fontWeight: 600 }}>Class: </span>
                    {captured.className}
                  </div>
                )}
              </div>
            </div>
            {previewSrcdoc && (
              <div style={{ padding: '0 12px 12px' }}>
                <Badge accent>AI Recreation</Badge>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', height: 300, marginTop: 8,
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
        {captureMode === 'component' && captured && activeTab === 'html' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
            <div style={{
              position: 'sticky', top: 0, zIndex: 2,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px',
              background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
              <Badge accent>HTML</Badge>
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
        {captureMode === 'component' && captured && activeTab === 'css' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
            <div style={{
              position: 'sticky', top: 0, zIndex: 2,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px',
              background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
              <Badge accent>CSS</Badge>
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
        {captureMode === 'component' && captured && activeTab === 'generated' && (
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
                  Generating code...
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
                  <Badge accent>React + Tailwind</Badge>
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

        {/* ═══════════════════════════════════════════════════════════════ */}
        {/* ═══ FULL PAGE MODE CONTENT ═══ */}
        {/* ═══════════════════════════════════════════════════════════════ */}

        {/* Page Preview tab */}
        {captureMode === 'page' && pageCaptured && pageTab === 'preview' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 0 }}>
            {/* Screenshot */}
            {pageCaptured.screenshot_b64 && (
              <div style={{ padding: 12 }}>
                <Badge accent>Screenshot</Badge>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', background: '#0d0d14', marginTop: 8,
                }}>
                  <img
                    src={`data:image/png;base64,${pageCaptured.screenshot_b64}`}
                    alt="Full page capture"
                    style={{ width: '100%', display: 'block' }}
                  />
                </div>
              </div>
            )}
            {/* Page info */}
            <div style={{ padding: '0 12px 12px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {pageCaptured.metadata.title && (
                  <div style={{
                    gridColumn: '1 / -1',
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '6px 10px',
                    fontSize: '11px', color: 'var(--mu)',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  }}>
                    <span style={{ color: 'var(--pl)', fontWeight: 600 }}>Title: </span>
                    {pageCaptured.metadata.title}
                  </div>
                )}
                <InfoRow label="Viewport" value={`${pageCaptured.viewport.width} x ${pageCaptured.viewport.height}`} />
                <InfoRow label="Elements" value={pageCaptured.element_count} />
                <InfoRow label="Colors" value={pageCaptured.colors.length} />
                <InfoRow label="Fonts" value={pageCaptured.fonts.length} />
              </div>

              {/* Color palette swatches (top 8) */}
              {pageCaptured.colors.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <Badge accent>Color Palette</Badge>
                  <div style={{
                    display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8,
                  }}>
                    {pageCaptured.colors.slice(0, 12).map((c, i) => (
                      <div key={i} style={{
                        width: 28, height: 28, borderRadius: 4,
                        background: c,
                        border: '1px solid rgba(255,255,255,0.1)',
                        cursor: 'pointer',
                      }} title={c} onClick={() => copyText(c, 'Color')} />
                    ))}
                  </div>
                </div>
              )}

              {/* Font list */}
              {pageCaptured.fonts.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <Badge accent>Fonts</Badge>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
                    {pageCaptured.fonts.slice(0, 6).map((f, i) => (
                      <span key={i} style={{
                        fontSize: '10.5px', padding: '3px 8px',
                        background: 'var(--s2)', border: '1px solid var(--b1)',
                        borderRadius: 3, color: 'var(--mu)',
                      }}>
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* AI output preview */}
            {previewSrcdoc && (
              <div style={{ padding: '0 12px 12px' }}>
                <Badge accent>AI Recreation</Badge>
                <div style={{
                  border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                  overflow: 'hidden', height: 300, marginTop: 8,
                }}>
                  <iframe
                    srcDoc={previewSrcdoc}
                    sandbox="allow-scripts"
                    style={{ width: '100%', height: '100%', border: 'none', background: '#0a0a0f' }}
                  />
                </div>
              </div>
            )}

            {/* AI Review output */}
            {generatedCode && !loading && !previewSrcdoc && (
              <div style={{ padding: '0 12px 12px' }}>
                <Badge accent>AI Review</Badge>
                <div style={{
                  marginTop: 8, padding: 12,
                  background: 'var(--s2)', border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  fontSize: '11.5px', lineHeight: 1.6, color: 'var(--mu)',
                  whiteSpace: 'pre-wrap', maxHeight: 400, overflow: 'auto',
                }}>
                  {generatedCode}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Page HTML tab */}
        {captureMode === 'page' && pageCaptured && pageTab === 'html' && (
          <div style={{ height: '100%', overflow: 'auto', background: '#0d0d14' }}>
            <div style={{
              position: 'sticky', top: 0, zIndex: 2,
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '6px 12px',
              background: 'rgba(13,13,20,0.95)', backdropFilter: 'blur(8px)',
              borderBottom: '1px solid rgba(255,255,255,0.04)',
            }}>
              <Badge accent>HTML</Badge>
              <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.3)', fontVariantNumeric: 'tabular-nums' }}>
                {pageCaptured.html.length.toLocaleString()} chars
              </span>
            </div>
            <pre
              style={{
                margin: 0, padding: '12px 14px', background: 'transparent', border: 'none',
                fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
                fontSize: '11.5px', lineHeight: 1.6, color: '#e2e0f0',
                whiteSpace: 'pre-wrap', wordBreak: 'break-word', overflow: 'visible',
              }}
              dangerouslySetInnerHTML={{
                __html: DOMPurify.sanitize(highlightCode(pageCaptured.html.slice(0, 50000)))
              }}
            />
          </div>
        )}

        {/* Page Colors tab */}
        {captureMode === 'page' && pageCaptured && pageTab === 'colors' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 12 }}>
            <Badge accent>Color Palette ({pageCaptured.colors.length} colors)</Badge>
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12,
            }}>
              {pageCaptured.colors.map((c, i) => (
                <div
                  key={i}
                  onClick={() => copyText(c, 'Color')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer',
                    padding: '6px 10px', borderRadius: 'var(--r-sm)',
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{
                    width: 28, height: 28, borderRadius: 4, flexShrink: 0,
                    background: c, border: '1px solid rgba(255,255,255,0.1)',
                  }} />
                  <span style={{ fontSize: '11.5px', color: 'var(--mu)', fontFamily: 'monospace' }}>
                    {c}
                  </span>
                  <div style={{ flex: 1 }} />
                  <Copy size={11} style={{ color: 'var(--mu)', opacity: 0.4 }} />
                </div>
              ))}
              {pageCaptured.colors.length === 0 && (
                <span style={{ fontSize: '12px', color: 'var(--mu)' }}>No colors detected</span>
              )}
            </div>
          </div>
        )}

        {/* Page Fonts tab */}
        {captureMode === 'page' && pageCaptured && pageTab === 'fonts' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 12 }}>
            <Badge accent>Font Stack ({pageCaptured.fonts.length} fonts)</Badge>
            <div style={{
              display: 'flex', flexDirection: 'column', gap: 6, marginTop: 12,
            }}>
              {pageCaptured.fonts.map((f, i) => (
                <div
                  key={i}
                  onClick={() => copyText(f, 'Font')}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer',
                    padding: '8px 12px', borderRadius: 'var(--r-sm)',
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                  }}
                >
                  <Type size={14} style={{ color: 'var(--pl)', flexShrink: 0 }} />
                  <span style={{ fontSize: '12px', color: 'var(--tx)', fontFamily: f }}>
                    {f}
                  </span>
                  <div style={{ flex: 1 }} />
                  <span style={{ fontSize: '10px', color: 'var(--mu)', fontFamily: f }}>
                    Aa Bb Cc 123
                  </span>
                </div>
              ))}
              {pageCaptured.fonts.length === 0 && (
                <span style={{ fontSize: '12px', color: 'var(--mu)' }}>No fonts detected</span>
              )}
            </div>
          </div>
        )}

        {/* Page Assets tab */}
        {captureMode === 'page' && pageCaptured && pageTab === 'assets' && (
          <div style={{ height: '100%', overflow: 'auto', padding: 12 }}>
            {/* Images */}
            <Badge accent>Images ({pageCaptured.asset_urls?.images?.length || 0})</Badge>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8, marginBottom: 16 }}>
              {(pageCaptured.asset_urls?.images || []).slice(0, 30).map((url, i) => (
                <div key={i} style={{
                  fontSize: '10.5px', color: 'var(--mu)', padding: '4px 8px',
                  background: 'var(--s2)', borderRadius: 3,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  border: '1px solid var(--b1)',
                }}>
                  {url}
                </div>
              ))}
              {(!pageCaptured.asset_urls?.images || pageCaptured.asset_urls.images.length === 0) && (
                <span style={{ fontSize: '11px', color: 'var(--mu)' }}>No images found</span>
              )}
            </div>

            {/* Stylesheets */}
            <Badge accent>Stylesheets ({pageCaptured.asset_urls?.stylesheets?.length || 0})</Badge>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 8 }}>
              {(pageCaptured.asset_urls?.stylesheets || []).map((url, i) => (
                <div key={i} style={{
                  fontSize: '10.5px', color: 'var(--mu)', padding: '4px 8px',
                  background: 'var(--s2)', borderRadius: 3,
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                  border: '1px solid var(--b1)',
                }}>
                  {url}
                </div>
              ))}
              {(!pageCaptured.asset_urls?.stylesheets || pageCaptured.asset_urls.stylesheets.length === 0) && (
                <span style={{ fontSize: '11px', color: 'var(--mu)' }}>No stylesheets found</span>
              )}
            </div>
          </div>
        )}

        {/* Loading overlay for page mode */}
        {captureMode === 'page' && loading && (
          <div style={{
            position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 12,
            background: 'rgba(10,10,15,0.85)', zIndex: 10,
          }}>
            <div className="aura-thinking">
              <span /><span /><span />
            </div>
            <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
              {status || 'Processing...'}
            </span>
          </div>
        )}
      </div>

      {/* ═══ Footer action bar ═══ */}
      {hasCapture && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          {/* Send to CLI — always visible */}
          <ActionBtn
            id="send-cli"
            icon={<Send size={13} />}
            label="Send to CLI"
            onClick={sendToCli}
            accent
            disabled={sendingToCli}
          />

          {/* Component-mode actions */}
          {captureMode === 'component' && captured && (
            <>
              <ActionBtn
                id="ai"
                icon={<Wand2 size={13} />}
                label="Recreate"
                onClick={recreateWithAI}
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
                label="HTML"
                onClick={() => copyText(captured.html, 'HTML')}
              />
              <ActionBtn
                id="copy-css"
                icon={<Copy size={13} />}
                label="CSS"
                onClick={() => copyText(cssMapToString(captured.css), 'CSS')}
              />
              {generatedCode && (
                <ActionBtn
                  id="copy-gen"
                  icon={<Copy size={13} />}
                  label="AI Code"
                  onClick={() => copyText(generatedCode, 'AI code')}
                />
              )}
            </>
          )}

          {/* Page-mode actions */}
          {captureMode === 'page' && pageCaptured && (
            <>
              <ActionBtn
                id="ai-page"
                icon={<Wand2 size={13} />}
                label="Recreate"
                onClick={recreateWithAI}
                disabled={loading}
              />
              <ActionBtn
                id="review"
                icon={<FileText size={13} />}
                label="Review"
                onClick={reviewDesign}
                disabled={loading}
              />
              {generatedCode && (
                <>
                  <ActionBtn
                    id="preview-page"
                    icon={<Eye size={13} />}
                    label="Preview"
                    onClick={previewCode}
                  />
                  <ActionBtn
                    id="copy-page-gen"
                    icon={<Copy size={13} />}
                    label="AI Code"
                    onClick={() => copyText(generatedCode, 'AI code')}
                  />
                </>
              )}
              <ActionBtn
                id="copy-page-html"
                icon={<Copy size={13} />}
                label="HTML"
                onClick={() => copyText(pageCaptured.html.slice(0, 100000), 'HTML')}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}
