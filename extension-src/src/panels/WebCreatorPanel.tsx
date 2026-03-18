import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Copy, Download, Maximize2, Minimize2, Code2, Eye,
  Monitor, Tablet, Smartphone, Globe, Layout, Sparkles,
  Send, Trash2, RotateCcw, Upload, ChevronRight, User, Bot,
} from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';

/* ─── Types ─── */
type ViewMode = 'preview' | 'code';
type DeviceMode = 'desktop' | 'tablet' | 'mobile';

interface ChatMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
}

interface Template {
  label: string;
  icon: React.ReactNode;
  prompt: string;
  color: string;
}

/* ─── Constants ─── */
const TEMPLATES: Template[] = [
  {
    label: 'Landing Page',
    icon: <Globe size={16} />,
    prompt: 'Create a modern SaaS landing page with a hero section featuring a bold headline, subtitle, and CTA button. Include sections for features (3 cards with icons), testimonials, pricing tiers, and a footer with links. Use a gradient purple/blue color scheme.',
    color: '#7c3aed',
  },
  {
    label: 'Portfolio',
    icon: <User size={16} />,
    prompt: 'Create a personal portfolio website with a hero section with name and role, an about section, a project gallery with 4 cards (image placeholder, title, description), a skills section with progress bars, and a contact form. Use a dark minimal theme with accent colors.',
    color: '#06b6d4',
  },
  {
    label: 'Blog',
    icon: <Layout size={16} />,
    prompt: 'Create a blog homepage with a header/nav, a featured post hero with large image placeholder, a grid of 6 blog post cards (thumbnail, date, title, excerpt, read more link), sidebar with categories and recent posts, and a newsletter signup. Use clean typography.',
    color: '#10b981',
  },
  {
    label: 'Dashboard',
    icon: <Monitor size={16} />,
    prompt: 'Create an analytics dashboard with a sidebar nav, top stats row (4 metric cards with icons and numbers), a large area chart placeholder, a table of recent transactions (5 rows), and a donut chart. Use a dark theme with card-based layout and subtle shadows.',
    color: '#f59e0b',
  },
  {
    label: 'Login Page',
    icon: <User size={16} />,
    prompt: 'Create a beautiful login page with a split layout — left side has a gradient background with branding/illustration, right side has a centered login form with email, password fields, "Remember me" checkbox, login button, "Forgot password?" link, and "Sign up" link. Add social login buttons (Google, GitHub).',
    color: '#ec4899',
  },
  {
    label: 'Pricing Page',
    icon: <Sparkles size={16} />,
    prompt: 'Create a pricing page with 3 tiers (Basic, Pro, Enterprise) in a card layout. The middle card should be highlighted/recommended. Each card has plan name, price, feature list with checkmarks, and a CTA button. Include a monthly/annual toggle switch and an FAQ section below. Use a clean modern style.',
    color: '#8b5cf6',
  },
  {
    label: '404 Page',
    icon: <RotateCcw size={16} />,
    prompt: 'Create a creative 404 error page with a large "404" display using CSS art or animation, a witty message like "Oops! This page got lost in space", a search bar, a "Go Home" button, and some floating animated elements. Make it fun and memorable with smooth CSS animations.',
    color: '#ef4444',
  },
];

const SYSTEM_PROMPT = `You are an expert web designer and developer. Generate a complete, beautiful HTML page with inline CSS and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag inside <head>
- Include ALL JavaScript in a <script> tag before </body>
- Use modern CSS: flexbox, grid, custom properties, smooth transitions
- Use clean typography with system fonts or Google Fonts via CDN
- Make it fully responsive
- Use professional color schemes with proper contrast
- Add subtle animations and hover effects
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications to a previous page, return the COMPLETE updated HTML (not a diff)`;

const DEVICE_WIDTHS: Record<DeviceMode, string> = {
  desktop: '100%',
  tablet: '768px',
  mobile: '375px',
};

/* ─── Helpers ─── */
function stripFences(s: string): string {
  return s.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
}

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/* ─── Shared styles ─── */
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

/* ═══════════════════════════════════════════════════════════════════════════
   WebCreatorPanel
   ═══════════════════════════════════════════════════════════════════════════ */
export default function WebCreatorPanel() {
  const { getModel } = useStore();

  /* ─── State ─── */
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentHtml, setCurrentHtml] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [deviceMode, setDeviceMode] = useState<DeviceMode>('desktop');
  const [fullscreen, setFullscreen] = useState(false);
  const [status, setStatus] = useState('');
  const [iframeError, setIframeError] = useState<string | null>(null);
  const [hoveredBtn, setHoveredBtn] = useState<string | null>(null);
  const [chatOpen, setChatOpen] = useState(true);
  const [conversationHistory, setConversationHistory] = useState<Array<{ role: string; content: string }>>([]);

  /* ─── Refs ─── */
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* ─── Cleanup ─── */
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
    };
  }, []);

  /* ─── iframe error listener ─── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.source !== iframeRef.current?.contentWindow) return;
      if (e.data?.type === 'artifact-error') {
        const msg = e.data.msg || 'Unknown error';
        const line = e.data.line ? ` (line ${e.data.line})` : '';
        setIframeError(`${msg}${line}`);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  /* ─── Auto-scroll chat ─── */
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages]);

  /* ─── Update preview ─── */
  const updatePreview = useCallback((html: string) => {
    if (!iframeRef.current || !html) return;
    setIframeError(null);
    const errorScript = `<script>
window.onerror = function(msg, src, line, col, err) {
  parent.postMessage({ type: 'artifact-error', msg: String(msg), line: line, col: col, stack: err ? err.stack : '' }, '*');
};
window.addEventListener('unhandledrejection', function(e) {
  parent.postMessage({ type: 'artifact-error', msg: String(e.reason), line: 0 }, '*');
});
</script>`;
    let srcdoc = html;
    if (html.includes('</head>')) {
      srcdoc = html.replace('</head>', errorScript + '</head>');
    } else {
      srcdoc = errorScript + html;
    }
    iframeRef.current.srcdoc = srcdoc;
  }, []);

  /* ─── Extract HTML from AI response ─── */
  const extractHtml = useCallback((text: string): string => {
    let cleaned = stripFences(text);
    // If the response contains HTML but also has explanatory text, extract just the HTML
    const docTypeMatch = cleaned.match(/(<!DOCTYPE html[\s\S]*)/i);
    if (docTypeMatch) {
      cleaned = docTypeMatch[1];
    } else {
      const htmlMatch = cleaned.match(/(<html[\s\S]*<\/html>)/i);
      if (htmlMatch) {
        cleaned = htmlMatch[1];
      }
    }
    return cleaned;
  }, []);

  /* ─── Generate / iterate ─── */
  const sendMessage = useCallback(async (overrideText?: string) => {
    const text = (overrideText ?? input).trim();
    if (!text || loading) return;

    // Add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setStatus('Generating...');
    setIframeError(null);

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    // Build conversation for context
    const newHistory = [
      ...conversationHistory,
      { role: 'user', content: text },
    ];

    // Build the message with system prompt + full conversation
    const contextMessage = currentHtml
      ? `${SYSTEM_PROMPT}\n\nCurrent HTML page code:\n\`\`\`html\n${currentHtml}\n\`\`\`\n\nUser request: ${text}`
      : `${SYSTEM_PROMPT}\n\nUser request: ${text}`;

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          message: contextMessage,
          model: getModel('webcreator') || undefined,
        }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        const errMsg = (d as any).detail || resp.statusText;
        setStatus(errMsg);
        setLoading(false);
        return;
      }

      const data = await resp.json();
      const responseText = data.response || data.text || data.content || data.reply || data.message || '';
      const html = extractHtml(responseText);

      if (html) {
        setCurrentHtml(html);
        updatePreview(html);
        setViewMode('preview');
      }

      // Add AI message (show a brief summary, not the full HTML)
      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'ai',
        text: html ? 'Page updated. Preview is live below.' : responseText,
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, aiMsg]);

      // Update conversation history
      setConversationHistory([
        ...newHistory,
        { role: 'assistant', content: responseText },
      ]);

      setStatus('');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setStatus(err.message || 'Request failed');
        const errMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'ai',
          text: `Error: ${err.message || 'Request failed'}`,
          timestamp: Date.now(),
        };
        setMessages(prev => [...prev, errMsg]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [input, loading, currentHtml, conversationHistory, getModel, extractHtml, updatePreview]);

  /* ─── Template click ─── */
  const handleTemplate = useCallback((t: Template) => {
    sendMessage(t.prompt);
  }, [sendMessage]);

  /* ─── Actions ─── */
  const copyCode = useCallback(() => {
    if (!currentHtml) return;
    navigator.clipboard.writeText(currentHtml).then(() => {
      setStatus('Copied!');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 1500);
    });
  }, [currentHtml]);

  const downloadFile = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'website.html';
    a.click();
    URL.revokeObjectURL(a.href);
  }, [currentHtml]);

  const sendToCli = useCallback(async () => {
    if (!currentHtml) return;
    try {
      await fetch(`${HTTP}/api/feed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          type: 'webcreator',
          content: currentHtml,
          title: 'Web Creator export',
        }),
      });
      setStatus('Sent to CLI feed');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
    } catch {
      setStatus('Failed to send');
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      statusTimerRef.current = setTimeout(() => setStatus(''), 2000);
    }
  }, [currentHtml]);

  const clearConversation = useCallback(() => {
    setMessages([]);
    setCurrentHtml('');
    setConversationHistory([]);
    setStatus('');
    setIframeError(null);
    if (iframeRef.current) iframeRef.current.srcdoc = '';
  }, []);

  /* ─── Syntax highlighting ─── */
  const highlightedCode = React.useMemo(() => {
    if (!currentHtml) return '';
    let h = escHtml(currentHtml);
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
  }, [currentHtml]);

  /* ─── Button helper ─── */
  const ActionBtn = ({ id, icon, label, onClick }: {
    id: string; icon: React.ReactNode; label: string; onClick: () => void;
  }) => (
    <button
      onClick={onClick}
      onMouseEnter={() => setHoveredBtn(id)}
      onMouseLeave={() => setHoveredBtn(null)}
      style={{
        ...btnBase,
        ...(hoveredBtn === id ? btnHover : {}),
      }}
    >
      {icon} {label}
    </button>
  );

  /* ─── Device mode tabs ─── */
  const deviceTabs: { mode: DeviceMode; icon: React.ReactNode; label: string }[] = [
    { mode: 'desktop', icon: <Monitor size={13} />, label: 'Desktop' },
    { mode: 'tablet', icon: <Tablet size={13} />, label: 'Tablet' },
    { mode: 'mobile', icon: <Smartphone size={13} />, label: 'Mobile' },
  ];

  /* ─── Panel layout ─── */
  const panelStyle: React.CSSProperties = fullscreen
    ? { position: 'fixed', inset: 0, zIndex: 9999, background: 'var(--bg)', display: 'flex', flexDirection: 'column' }
    : { display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' };

  const hasHtml = !!currentHtml;
  const isEmptyState = messages.length === 0 && !hasHtml;

  return (
    <div style={panelStyle}>
      {/* ═══ Top bar ═══ */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px', flexShrink: 0,
        borderBottom: '1px solid var(--b1)',
      }}>
        <Layout size={15} style={{ color: 'var(--pl)', flexShrink: 0 }} />
        <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--tx)', letterSpacing: '0.02em' }}>
          Web Creator
        </span>

        {/* Device mode toggles (visible when we have content) */}
        {hasHtml && viewMode === 'preview' && (
          <div style={{
            display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
            border: '1px solid var(--b1)', overflow: 'hidden', marginLeft: 4,
          }}>
            {deviceTabs.map(d => (
              <button
                key={d.mode}
                onClick={() => setDeviceMode(d.mode)}
                title={d.label}
                style={{
                  display: 'flex', alignItems: 'center', gap: 3,
                  padding: '3px 8px', border: 'none', cursor: 'pointer',
                  fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                  background: deviceMode === d.mode ? 'var(--pg)' : 'transparent',
                  color: deviceMode === d.mode ? 'var(--pl)' : 'var(--mu)',
                  transition: 'all 0.15s ease',
                }}
              >
                {d.icon}
              </button>
            ))}
          </div>
        )}

        {/* View mode: Preview / Code */}
        {hasHtml && (
          <div style={{
            display: 'flex', background: 'var(--s1)', borderRadius: 'var(--r-pill)',
            border: '1px solid var(--b1)', overflow: 'hidden', marginLeft: 4,
          }}>
            <button
              onClick={() => setViewMode('preview')}
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: viewMode === 'preview' ? 'var(--pg)' : 'transparent',
                color: viewMode === 'preview' ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s ease',
              }}
            >
              <Eye size={11} /> Preview
            </button>
            <button
              onClick={() => setViewMode('code')}
              style={{
                display: 'flex', alignItems: 'center', gap: 3,
                padding: '3px 8px', border: 'none', cursor: 'pointer',
                fontSize: '10.5px', fontFamily: 'inherit', fontWeight: 500,
                background: viewMode === 'code' ? 'var(--pg)' : 'transparent',
                color: viewMode === 'code' ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s ease',
              }}
            >
              <Code2 size={11} /> Code
            </button>
          </div>
        )}

        <div style={{ flex: 1 }} />

        {fullscreen && (
          <button onClick={() => setFullscreen(false)} style={{ ...btnBase, padding: '4px 8px' }}>
            <Minimize2 size={13} /> Exit
          </button>
        )}
        <ModelPill featureKey="webcreator" />
      </div>

      {/* ═══ Status bar ═══ */}
      {(status || iframeError) && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '5px 12px', fontSize: '11px', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
          background: iframeError ? 'rgba(239,68,68,0.06)' : status === 'Copied!' || status === 'Sent to CLI feed' ? 'rgba(16,185,129,0.06)' : 'rgba(124,58,237,0.04)',
        }}>
          <span style={{
            color: iframeError ? 'var(--rd)' : status === 'Copied!' || status === 'Sent to CLI feed' ? 'var(--gr)' : 'var(--pl)',
            flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {iframeError ? `Error: ${iframeError}` : status}
          </span>
        </div>
      )}

      {/* ═══ Main content: chat + preview ═══ */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* ── Empty state with templates ── */}
        {isEmptyState && (
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center', gap: 20, padding: 24,
            overflow: 'auto',
          }}>
            <div style={{
              width: 56, height: 56, borderRadius: '50%',
              background: 'var(--pg)', border: '1px solid rgba(124,58,237,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Layout size={24} style={{ color: 'var(--pl)' }} />
            </div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
                Build websites with AI
              </div>
              <div style={{ fontSize: '11.5px', color: 'var(--mu)', maxWidth: 280, lineHeight: 1.5 }}>
                Describe what you want, iterate with natural language. Pick a template or start from scratch.
              </div>
            </div>

            {/* Template grid */}
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
              gap: 8, width: '100%', maxWidth: 400,
            }}>
              {TEMPLATES.map(t => (
                <button
                  key={t.label}
                  onClick={() => handleTemplate(t)}
                  onMouseEnter={() => setHoveredBtn(`tmpl-${t.label}`)}
                  onMouseLeave={() => setHoveredBtn(null)}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center',
                    gap: 6, padding: '14px 10px',
                    background: hoveredBtn === `tmpl-${t.label}` ? 'var(--s2)' : 'var(--s1)',
                    border: '1px solid var(--b1)',
                    borderColor: hoveredBtn === `tmpl-${t.label}` ? `${t.color}40` : 'var(--b1)',
                    borderRadius: 'var(--r-md)',
                    cursor: 'pointer', transition: 'all 0.2s ease',
                    fontFamily: 'inherit',
                  }}
                >
                  <div style={{
                    width: 36, height: 36, borderRadius: '50%',
                    background: `${t.color}15`,
                    border: `1px solid ${t.color}25`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    color: t.color,
                    transition: 'all 0.2s ease',
                    transform: hoveredBtn === `tmpl-${t.label}` ? 'scale(1.1)' : 'scale(1)',
                  }}>
                    {t.icon}
                  </div>
                  <span style={{
                    fontSize: '10.5px', fontWeight: 500,
                    color: hoveredBtn === `tmpl-${t.label}` ? 'var(--tx)' : 'var(--mu)',
                    transition: 'color 0.15s ease',
                  }}>
                    {t.label}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── Content area (chat messages + preview) ── */}
        {!isEmptyState && (
          <>
            {/* Chat messages (collapsible) */}
            {chatOpen && messages.length > 0 && (
              <div
                ref={chatScrollRef}
                style={{
                  maxHeight: hasHtml ? 160 : '40%', overflow: 'auto', flexShrink: 0,
                  borderBottom: '1px solid var(--b1)',
                  padding: '8px 12px',
                }}
              >
                {messages.map(m => (
                  <div
                    key={m.id}
                    style={{
                      display: 'flex', gap: 8, marginBottom: 8,
                      alignItems: 'flex-start',
                    }}
                  >
                    <div style={{
                      width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                      background: m.role === 'user' ? 'var(--pg)' : 'rgba(16,185,129,0.08)',
                      border: `1px solid ${m.role === 'user' ? 'rgba(124,58,237,0.2)' : 'rgba(16,185,129,0.15)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      marginTop: 1,
                    }}>
                      {m.role === 'user'
                        ? <User size={11} style={{ color: 'var(--pl)' }} />
                        : <Bot size={11} style={{ color: '#10b981' }} />
                      }
                    </div>
                    <div style={{
                      fontSize: '11.5px', lineHeight: 1.5,
                      color: m.role === 'user' ? 'var(--tx)' : 'var(--mu)',
                      flex: 1, minWidth: 0,
                      wordBreak: 'break-word',
                    }}>
                      {m.text.length > 200 ? m.text.slice(0, 200) + '...' : m.text}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
                    <div style={{
                      width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                      background: 'rgba(16,185,129,0.08)',
                      border: '1px solid rgba(16,185,129,0.15)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <Bot size={11} style={{ color: '#10b981' }} />
                    </div>
                    <div className="aura-thinking" style={{ transform: 'scale(0.7)', transformOrigin: 'left center' }}>
                      <span /><span /><span />
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Preview / Code area */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              {/* Preview iframe */}
              {hasHtml && viewMode === 'preview' && (
                <div style={{
                  width: '100%', height: '100%',
                  display: 'flex', justifyContent: 'center',
                  background: deviceMode !== 'desktop' ? 'var(--s1)' : 'transparent',
                  overflow: 'hidden',
                }}>
                  <div style={{
                    width: DEVICE_WIDTHS[deviceMode],
                    maxWidth: '100%',
                    height: '100%',
                    position: 'relative',
                    transition: 'width 0.3s ease',
                    ...(deviceMode !== 'desktop' ? {
                      border: '1px solid var(--b1)',
                      borderRadius: '8px',
                      overflow: 'hidden',
                      margin: '8px 0',
                      boxShadow: '0 4px 24px rgba(0,0,0,0.3)',
                    } : {}),
                  }}>
                    <iframe
                      ref={iframeRef}
                      sandbox="allow-scripts"
                      style={{
                        width: '100%', height: '100%', border: 'none',
                        background: '#fff',
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Code view */}
              {hasHtml && viewMode === 'code' && (
                <div style={{ width: '100%', height: '100%', overflow: 'auto', background: '#0d0d14' }}>
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
                      {currentHtml.length.toLocaleString()} chars
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

              {/* Loading overlay on preview */}
              {loading && hasHtml && viewMode === 'preview' && (
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex',
                  flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12,
                  background: 'rgba(3,3,3,0.5)', backdropFilter: 'blur(4px)',
                  zIndex: 5,
                }}>
                  <div className="aura-thinking">
                    <span /><span /><span />
                  </div>
                  <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                    Updating page...
                  </span>
                </div>
              )}

              {/* Loading state when no html yet */}
              {loading && !hasHtml && (
                <div style={{
                  position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', gap: 12,
                }}>
                  <div className="aura-thinking">
                    <span /><span /><span />
                  </div>
                  <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>
                    Creating your website...
                  </span>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* ═══ Chat input (always visible when not in empty state, or always) ═══ */}
      <div style={{
        padding: '8px 12px', flexShrink: 0,
        borderTop: '1px solid var(--b1)',
        display: 'flex', gap: 8, alignItems: 'flex-end',
      }}>
        {/* Quick actions when we have content */}
        {hasHtml && (
          <div style={{ display: 'flex', gap: 4, flexShrink: 0, alignItems: 'center' }}>
            <button
              onClick={clearConversation}
              onMouseEnter={() => setHoveredBtn('clear')}
              onMouseLeave={() => setHoveredBtn(null)}
              title="New page"
              style={{
                ...btnBase, padding: '6px 7px',
                ...(hoveredBtn === 'clear' ? { background: 'rgba(239,68,68,0.1)', borderColor: 'rgba(239,68,68,0.2)', color: 'var(--rd)' } : {}),
              }}
            >
              <Trash2 size={13} />
            </button>
          </div>
        )}

        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={hasHtml ? 'Describe changes... "Make the header sticky", "Add a contact form"' : 'Describe the website you want...'}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
          rows={1}
          style={{
            flex: 1, background: 'var(--s2)', border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
            padding: '8px 10px', resize: 'none', minHeight: 36, maxHeight: 80,
            outline: 'none', fontFamily: 'inherit', lineHeight: 1.5,
            transition: 'border-color 0.2s ease',
          }}
          onFocus={e => { e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)'; }}
          onBlur={e => { e.currentTarget.style.borderColor = 'var(--b1)'; }}
          onInput={e => {
            const el = e.currentTarget;
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 80) + 'px';
          }}
        />
        <button
          onClick={() => sendMessage()}
          disabled={loading || !input.trim()}
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: loading ? 'var(--s3)' : 'var(--p)',
            border: 'none', borderRadius: 'var(--r-md)', color: '#fff',
            padding: '0 14px', cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
            fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
            alignSelf: 'stretch', minWidth: 40, minHeight: 36,
            opacity: !input.trim() && !loading ? 0.5 : 1,
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
            <Send size={14} />
          )}
        </button>
      </div>

      {/* ═══ Footer action bar ═══ */}
      {hasHtml && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '6px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          <ActionBtn id="copy" icon={<Copy size={13} />} label="Copy HTML" onClick={copyCode} />
          <ActionBtn id="download" icon={<Download size={13} />} label="Download" onClick={downloadFile} />
          <ActionBtn id="sendcli" icon={<Upload size={13} />} label="Send to CLI" onClick={sendToCli} />
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
