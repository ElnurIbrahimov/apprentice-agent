import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  Copy, Download, Maximize2, Minimize2, Code2, Eye,
  Monitor, Tablet, Smartphone, Globe, Layout, Sparkles,
  Send, Trash2, RotateCcw, Upload, User, Bot,
  Undo2, Redo2, MousePointer2, Pencil, Palette, ExternalLink, Square,
} from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
import { streamRawGenerate } from '../utils/streamChat';
import { StreamingPreviewController } from '../utils/StreamingPreviewController';
import { useVersionHistory } from '../utils/useVersionHistory';
import { highlightCode } from '../utils/highlighter';
import OfflineBanner from '../components/OfflineBanner';

/* ─── Chrome storage helpers ─── */
declare const browser: any; // Firefox compat
const ext = typeof chrome !== 'undefined' ? chrome : typeof browser !== 'undefined' ? browser : null;
const WC_STORAGE_KEY = 'aura_webcreator_state';

function wcStorageGet(): Promise<{ html?: string; messages?: ChatMessage[]; history?: Array<{ role: string; content: string }> } | null> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) { resolve(null); return; }
    ext.storage.local.get([WC_STORAGE_KEY], (d: any) => {
      try { resolve(d?.[WC_STORAGE_KEY] ? JSON.parse(d[WC_STORAGE_KEY]) : null); }
      catch { resolve(null); }
    });
  });
}

function wcStorageSave(html: string, messages: ChatMessage[], history: Array<{ role: string; content: string }>) {
  if (!ext?.storage?.local) return;
  // Only save if there's actual content
  if (!html && messages.length === 0) {
    ext.storage.local.remove([WC_STORAGE_KEY]);
    return;
  }
  const data = JSON.stringify({ html, messages: messages.slice(-30), history: history.slice(-20) });
  ext.storage.local.set({ [WC_STORAGE_KEY]: data });
}

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

/** Inject error-handler + element-selection scripts into HTML for iframe srcdoc */
function injectIframeScripts(html: string): string {
  const script = `<script>
window.onerror = function(msg, src, line, col, err) {
  parent.postMessage({ type: 'artifact-error', msg: String(msg), line: line, col: col, stack: err ? err.stack : '' }, '*');
};
window.addEventListener('unhandledrejection', function(e) {
  parent.postMessage({ type: 'artifact-error', msg: String(e.reason), line: 0 }, '*');
});
var _auraSelectMode = false;
window.addEventListener('message', function(e) {
  if (e.data && e.data.type === 'toggle-select-mode') _auraSelectMode = e.data.enabled;
});
var _auraStyle = document.createElement('style');
_auraStyle.textContent = '.aura-highlight { outline: 2px solid #3b82f6 !important; outline-offset: 2px; cursor: crosshair !important; }';
document.head.appendChild(_auraStyle);
function _auraGetPath(el) {
  var parts = [];
  while (el && el !== document.body && el !== document.documentElement) {
    var tag = el.tagName.toLowerCase();
    if (el.id) { parts.unshift(tag + '#' + el.id); break; }
    else if (el.className && typeof el.className === 'string') { parts.unshift(tag + '.' + el.className.trim().split(/\\s+/).join('.')); }
    else { parts.unshift(tag); }
    el = el.parentElement;
  }
  return parts.join(' > ');
}
document.addEventListener('mouseover', function(e) {
  if (!_auraSelectMode) return;
  document.querySelectorAll('.aura-highlight').forEach(function(el) { el.classList.remove('aura-highlight'); });
  if (e.target !== document.body && e.target !== document.documentElement) e.target.classList.add('aura-highlight');
}, true);
document.addEventListener('mouseout', function(e) {
  if (!_auraSelectMode) return;
  e.target.classList.remove('aura-highlight');
}, true);
document.addEventListener('click', function(e) {
  if (!_auraSelectMode) return;
  e.preventDefault();
  e.stopPropagation();
  var el = e.target;
  el.classList.remove('aura-highlight');
  parent.postMessage({
    type: 'element-selected',
    tagName: el.tagName,
    classes: (typeof el.className === 'string') ? el.className : '',
    id: el.id || '',
    text: (el.textContent || '').slice(0, 100).trim(),
    outerHTML: el.outerHTML.slice(0, 500),
    path: _auraGetPath(el)
  }, '*');
}, true);
</script>`;
  if (html.includes('</head>')) {
    return html.replace('</head>', script + '</head>');
  }
  return script + html;
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

const exportItemStyle: React.CSSProperties = {
  display: 'block', width: '100%', textAlign: 'left', background: 'none',
  border: 'none', color: '#ccc', padding: '6px 12px', fontSize: '11px',
  cursor: 'pointer', fontFamily: 'inherit',
};

/* ─── ActionBtn (outside component to avoid re-creation each render) ─── */
function ActionBtn({ id, icon, label, onClick, hoveredBtn, setHoveredBtn }: {
  id: string; icon: React.ReactNode; label: string; onClick: () => void;
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
      }}
    >
      {icon} {label}
    </button>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   WebCreatorPanel
   ═══════════════════════════════════════════════════════════════════════════ */
export default function WebCreatorPanel() {
  const { getModel } = useStore();
  const { versions, currentIdx, pushVersion, goToVersion, undo, redo, canUndo, canRedo, clear: clearVersions } = useVersionHistory(20, 'aura_webcreator_versions');

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
  const [selectMode, setSelectMode] = useState(false);
  const [selectedElement, setSelectedElement] = useState<{tagName: string, classes: string, id: string, text: string, outerHTML: string, path: string} | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [detached, setDetached] = useState(false);
  const [themeOpen, setThemeOpen] = useState(false);
  const [themeColors, setThemeColors] = useState({
    primary: '#3b82f6',
    secondary: '#6366f1',
    accent: '#f59e0b',
    background: '#ffffff',
    text: '#111827',
  });

  /* ─── Refs ─── */
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const statusTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const detachedWindowRef = useRef<Window | null>(null);

  /* ─── Restore persisted state on mount ─── */
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    wcStorageGet().then(saved => {
      if (!saved) return;
      if (saved.html) {
        setCurrentHtml(saved.html);
        if (iframeRef.current) {
          iframeRef.current.srcdoc = injectIframeScripts(saved.html);
        }
      }
      if (saved.messages?.length) setMessages(saved.messages);
      if (saved.history?.length) setConversationHistory(saved.history);
    });
  }, []);

  /* ─── Auto-save state on changes ─── */
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
    saveTimerRef.current = setTimeout(() => {
      wcStorageSave(currentHtml, messages, conversationHistory);
    }, 1000); // Debounce 1s
    return () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); };
  }, [currentHtml, messages, conversationHistory]);

  /* ─── Cleanup ─── */
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (statusTimerRef.current) clearTimeout(statusTimerRef.current);
      if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
        detachedWindowRef.current.close();
      }
    };
  }, []);

  /* ─── iframe error listener ─── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (!e.data || typeof e.data !== 'object') return;
      if (e.source !== iframeRef.current?.contentWindow) return;
      if (e.data.type === 'artifact-error') {
        const msg = e.data.msg || 'Unknown error';
        const line = e.data.line ? ` (line ${e.data.line})` : '';
        setIframeError(`${msg}${line}`);
      }
      if (e.data.type === 'element-selected') {
        setSelectedElement(e.data);
        setSelectMode(false);
        if (iframeRef.current?.contentWindow) {
          iframeRef.current.contentWindow.postMessage({ type: 'toggle-select-mode', enabled: false }, '*');
        }
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
    iframeRef.current.srcdoc = injectIframeScripts(html);

    // Sync to detached window if open
    if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
      detachedWindowRef.current.document.open();
      detachedWindowRef.current.document.write(html);
      detachedWindowRef.current.document.close();
    }
  }, []);

  /* ─── Detach preview to separate window ─── */
  const detachPreview = useCallback(() => {
    if (detached && detachedWindowRef.current && !detachedWindowRef.current.closed) {
      detachedWindowRef.current.focus();
      return;
    }
    const win = window.open('', 'aura-preview', 'width=1024,height=768');
    if (!win) return;
    detachedWindowRef.current = win;
    setDetached(true);

    // Write current HTML to the new window
    if (currentHtml) {
      win.document.open();
      win.document.write(currentHtml);
      win.document.close();
    }

    // Listen for window close
    const checkClosed = setInterval(() => {
      if (win.closed) {
        clearInterval(checkClosed);
        setDetached(false);
        detachedWindowRef.current = null;
      }
    }, 500);
  }, [detached, currentHtml]);

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

    // Build conversation history for context continuity
    const newHistory = [
      ...conversationHistory,
      { role: 'user', content: text },
    ];

    // Build the user message (just the request + context, NOT the system prompt)
    let userMessage = text;
    if (currentHtml) {
      let elementContext = '';
      if (selectedElement) {
        elementContext = `\n\nThe user has selected a specific element to modify:\nElement: <${selectedElement.tagName.toLowerCase()}>\nCSS path: ${selectedElement.path}\nElement HTML: ${selectedElement.outerHTML}\n\nOnly modify this specific element. Keep all other elements unchanged.`;
      }
      userMessage = `Current HTML page code:\n\`\`\`html\n${currentHtml}\n\`\`\`${elementContext}\n\nUser request: ${text}`;
    }

    const model = getModel('webcreator') || undefined;

    // Set up streaming preview controller
    const previewCtrl = new StreamingPreviewController((html) => {
      if (iframeRef.current) {
        iframeRef.current.srcdoc = injectIframeScripts(html);
      }
    });

    try {
      let streamedText = '';
      let htmlStartIdx = -1; // Track where HTML begins in the stream

      for await (const chunk of streamRawGenerate(userMessage, {
        systemPrompt: SYSTEM_PROMPT,
        model,
        history: conversationHistory.length > 0 ? conversationHistory : undefined,
        signal: ctrl.signal,
      })) {
        streamedText += chunk;

        // Progressive preview: detect HTML start and feed ALL content from that point
        if (htmlStartIdx === -1) {
          const docTypeIdx = streamedText.search(/<!DOCTYPE html>/i);
          const htmlIdx = streamedText.search(/<html[\s >]/i);
          const startIdx = docTypeIdx !== -1 ? docTypeIdx : htmlIdx;
          if (startIdx !== -1) {
            htmlStartIdx = startIdx;
            // Feed everything from HTML start to preview controller (catches initial chunks)
            previewCtrl.append(streamedText.slice(htmlStartIdx));
          }
        } else {
          // HTML already started — just append the new chunk
          previewCtrl.append(chunk);
        }
      }

      // Final extraction
      const finalHtml = extractHtml(streamedText);
      if (finalHtml) {
        setCurrentHtml(finalHtml);
        updatePreview(finalHtml);
        pushVersion(text, finalHtml);
        setViewMode('preview');
      }

      // Add AI message
      const aiMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'ai',
        text: finalHtml ? 'Page updated. Preview is live below.' : streamedText.slice(0, 200),
        timestamp: Date.now(),
      };
      setMessages(prev => [...prev, aiMsg]);

      // Update conversation history (store summary, not full HTML)
      setConversationHistory([
        ...newHistory,
        { role: 'assistant', content: finalHtml ? 'Generated/updated the HTML page as requested.' : streamedText },
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
      previewCtrl.dispose();
      setLoading(false);
      abortRef.current = null;
    }
  }, [input, loading, currentHtml, conversationHistory, getModel, extractHtml, updatePreview, pushVersion, selectedElement]);

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

  const downloadHtml = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'website.html';
    a.click();
    URL.revokeObjectURL(url);
  }, [currentHtml]);

  const copyDataUrl = useCallback(async () => {
    if (!currentHtml) return;
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(currentHtml);
    try {
      await navigator.clipboard.writeText(dataUrl);
      setStatus('Data URL copied!');
      setTimeout(() => setStatus(''), 1500);
    } catch {
      // fallback
    }
  }, [currentHtml]);

  const openInCodeSandbox = useCallback(() => {
    if (!currentHtml) return;
    const params = new URLSearchParams({
      parameters: btoa(JSON.stringify({
        files: {
          'index.html': { content: currentHtml, isBinary: false },
          'package.json': { content: JSON.stringify({ dependencies: {} }), isBinary: false },
        },
      })),
    });
    window.open(`https://codesandbox.io/api/v1/sandboxes/define?${params}`, '_blank');
  }, [currentHtml]);

  const openInStackBlitz = useCallback(() => {
    if (!currentHtml) return;
    const form = document.createElement('form');
    form.method = 'POST';
    form.action = 'https://stackblitz.com/run';
    form.target = '_blank';
    const addField = (name: string, value: string) => {
      const input = document.createElement('input');
      input.type = 'hidden';
      input.name = name;
      input.value = value;
      form.appendChild(input);
    };
    addField('project[template]', 'html');
    addField('project[files][index.html]', currentHtml);
    addField('project[title]', 'AURA WebCreator Export');
    document.body.appendChild(form);
    form.submit();
    document.body.removeChild(form);
  }, [currentHtml]);

  /* ─── Close export dropdown on outside click ─── */
  useEffect(() => {
    if (!exportOpen) return;
    const close = () => setExportOpen(false);
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [exportOpen]);

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
    clearVersions();
    if (iframeRef.current) iframeRef.current.srcdoc = '';
  }, [clearVersions]);

  /* ─── Toggle element select mode ─── */
  const toggleSelectMode = useCallback(() => {
    const next = !selectMode;
    setSelectMode(next);
    setSelectedElement(null);
    if (iframeRef.current?.contentWindow) {
      iframeRef.current.contentWindow.postMessage({ type: 'toggle-select-mode', enabled: next }, '*');
    }
  }, [selectMode]);

  /* ─── Apply theme ─── */
  const applyTheme = useCallback((colors: typeof themeColors) => {
    if (!currentHtml) return;
    const cssVars = `<!-- AURA_THEME --><style>:root { --primary: ${colors.primary}; --secondary: ${colors.secondary}; --accent: ${colors.accent}; --bg: ${colors.background}; --text: ${colors.text}; } body { background-color: ${colors.background}; color: ${colors.text}; }</style><!-- /AURA_THEME -->`;
    // Remove previous theme block if exists
    let html = currentHtml.replace(/<!-- AURA_THEME -->[\s\S]*?<!-- \/AURA_THEME -->/g, '');
    if (html.includes('</head>')) {
      html = html.replace('</head>', cssVars + '</head>');
    } else {
      html = cssVars + html;
    }
    setCurrentHtml(html);
    updatePreview(html);
  }, [currentHtml, updatePreview]);

  /* ─── Version restore ─── */
  const restoreVersion = useCallback((idx: number) => {
    const v = goToVersion(idx);
    if (v) {
      setCurrentHtml(v.code);
      updatePreview(v.code);
    }
  }, [goToVersion, updatePreview]);

  /* ─── Syntax highlighting ─── */
  const highlightedCode = React.useMemo(() => {
    if (!currentHtml) return '';
    return highlightCode(currentHtml, 'html');
  }, [currentHtml]);

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
      <OfflineBanner />
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

        {/* Select element mode */}
        {hasHtml && viewMode === 'preview' && (
          <button
            onClick={toggleSelectMode}
            title={selectMode ? 'Exit selection mode' : 'Select element to edit'}
            style={{
              background: selectMode ? 'rgba(59,130,246,0.15)' : 'var(--s2)',
              border: `1px solid ${selectMode ? 'rgba(59,130,246,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: selectMode ? '#3b82f6' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit', marginLeft: 4,
            }}
          >
            <MousePointer2 size={12} /> {selectMode ? 'Selecting...' : 'Select'}
          </button>
        )}

        {/* Pop out preview */}
        {currentHtml && viewMode === 'preview' && (
          <button
            onClick={detachPreview}
            title={detached ? 'Preview detached — click to focus' : 'Pop out preview'}
            style={{
              background: detached ? 'rgba(34,197,94,0.15)' : 'var(--s2)',
              border: `1px solid ${detached ? 'rgba(34,197,94,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: detached ? '#22c55e' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit',
            }}
          >
            <ExternalLink size={12} /> {detached ? 'Detached' : 'Pop Out'}
          </button>
        )}

        {/* Theme toggle */}
        {hasHtml && viewMode === 'preview' && (
          <button
            onClick={() => setThemeOpen(!themeOpen)}
            title="Theme"
            style={{
              background: themeOpen ? 'rgba(245,158,11,0.15)' : 'var(--s2)',
              border: `1px solid ${themeOpen ? 'rgba(245,158,11,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: themeOpen ? '#f59e0b' : 'var(--mu)',
              padding: '3px 8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
              fontSize: '10px', fontFamily: 'inherit',
            }}
          >
            <Palette size={12} /> Theme
          </button>
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

      {/* ═══ Selected element toolbar ═══ */}
      {selectedElement && (
        <div style={{
          padding: '6px 10px', background: 'rgba(59,130,246,0.08)', borderBottom: '1px solid rgba(59,130,246,0.2)',
          display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', flexShrink: 0,
        }}>
          <span style={{ fontSize: '10px', color: '#3b82f6', fontWeight: 600 }}>
            &lt;{selectedElement.tagName.toLowerCase()}&gt;
          </span>
          {selectedElement.text && (
            <span style={{ fontSize: '10px', color: 'var(--mu)', maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              "{selectedElement.text}"
            </span>
          )}
          <span style={{ flex: 1 }} />
          <button
            onClick={() => {
              const desc = selectedElement.text ? `"${selectedElement.text.slice(0, 40)}"` : `<${selectedElement.tagName.toLowerCase()}>`;
              setInput(`Edit the ${selectedElement.tagName.toLowerCase()} element ${desc}: `);
              setSelectedElement(null);
              inputRef.current?.focus();
            }}
            style={{
              background: '#3b82f6', border: 'none', borderRadius: 'var(--r-sm)',
              color: 'white', padding: '3px 10px', fontSize: '10px', cursor: 'pointer',
              fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4,
            }}
          >
            <Pencil size={10} /> Edit with AI
          </button>
          <button
            onClick={() => setSelectedElement(null)}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--mu)', padding: '3px 8px', fontSize: '10px', cursor: 'pointer', fontFamily: 'inherit',
            }}
          >
            Cancel
          </button>
        </div>
      )}

      {/* ═══ Theme panel ═══ */}
      {themeOpen && currentHtml && (
        <div style={{
          padding: '8px 10px', borderBottom: '1px solid var(--b1)', background: 'var(--s1)',
        }}>
          <div style={{ fontSize: '10px', color: 'var(--mu)', fontWeight: 600, marginBottom: 6 }}>Theme Colors</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {Object.entries(themeColors).map(([key, value]) => (
              <label key={key} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: '10px', color: 'var(--mu)' }}>
                <input
                  type="color"
                  value={value}
                  onChange={e => {
                    const newColors = { ...themeColors, [key]: e.target.value };
                    setThemeColors(newColors);
                    applyTheme(newColors);
                  }}
                  style={{ width: 20, height: 20, border: '1px solid var(--b1)', borderRadius: 3, padding: 0, cursor: 'pointer' }}
                />
                {key}
              </label>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
            <button
              onClick={() => {
                const darkColors = { primary: '#3b82f6', secondary: '#8b5cf6', accent: '#f59e0b', background: '#0f172a', text: '#f1f5f9' };
                setThemeColors(darkColors);
                applyTheme(darkColors);
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              Dark
            </button>
            <button
              onClick={() => {
                const lightColors = { primary: '#3b82f6', secondary: '#6366f1', accent: '#f59e0b', background: '#ffffff', text: '#111827' };
                setThemeColors(lightColors);
                applyTheme(lightColors);
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              Light
            </button>
            <button
              onClick={() => {
                setInput('Suggest a harmonious color palette for this design and update the CSS accordingly');
              }}
              style={{ fontSize: '9px', padding: '2px 8px', background: 'var(--s2)', color: 'var(--mu)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', cursor: 'pointer', fontFamily: 'inherit' }}
            >
              AI Suggest
            </button>
          </div>
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
                    <span style={{
                      fontSize: '10.5px', color: 'var(--pl)', fontWeight: 500,
                      marginLeft: 4, animation: 'streamPulse 1.5s ease-in-out infinite',
                    }}>
                      Generating...
                    </span>
                    <style>{`@keyframes streamPulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
                  </div>
                )}
              </div>
            )}

            {/* Preview / Code area */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              {/* Preview iframe */}
              {hasHtml && viewMode === 'preview' && (
                detached ? (
                  <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                    justifyContent: 'center', gap: 8, color: 'var(--mu)',
                  }}>
                    <ExternalLink size={24} />
                    <div style={{ fontSize: '12px' }}>Preview detached to separate window</div>
                    <button
                      onClick={() => {
                        if (detachedWindowRef.current && !detachedWindowRef.current.closed) {
                          detachedWindowRef.current.close();
                        }
                        setDetached(false);
                        detachedWindowRef.current = null;
                      }}
                      style={{
                        background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                        color: 'var(--mu)', padding: '4px 12px', fontSize: '11px', cursor: 'pointer', fontFamily: 'inherit',
                      }}
                    >
                      Bring Back
                    </button>
                  </div>
                ) : (
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
                )
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
        {loading ? (
          <button
            onClick={() => {
              if (abortRef.current) abortRef.current.abort();
              abortRef.current = null;
              setLoading(false);
              setStatus('Cancelled');
              setMessages(prev => [...prev, {
                id: crypto.randomUUID(),
                role: 'ai' as const,
                text: 'Generation stopped.',
                timestamp: Date.now(),
              }]);
            }}
            className="stop-stream-btn"
            aria-label="Stop generating"
            style={{
              alignSelf: 'stretch', minWidth: 40, minHeight: 36,
              padding: '0 12px',
            }}
          >
            <Square size={10} />
            <span>Stop</span>
          </button>
        ) : (
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim()}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'var(--p)',
              border: 'none', borderRadius: 'var(--r-md)', color: '#fff',
              padding: '0 14px', cursor: !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: '12px', fontFamily: 'inherit', fontWeight: 600,
              alignSelf: 'stretch', minWidth: 40, minHeight: 36,
              opacity: !input.trim() ? 0.5 : 1,
              transition: 'all 0.2s ease',
              boxShadow: '0 2px 10px rgba(124,58,237,0.3)',
            }}
          >
            <Send size={14} />
          </button>
        )}
      </div>

      {/* ═══ Version timeline strip ═══ */}
      {versions.length > 1 && (
        <div style={{
          display: 'flex', gap: 4, padding: '4px 8px', borderTop: '1px solid var(--b1)',
          overflowX: 'auto', background: 'var(--s1)', flexShrink: 0,
        }}>
          {versions.map((v, i) => (
            <button
              key={v.id}
              onClick={() => restoreVersion(i)}
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

      {/* ═══ Footer action bar ═══ */}
      {hasHtml && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: 6, padding: '6px 10px', flexShrink: 0,
          borderTop: '1px solid var(--b1)',
        }}>
          <button
            onClick={() => { const v = undo(); if (v) { setCurrentHtml(v.code); updatePreview(v.code); } }}
            disabled={!canUndo}
            title="Undo"
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: canUndo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canUndo ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', opacity: canUndo ? 1 : 0.4,
            }}
          >
            <Undo2 size={12} />
          </button>
          <button
            onClick={() => { const v = redo(); if (v) { setCurrentHtml(v.code); updatePreview(v.code); } }}
            disabled={!canRedo}
            title="Redo"
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: canRedo ? 'var(--mu)' : 'var(--s3)', padding: '3px 6px', cursor: canRedo ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', opacity: canRedo ? 1 : 0.4,
            }}
          >
            <Redo2 size={12} />
          </button>
          <ActionBtn id="copy" icon={<Copy size={13} />} label="Copy HTML" onClick={copyCode} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
          <div style={{ position: 'relative' }}>
            <button
              onClick={() => setExportOpen(!exportOpen)}
              title="Export"
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                color: 'var(--mu)', padding: '3px 8px', cursor: 'pointer', display: 'flex',
                alignItems: 'center', gap: 4, fontSize: '10px', fontFamily: 'inherit',
              }}
            >
              <Download size={12} /> Export
            </button>
            {exportOpen && (
              <div style={{
                position: 'absolute', bottom: '100%', left: 0, marginBottom: 4,
                background: '#1e1e1e', border: '1px solid #333', borderRadius: 'var(--r-md)',
                padding: '4px 0', minWidth: 160, zIndex: 20, boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
              }}>
                <button onClick={() => { downloadHtml(); setExportOpen(false); }} style={exportItemStyle}>
                  Download HTML
                </button>
                <button onClick={() => { copyDataUrl(); setExportOpen(false); }} style={exportItemStyle}>
                  Copy as Data URL
                </button>
                <button onClick={() => { openInCodeSandbox(); setExportOpen(false); }} style={exportItemStyle}>
                  Open in CodeSandbox
                </button>
                <button onClick={() => { openInStackBlitz(); setExportOpen(false); }} style={exportItemStyle}>
                  Open in StackBlitz
                </button>
              </div>
            )}
          </div>
          <ActionBtn id="sendcli" icon={<Upload size={13} />} label="Send to CLI" onClick={sendToCli} hoveredBtn={hoveredBtn} setHoveredBtn={setHoveredBtn} />
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
    </div>
  );
}
