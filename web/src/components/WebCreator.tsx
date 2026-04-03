import { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon, ArrowDownTrayIcon,
  DevicePhoneMobileIcon, DeviceTabletIcon, ComputerDesktopIcon,
  ArrowUturnLeftIcon, ArrowUturnRightIcon,
  ArrowTopRightOnSquareIcon, CodeBracketIcon, EyeIcon,
  StopIcon,
} from '@heroicons/react/24/outline';
import { buildSrcdoc } from '../utils/artifactRenderer';
import { highlightCode } from '../utils/codeHighlighter';

/* ── Types ── */
type DeviceSize = 'desktop' | 'tablet' | 'mobile';
type ViewMode = 'preview' | 'code' | 'split';

const DEVICE_WIDTHS: Record<DeviceSize, string> = {
  desktop: '100%',
  tablet: '768px',
  mobile: '375px',
};

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface Version {
  html: string;
  timestamp: number;
}

/* ── Templates ── */
const TEMPLATES = [
  { label: 'Landing Page', icon: '🚀', color: '#7c3aed', prompt: 'Create a modern SaaS landing page with a hero section featuring a bold headline, subtitle, and CTA button. Include sections for features (3 cards with icons), testimonials, pricing tiers, and a footer. Use a gradient purple/blue color scheme.' },
  { label: 'Portfolio', icon: '👤', color: '#06b6d4', prompt: 'Create a personal portfolio website with a hero section, about section, a project gallery with 4 cards, a skills section with progress bars, and a contact form. Use a dark minimal theme.' },
  { label: 'Blog', icon: '📝', color: '#10b981', prompt: 'Create a blog homepage with a header/nav, a featured post hero, a grid of 6 blog post cards, sidebar with categories, and a newsletter signup. Use clean typography.' },
  { label: 'Dashboard', icon: '📊', color: '#f59e0b', prompt: 'Create an analytics dashboard with a sidebar nav, top stats row (4 metric cards), a large chart placeholder, a table of recent transactions, and a donut chart. Dark theme.' },
  { label: 'Login', icon: '🔐', color: '#ec4899', prompt: 'Create a beautiful login page with a split layout — left side has a gradient background with branding, right side has a centered login form with social login buttons.' },
  { label: 'Pricing', icon: '💎', color: '#8b5cf6', prompt: 'Create a pricing page with 3 tiers (Basic, Pro, Enterprise) in cards. Middle card highlighted. Monthly/annual toggle and FAQ section.' },
  { label: '404 Page', icon: '🔍', color: '#ef4444', prompt: 'Create a creative 404 error page with a large "404" display using CSS animation, a witty message, search bar, and "Go Home" button with floating animated elements.' },
];

const SYSTEM_PROMPT = `You are an expert web designer and developer. Generate a complete, beautiful HTML page with inline CSS and JavaScript.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Include ALL CSS in a <style> tag inside <head>
- Include ALL JavaScript in a <script> tag before </body>
- Use modern CSS: flexbox, grid, custom properties, smooth transitions
- Use clean typography with system fonts
- Make it fully responsive
- Use professional color schemes with proper contrast
- Add subtle animations and hover effects
- NO markdown fences, NO explanation text, ONLY the HTML document
- If the user asks for modifications, return the COMPLETE updated HTML`;

/* ── Main Component ── */
export function WebCreator() {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentHtml, setCurrentHtml] = useState('');
  const [streamingCode, setStreamingCode] = useState('');
  const [versions, setVersions] = useState<Version[]>([]);
  const [versionIndex, setVersionIndex] = useState(-1);
  const [device, setDevice] = useState<DeviceSize>('desktop');
  const [viewMode, setViewMode] = useState<ViewMode>('preview');
  const [preGenViewMode, setPreGenViewMode] = useState<ViewMode | null>(null);
  const [codeHtml, setCodeHtml] = useState('');
  const [showTemplates, setShowTemplates] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const codeEndRef = useRef<HTMLPreElement>(null);

  const chatScrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Syntax highlight current HTML for code view
  useEffect(() => {
    if (currentHtml) {
      const isDark = !document.documentElement.classList.contains('light');
      highlightCode(currentHtml, 'html', isDark ? 'dark' : 'light')
        .then(setCodeHtml).catch(() => {});
    }
  }, [currentHtml]);

  // Auto-scroll chat
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [chatMessages]);

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Fetch available models
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  // Close model menu on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Auto-scroll code view during streaming
  useEffect(() => {
    if (isGenerating && codeEndRef.current) {
      codeEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamingCode, isGenerating]);

  const addVersion = useCallback((html: string) => {
    setVersions((prev) => {
      const next = [...prev, { html, timestamp: Date.now() }].slice(-20);
      setVersionIndex(next.length - 1);
      return next;
    });
    setCurrentHtml(html);
  }, []);

  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() || isGenerating) return;
    setShowTemplates(false);

    const userMsg: ChatMessage = { role: 'user', content: message };
    setChatMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsGenerating(true);

    // Build context with current HTML if editing
    const systemCtx = currentHtml
      ? `${SYSTEM_PROMPT}\n\nCurrent page HTML:\n${currentHtml}`
      : SYSTEM_PROMPT;

    // Build history from prior chat messages
    const history = chatMessages.map((m) => ({ role: m.role, content: m.content }));

    const controller = new AbortController();
    abortRef.current = controller;

    // Switch to split/code view so user sees code being written
    setPreGenViewMode(viewMode);
    if (viewMode === 'preview') setViewMode('split');
    setStreamingCode('');

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: message,
          system_prompt: systemCtx,
          history: history,
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';
      const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
      setChatMessages((prev) => [...prev, assistantMsg]);

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          // Parse SSE or raw chunks
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) {
                  fullResponse += text;
                  setStreamingCode(fullResponse);
                  setChatMessages((prev) => {
                    const updated = [...prev];
                    updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                    return updated;
                  });
                }
              } catch {
                // Raw text chunk
                fullResponse += data;
                setStreamingCode(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              // Non-SSE — raw streaming
              fullResponse += line;
              setStreamingCode(fullResponse);
              setChatMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                return updated;
              });
            }
          }
        }
      } else {
        const text = await res.text();
        fullResponse = text;
        setChatMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
          return updated;
        });
      }

      // Extract HTML from response
      let html = fullResponse.trim();
      // Strip markdown fences if present
      const fenceMatch = html.match(/```html?\s*\n([\s\S]*?)```/);
      if (fenceMatch) html = fenceMatch[1].trim();
      // Validate it looks like HTML
      if (html.includes('<!DOCTYPE') || html.includes('<html') || html.includes('<body') || html.includes('<div')) {
        addVersion(html);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setChatMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Error: ${e.message}. Make sure the backend is running.` },
        ]);
      }
    } finally {
      setIsGenerating(false);
      setStreamingCode('');
      // Restore previous view mode after generation
      if (preGenViewMode !== null) {
        setViewMode(preGenViewMode);
        setPreGenViewMode(null);
      }
      abortRef.current = null;
    }
  }, [chatMessages, currentHtml, isGenerating, addVersion]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleUndo = useCallback(() => {
    if (versionIndex > 0) {
      const i = versionIndex - 1;
      setVersionIndex(i);
      setCurrentHtml(versions[i].html);
    }
  }, [versionIndex, versions]);

  const handleRedo = useCallback(() => {
    if (versionIndex < versions.length - 1) {
      const i = versionIndex + 1;
      setVersionIndex(i);
      setCurrentHtml(versions[i].html);
    }
  }, [versionIndex, versions]);

  const handleDownload = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aura-website-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentHtml]);

  const handleOpenNewTab = useCallback(() => {
    if (!currentHtml) return;
    const win = window.open('', '_blank');
    if (win) { win.document.write(currentHtml); win.document.close(); }
  }, [currentHtml]);

  const srcdoc = currentHtml ? buildSrcdoc('html', currentHtml) : '';
  const showPreview = viewMode === 'preview' || viewMode === 'split';
  const showCode = viewMode === 'code' || viewMode === 'split';

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Chat panel — full width on mobile, fixed width on desktop */}
      <div className="flex flex-col md:w-[400px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[40vh] bg-surface-0">
        {/* Chat header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Web Creator</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Describe a website and Aura will build it</p>
        </div>

        {/* Chat messages */}
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Template picker */}
          {showTemplates && chatMessages.length === 0 && (
            <div className="space-y-2">
              <p className="text-xs text-chat-text-secondary mb-3">Start with a template or describe what you want:</p>
              <div className="grid grid-cols-2 gap-2">
                {TEMPLATES.map((t) => (
                  <button
                    key={t.label}
                    onClick={() => handleSend(t.prompt)}
                    className="flex items-center gap-2 p-3 rounded-lg border border-chat-border hover:border-purple-500/30 text-left transition-all group bg-surface-1"
                  >
                    <span className="text-lg">{t.icon}</span>
                    <span className="text-xs font-medium text-chat-text group-hover:text-white transition-colors">{t.label}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {chatMessages.map((msg, i) => (
            <div
              key={i}
              className={`text-sm ${msg.role === 'user' ? 'text-right' : ''}`}
            >
              {msg.role === 'user' ? (
                <div className="inline-block px-3 py-2 rounded-xl bg-chat-accent text-white max-w-[90%] text-left">
                  {msg.content.length > 200 ? msg.content.slice(0, 200) + '...' : msg.content}
                </div>
              ) : (
                <div className="text-xs text-chat-text-secondary">
                  {msg.content.includes('<!DOCTYPE') || msg.content.includes('<html')
                    ? <span className="text-green-400">
                        {isGenerating && i === chatMessages.length - 1
                          ? `Writing code... (${Math.round(msg.content.length / 1024)}KB)`
                          : `Generated website (${Math.round(msg.content.length / 1024)}KB)`
                        }
                      </span>
                    : msg.content.length > 300 ? msg.content.slice(0, 300) + '...' : msg.content
                  }
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex items-center gap-2 text-xs text-purple-400">
              <div className="shimmer-bar h-2 w-20" />
              Generating...
            </div>
          )}
        </div>

        {/* Chat input */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(input);
                }
              }}
              placeholder={currentHtml ? 'Describe changes...' : 'Describe your website...'}
              className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={2}
              disabled={isGenerating}
            />
            <button
              onClick={isGenerating ? handleStop : () => handleSend(input)}
              disabled={!isGenerating && !input.trim()}
              className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
            >
              {isGenerating ? <StopIcon className="w-4 h-4" /> : <PaperAirplaneIcon className="w-4 h-4" />}
            </button>
          </div>
          {/* Model selector */}
          <div className="flex items-center mt-1.5" ref={modelMenuRef}>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu(p => !p)}
                className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
                style={{ background: 'var(--border-subtle)' }}
              >
                <span className="max-w-[140px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
                <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
              </button>
              {showModelMenu && availableModels.length > 0 && (
                <div
                  style={{
                    position: 'absolute', bottom: 28, left: 0, width: 220, maxHeight: 280,
                    background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                    borderRadius: 10, overflow: 'hidden', zIndex: 50,
                  }}
                >
                  <div style={{ maxHeight: 280, overflowY: 'auto', padding: 4 }}>
                    <button
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                      style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map((m) => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                        style={{ color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === m ? 'var(--surface-3)' : 'transparent' }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Right: Preview + toolbar */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-chat-border flex-shrink-0 flex-wrap">
          {/* View modes */}
          <div className="flex rounded-md border border-chat-border overflow-hidden">
            {(['preview', 'code', 'split'] as ViewMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setViewMode(m)}
                className={`px-2 py-1 text-[10px] capitalize transition-colors ${viewMode === m ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
              >
                {m === 'preview' ? <EyeIcon className="w-3.5 h-3.5 inline" /> : m === 'code' ? <CodeBracketIcon className="w-3.5 h-3.5 inline" /> : 'Split'}
              </button>
            ))}
          </div>

          {/* Device */}
          <div className="flex rounded-md border border-chat-border overflow-hidden ml-1">
            {([['desktop', ComputerDesktopIcon], ['tablet', DeviceTabletIcon], ['mobile', DevicePhoneMobileIcon]] as [DeviceSize, any][]).map(([d, Icon]) => (
              <button
                key={d}
                onClick={() => setDevice(d)}
                className={`p-1 transition-colors ${device === d ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
              >
                <Icon className="w-3.5 h-3.5" />
              </button>
            ))}
          </div>

          <div className="flex-1" />

          {/* Version controls */}
          {versions.length > 0 && (
            <>
              <span className="text-[10px] text-chat-text-secondary mr-1">v{versionIndex + 1}/{versions.length}</span>
              <button onClick={handleUndo} disabled={versionIndex <= 0} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30">
                <ArrowUturnLeftIcon className="w-3.5 h-3.5" />
              </button>
              <button onClick={handleRedo} disabled={versionIndex >= versions.length - 1} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30">
                <ArrowUturnRightIcon className="w-3.5 h-3.5" />
              </button>
            </>
          )}

          {/* Actions */}
          <button onClick={handleDownload} disabled={!currentHtml} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30" title="Download HTML">
            <ArrowDownTrayIcon className="w-3.5 h-3.5" />
          </button>
          <button onClick={handleOpenNewTab} disabled={!currentHtml} className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30" title="Open in new tab">
            <ArrowTopRightOnSquareIcon className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Preview area */}
        <div className={`flex-1 overflow-hidden flex ${viewMode === 'split' ? 'flex-row' : 'flex-col'}`}>
          {showPreview && (
            <div className={`${viewMode === 'split' ? 'w-1/2 border-r border-chat-border' : 'flex-1'} overflow-auto flex justify-center`} style={{ background: currentHtml ? 'white' : 'var(--surface-1)' }}>
              {currentHtml ? (
                <div style={{ width: DEVICE_WIDTHS[device], maxWidth: '100%', height: '100%' }} className="transition-all duration-300">
                  <iframe
                    srcDoc={srcdoc}
                    sandbox="allow-scripts allow-same-origin"
                    className="w-full h-full border-none"
                    title="Website preview"
                  />
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">
                  <div className="text-center">
                    <p className="text-2xl mb-2">🌐</p>
                    <p>Your website will appear here</p>
                    <p className="text-[10px] mt-1">Pick a template or describe what you want</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {showCode && (
            <div className={`${viewMode === 'split' ? 'w-1/2' : 'flex-1'} overflow-auto bg-surface-1`}>
              {isGenerating && streamingCode ? (
                /* Live streaming code view */
                <pre className="p-4 text-xs font-mono text-green-400 whitespace-pre-wrap leading-relaxed">
                  {streamingCode}
                  <span ref={codeEndRef} className="inline-block w-1.5 h-3.5 bg-green-400 animate-pulse ml-0.5 align-middle" />
                </pre>
              ) : currentHtml ? (
                codeHtml ? (
                  <div
                    dangerouslySetInnerHTML={{ __html: codeHtml }}
                    className="shiki-block p-4 text-sm [&_pre]:!bg-transparent [&_pre]:!m-0 [&_pre]:!p-0 [&_code]:!bg-transparent"
                  />
                ) : (
                  <pre className="p-4 text-sm font-mono text-chat-text whitespace-pre-wrap">{currentHtml}</pre>
                )
              ) : (
                <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">No code yet</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
