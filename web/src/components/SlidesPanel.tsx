import { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon,
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  ArrowRightIcon,
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon,
  StopIcon,
  PresentationChartBarIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

type SlideCount = 5 | 8 | 10 | 15;
type SlideStyle = 'Professional' | 'Minimal' | 'Creative' | 'Dark' | 'Colorful';

/* ── Constants ── */
const SLIDE_COUNTS: SlideCount[] = [5, 8, 10, 15];

const SLIDE_STYLES: { label: SlideStyle; color: string }[] = [
  { label: 'Professional', color: '#2563eb' },
  { label: 'Minimal',      color: '#6b7280' },
  { label: 'Creative',     color: '#7c3aed' },
  { label: 'Dark',         color: '#1e293b' },
  { label: 'Colorful',     color: '#f59e0b' },
];

const SYSTEM_PROMPT = `You are a presentation designer. Generate a complete HTML presentation with inline CSS and JavaScript.

Rules:
- Output ONLY complete HTML starting with <!DOCTYPE html>
- Each slide is a <section class='slide'> element
- Include CSS for: slide layout (100vw x 100vh per slide), navigation, transitions
- Include JS for keyboard navigation (arrow keys) and slide counter
- Use professional typography and colors matching the requested style
- Make slides visually impactful with proper hierarchy
- Include speaker notes as data-notes attributes
- NO markdown, NO explanation, ONLY the HTML document
- If user asks for modifications, return the COMPLETE updated HTML`;

/* ── Component ── */
export function SlidesPanel() {
  const [chatMessages, setChatMessages]     = useState<ChatMessage[]>([]);
  const [input, setInput]                   = useState('');
  const [topic, setTopic]                   = useState('');
  const [slideCount, setSlideCount]         = useState<SlideCount>(8);
  const [slideStyle, setSlideStyle]         = useState<SlideStyle>('Professional');
  const [isGenerating, setIsGenerating]     = useState(false);
  const [currentHtml, setCurrentHtml]       = useState('');
  const [streamingCode, setStreamingCode]   = useState('');
  const [currentSlide, setCurrentSlide]     = useState(1);
  const [totalSlides, setTotalSlides]       = useState(0);
  const [isFullscreen, setIsFullscreen]     = useState(false);
  const [selectedModel, setSelectedModel]   = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu]   = useState(false);

  const chatScrollRef  = useRef<HTMLDivElement>(null);
  const abortRef       = useRef<AbortController | null>(null);
  const iframeRef      = useRef<HTMLIFrameElement>(null);
  const modelMenuRef   = useRef<HTMLDivElement>(null);
  const fullscreenRef  = useRef<HTMLDivElement>(null);

  /* ── Auto-scroll chat ── */
  useEffect(() => {
    chatScrollRef.current?.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [chatMessages]);

  /* ── Cleanup abort on unmount ── */
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  /* ── Fetch models ── */
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models    || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models      || []),
          ...(data.local_models      || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  /* ── Close model menu on outside click ── */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* ── Count slides in generated HTML ── */
  const countSlides = useCallback((html: string): number => {
    const matches = html.match(/<section[^>]*class=['"][^'"]*slide[^'"]*['"]/gi);
    return matches ? matches.length : 0;
  }, []);

  /* ── Update slide counter from iframe postMessage ── */
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'slideChange') {
        setCurrentSlide(e.data.current ?? 1);
        setTotalSlides(e.data.total ?? 0);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  /* ── Reset slide position when new HTML is set ── */
  useEffect(() => {
    if (currentHtml) {
      const count = countSlides(currentHtml);
      setTotalSlides(count);
      setCurrentSlide(1);
    }
  }, [currentHtml, countSlides]);

  /* ── Navigate via postMessage ── */
  const sendNavMessage = useCallback((direction: 'prev' | 'next') => {
    iframeRef.current?.contentWindow?.postMessage({ type: 'navigate', direction }, '*');
    setCurrentSlide(prev => {
      if (direction === 'prev') return Math.max(1, prev - 1);
      return Math.min(totalSlides || prev, prev + 1);
    });
  }, [totalSlides]);

  /* ── Fullscreen toggle ── */
  const toggleFullscreen = useCallback(() => {
    if (!isFullscreen) {
      fullscreenRef.current?.requestFullscreen?.().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.().catch(() => {});
      setIsFullscreen(false);
    }
  }, [isFullscreen]);

  useEffect(() => {
    const handler = () => {
      if (!document.fullscreenElement) setIsFullscreen(false);
    };
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  /* ── Generate slides ── */
  const handleSend = useCallback(async (message: string) => {
    if (!message.trim() || isGenerating) return;

    const userMsg: ChatMessage = { role: 'user', content: message };
    setChatMessages(prev => [...prev, userMsg]);
    setInput('');
    setTopic('');
    setIsGenerating(true);

    const isFirstGeneration = chatMessages.length === 0;
    const systemCtx = isFirstGeneration
      ? `${SYSTEM_PROMPT}\n\nStyle: ${slideStyle}. Number of slides: ${slideCount}.`
      : currentHtml
        ? `${SYSTEM_PROMPT}\n\nCurrent presentation HTML:\n${currentHtml}`
        : SYSTEM_PROMPT;

    const history = chatMessages.map(m => ({ role: m.role, content: m.content }));
    const controller = new AbortController();
    abortRef.current = controller;
    setStreamingCode('');

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          system_prompt: systemCtx,
          history,
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';
      const assistantMsg: ChatMessage = { role: 'assistant', content: '' };
      setChatMessages(prev => [...prev, assistantMsg]);

      if (res.body) {
        const reader  = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

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
                  setChatMessages(prev => {
                    const updated = [...prev];
                    updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                    return updated;
                  });
                }
              } catch {
                fullResponse += data;
                setStreamingCode(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setStreamingCode(fullResponse);
              setChatMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
                return updated;
              });
            }
          }
        }
      } else {
        fullResponse = await res.text();
        setChatMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: 'assistant', content: fullResponse };
          return updated;
        });
      }

      /* Extract HTML */
      let html = fullResponse.trim();
      const fenceMatch = html.match(/```html?\s*\n([\s\S]*?)```/);
      if (fenceMatch) html = fenceMatch[1].trim();

      if (html.includes('<!DOCTYPE') || html.includes('<html') || html.includes('<section')) {
        setCurrentHtml(html);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setChatMessages(prev => [
          ...prev,
          { role: 'assistant', content: `Error: ${e.message}. Make sure the backend is running.` },
        ]);
      }
    } finally {
      setIsGenerating(false);
      setStreamingCode('');
      abortRef.current = null;
    }
  }, [chatMessages, currentHtml, isGenerating, selectedModel, slideCount, slideStyle]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleDownload = useCallback(() => {
    if (!currentHtml) return;
    const blob = new Blob([currentHtml], { type: 'text/html' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `aura-slides-${Date.now()}.html`;
    a.click();
    URL.revokeObjectURL(url);
  }, [currentHtml]);

  const handleGenerate = useCallback(() => {
    if (!topic.trim()) return;
    const prompt = `Create a ${slideCount}-slide ${slideStyle.toLowerCase()} presentation about: ${topic}`;
    handleSend(prompt);
  }, [topic, slideCount, slideStyle, handleSend]);

  /* ── Inject postMessage listener into srcdoc ── */
  const buildSrcdoc = (html: string): string => {
    const listenerScript = `
<script>
(function() {
  window.addEventListener('message', function(e) {
    if (!e.data || e.data.type !== 'navigate') return;
    var slides = document.querySelectorAll('.slide');
    if (!slides.length) return;
    var current = parseInt(document.body.dataset.currentSlide || '0', 10);
    if (e.data.direction === 'next') current = Math.min(slides.length - 1, current + 1);
    else current = Math.max(0, current - 1);
    document.body.dataset.currentSlide = current;
    slides.forEach(function(s, i) {
      s.style.display = i === current ? '' : 'none';
    });
    try { window.parent.postMessage({ type: 'slideChange', current: current + 1, total: slides.length }, '*'); } catch(e) {}
  });
  // Report initial slide count after load
  window.addEventListener('load', function() {
    var slides = document.querySelectorAll('.slide');
    if (slides.length) {
      try { window.parent.postMessage({ type: 'slideChange', current: 1, total: slides.length }, '*'); } catch(e) {}
    }
  });
})();
</script>`;
    // Insert before </body> if present, otherwise append
    if (html.includes('</body>')) {
      return html.replace('</body>', listenerScript + '</body>');
    }
    return html + listenerScript;
  };

  const isFirstMessage = chatMessages.length === 0;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* ── Left: Chat panel ── */}
      <div className="flex flex-col md:w-[400px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0">
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Slides Builder</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Describe a topic and Aura will generate a presentation</p>
        </div>

        {/* Chat messages */}
        <div ref={chatScrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {/* Initial configuration form */}
          {isFirstMessage && (
            <div className="space-y-3">
              <p className="text-xs text-chat-text-secondary">Configure your presentation:</p>

              {/* Topic input */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Topic</label>
                <input
                  type="text"
                  value={topic}
                  onChange={e => setTopic(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') handleGenerate(); }}
                  placeholder="e.g. The Future of Renewable Energy"
                  className="w-full px-3 py-2 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
                  disabled={isGenerating}
                />
              </div>

              {/* Slide count */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Slides</label>
                <div className="flex gap-1.5">
                  {SLIDE_COUNTS.map(n => (
                    <button
                      key={n}
                      onClick={() => setSlideCount(n)}
                      className="flex-1 py-1.5 rounded-lg text-xs font-medium border transition-all"
                      style={{
                        background:   slideCount === n ? 'var(--chat-accent)' : 'var(--surface-1)',
                        borderColor:  slideCount === n ? 'var(--chat-accent)' : 'var(--border-default)',
                        color:        slideCount === n ? '#fff' : 'var(--text-secondary)',
                      }}
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>

              {/* Style selector */}
              <div>
                <label className="text-[10px] text-chat-text-secondary uppercase tracking-wide mb-1 block">Style</label>
                <div className="grid grid-cols-2 gap-1.5">
                  {SLIDE_STYLES.map(s => (
                    <button
                      key={s.label}
                      onClick={() => setSlideStyle(s.label)}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-all"
                      style={{
                        background:  slideStyle === s.label ? 'var(--surface-2)' : 'var(--surface-1)',
                        borderColor: slideStyle === s.label ? s.color : 'var(--border-default)',
                        color:       slideStyle === s.label ? 'var(--text-primary)' : 'var(--text-secondary)',
                      }}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{ background: s.color }}
                      />
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Generate button */}
              <button
                onClick={handleGenerate}
                disabled={!topic.trim() || isGenerating}
                className="w-full py-2 rounded-lg text-sm font-medium text-white transition-opacity disabled:opacity-40"
                style={{ background: 'var(--chat-accent)' }}
              >
                Generate Presentation
              </button>

              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-chat-border" />
                </div>
                <div className="relative flex justify-center">
                  <span className="px-2 text-[10px] text-chat-text-secondary bg-surface-0">or type a request</span>
                </div>
              </div>
            </div>
          )}

          {/* Messages */}
          {chatMessages.map((msg, i) => (
            <div key={i} className={`text-sm ${msg.role === 'user' ? 'text-right' : ''}`}>
              {msg.role === 'user' ? (
                <div className="inline-block px-3 py-2 rounded-xl bg-chat-accent text-white max-w-[90%] text-left text-xs">
                  {msg.content.length > 200 ? msg.content.slice(0, 200) + '...' : msg.content}
                </div>
              ) : (
                <div className="text-xs text-chat-text-secondary">
                  {msg.content.includes('<!DOCTYPE') || msg.content.includes('<section')
                    ? (
                      <span className="text-green-400">
                        {isGenerating && i === chatMessages.length - 1
                          ? `Building slides... (${Math.round(msg.content.length / 1024)}KB)`
                          : `Generated presentation (${Math.round(msg.content.length / 1024)}KB)`
                        }
                      </span>
                    )
                    : msg.content.length > 300 ? msg.content.slice(0, 300) + '...' : msg.content
                  }
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex items-center gap-2 text-xs text-purple-400">
              <div className="shimmer-bar h-2 w-20" />
              Generating slides...
            </div>
          )}
        </div>

        {/* Chat input */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(input);
                }
              }}
              placeholder={currentHtml ? 'e.g. "Make slide 3 more visual"' : 'Describe your presentation...'}
              className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={2}
              disabled={isGenerating}
            />
            <button
              onClick={isGenerating ? handleStop : () => handleSend(input)}
              disabled={!isGenerating && !input.trim()}
              className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
            >
              {isGenerating
                ? <StopIcon className="w-4 h-4" />
                : <PaperAirplaneIcon className="w-4 h-4" />
              }
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
                <span className="max-w-[140px] truncate">
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                </span>
                <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
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
                      style={{
                        color:      !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
                        background: !selectedModel ? 'var(--surface-3)' : 'transparent',
                      }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map(m => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                        style={{
                          color:      selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                          background: selectedModel === m ? 'var(--surface-3)' : 'transparent',
                        }}
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

      {/* ── Right: Preview ── */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Toolbar */}
        <div className="flex items-center gap-1 px-3 py-2 border-b border-chat-border flex-shrink-0 flex-wrap">
          {/* Slide counter */}
          {totalSlides > 0 && (
            <span className="text-[10px] text-chat-text-secondary px-2 py-1 rounded-md" style={{ background: 'var(--surface-1)' }}>
              {currentSlide} / {totalSlides}
            </span>
          )}

          {/* Navigation */}
          <button
            onClick={() => sendNavMessage('prev')}
            disabled={!currentHtml || currentSlide <= 1}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Previous slide"
          >
            <ArrowLeftIcon className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => sendNavMessage('next')}
            disabled={!currentHtml || currentSlide >= totalSlides}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Next slide"
          >
            <ArrowRightIcon className="w-3.5 h-3.5" />
          </button>

          <div className="flex-1" />

          {/* Live streaming indicator */}
          {isGenerating && streamingCode && (
            <span className="text-[10px] text-green-400 px-2">
              Writing... {Math.round(streamingCode.length / 1024)}KB
            </span>
          )}

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            disabled={!currentHtml}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title={isFullscreen ? 'Exit fullscreen' : 'Present fullscreen'}
          >
            {isFullscreen
              ? <ArrowsPointingInIcon className="w-3.5 h-3.5" />
              : <ArrowsPointingOutIcon className="w-3.5 h-3.5" />
            }
          </button>

          {/* Download */}
          <button
            onClick={handleDownload}
            disabled={!currentHtml}
            className="p-1 text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            title="Download as HTML"
          >
            <ArrowDownTrayIcon className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Preview area */}
        <div ref={fullscreenRef} className="flex-1 overflow-hidden relative">
          {isGenerating && streamingCode && !currentHtml ? (
            /* Live code stream before first render */
            <pre className="p-4 text-xs font-mono text-green-400 whitespace-pre-wrap leading-relaxed h-full overflow-auto bg-surface-1">
              {streamingCode}
              <span className="inline-block w-1.5 h-3.5 bg-green-400 animate-pulse ml-0.5 align-middle" />
            </pre>
          ) : currentHtml ? (
            <iframe
              ref={iframeRef}
              srcDoc={buildSrcdoc(currentHtml)}
              sandbox="allow-scripts allow-same-origin"
              className="w-full h-full border-none"
              title="Slides preview"
            />
          ) : (
            <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">
              <div className="text-center">
                <PresentationChartBarIcon className="w-12 h-12 mx-auto mb-3 opacity-20" />
                <p className="font-medium">Your slides will appear here</p>
                <p className="text-[10px] mt-1 opacity-60">Enter a topic and click Generate</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
