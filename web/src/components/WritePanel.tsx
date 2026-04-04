import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ClipboardDocumentIcon,
  ArrowDownTrayIcon,
  TrashIcon,
  StopIcon,
  PaperAirplaneIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
type WriteFormat = 'Essay' | 'Email' | 'Blog Post' | 'Report' | 'Story' | 'Social Post' | 'Letter';
type WriteTone = 'Professional' | 'Casual' | 'Academic' | 'Creative' | 'Persuasive';

const FORMATS: WriteFormat[] = ['Essay', 'Email', 'Blog Post', 'Report', 'Story', 'Social Post', 'Letter'];
const TONES: WriteTone[] = ['Professional', 'Casual', 'Academic', 'Creative', 'Persuasive'];

function buildSystemPrompt(format: WriteFormat, tone: WriteTone): string {
  return `You are an expert writer. Write ${format} content in a ${tone} tone based on the user's description. Output ONLY the content, no meta-commentary. Write naturally and engagingly.`;
}

const DRAFT_KEY = 'aura-draft-write';

/* ── Main Component ── */
export function WritePanel() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [format, setFormat] = useState<WriteFormat>('Essay');
  const [tone, setTone] = useState<WriteTone>('Professional');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const outputRef = useRef<HTMLTextAreaElement>(null);

  // Load draft on mount
  useEffect(() => {
    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) {
        const parsed = JSON.parse(draft);
        if (parsed.input) setInput(parsed.input);
        if (parsed.output) setOutput(parsed.output);
        if (parsed.format) setFormat(parsed.format);
        if (parsed.tone) setTone(parsed.tone);
        setDraftLoaded(true);
      }
    } catch {}
  }, []);

  // Auto-save on change
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({ input, output, format, tone, timestamp: Date.now() }));
      } catch {}
    }, 5000);
    return () => clearTimeout(timer);
  }, [input, output, format, tone]);

  // Fetch models on mount
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

  // Close model menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const wordCount = output.trim() ? output.trim().split(/\s+/).length : 0;

  const handleGenerate = useCallback(async () => {
    if (!input.trim() || isGenerating) return;
    setError(null);
    setOutput('');
    setIsGenerating(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          system_prompt: buildSystemPrompt(format, tone),
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';

      if (res.body) {
        const reader = res.body.getReader();
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
                  setOutput(fullResponse);
                }
              } catch {
                fullResponse += data;
                setOutput(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setOutput(fullResponse);
            }
          }
        }
      } else {
        const text = await res.text();
        setOutput(text);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Generation failed. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [input, format, tone, selectedModel, isGenerating]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleCopy = useCallback(() => {
    if (!output) return;
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [output]);

  const handleDownload = useCallback(() => {
    if (!output) return;
    const blob = new Blob([output], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aura-${format.toLowerCase().replace(' ', '-')}-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [output, format]);

  const handleClear = useCallback(() => {
    setOutput('');
    setInput('');
    setError(null);
    setDraftLoaded(false);
    try { localStorage.removeItem(DRAFT_KEY); } catch {}
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Write</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">Describe what you want to write, or paste text to improve</p>
      </div>

      {/* Draft recovered banner */}
      {draftLoaded && (
        <div className="flex items-center gap-2 px-3 py-1.5 mx-4 mt-2 bg-blue-500/10 text-blue-400 text-[10px] rounded-lg flex-shrink-0">
          <span className="flex-1">Draft recovered</span>
          <button onClick={() => { setDraftLoaded(false); try { localStorage.removeItem(DRAFT_KEY); } catch {} }} className="hover:text-blue-300 transition-colors">Dismiss</button>
        </div>
      )}

      {/* Main layout */}
      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">

        {/* Left: Controls */}
        <div className="flex flex-col md:w-[360px] md:min-w-[280px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">

            {/* Input */}
            <div>
              <label className="block text-[10px] font-medium text-chat-text-secondary uppercase tracking-wider mb-1.5">
                Your prompt or text to improve
              </label>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    e.preventDefault();
                    handleGenerate();
                  }
                }}
                placeholder="Describe what you want to write, or paste existing text to rewrite/improve..."
                className="w-full p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 leading-relaxed"
                rows={6}
                disabled={isGenerating}
              />
            </div>

            {/* Format selector */}
            <div>
              <label className="block text-[10px] font-medium text-chat-text-secondary uppercase tracking-wider mb-1.5">
                Format
              </label>
              <div className="flex flex-wrap gap-1.5">
                {FORMATS.map(f => (
                  <button
                    key={f}
                    onClick={() => setFormat(f)}
                    className="px-2.5 py-1 rounded-md text-xs font-medium transition-all"
                    style={{
                      background: format === f ? 'var(--chat-accent)' : 'var(--surface-2)',
                      color: format === f ? '#fff' : 'var(--text-secondary)',
                      border: format === f ? '1px solid transparent' : '1px solid var(--border-default)',
                    }}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            {/* Tone selector */}
            <div>
              <label className="block text-[10px] font-medium text-chat-text-secondary uppercase tracking-wider mb-1.5">
                Tone
              </label>
              <div className="flex flex-wrap gap-1.5">
                {TONES.map(t => (
                  <button
                    key={t}
                    onClick={() => setTone(t)}
                    className="px-2.5 py-1 rounded-md text-xs font-medium transition-all"
                    style={{
                      background: tone === t ? 'var(--chat-accent)' : 'var(--surface-2)',
                      color: tone === t ? '#fff' : 'var(--text-secondary)',
                      border: tone === t ? '1px solid transparent' : '1px solid var(--border-default)',
                    }}
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>

            {/* Model selector */}
            <div>
              <label className="block text-[10px] font-medium text-chat-text-secondary uppercase tracking-wider mb-1.5">
                Model
              </label>
              <div ref={modelMenuRef} className="relative inline-block">
                <button
                  type="button"
                  onClick={() => setShowModelMenu(p => !p)}
                  className="flex items-center gap-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2.5 py-1.5 rounded-md"
                  style={{ background: 'var(--border-subtle)', border: '1px solid var(--border-default)' }}
                >
                  <span className="max-w-[180px] truncate">
                    {selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}
                  </span>
                  <svg className="w-3 h-3 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {showModelMenu && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '100%',
                      left: 0,
                      marginTop: 4,
                      width: 240,
                      maxHeight: 260,
                      background: 'var(--surface-1)',
                      border: '1px solid var(--border-default)',
                      borderRadius: 10,
                      overflow: 'hidden',
                      zIndex: 50,
                      boxShadow: '0 8px 24px rgba(0,0,0,0.3)',
                    }}
                  >
                    <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
                      <button
                        onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                        style={{
                          color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
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
                            color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
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

          {/* Generate button */}
          <div className="p-4 border-t border-chat-border flex-shrink-0">
            <button
              onClick={isGenerating ? handleStop : handleGenerate}
              disabled={!isGenerating && !input.trim()}
              className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg text-sm font-medium transition-all disabled:opacity-40"
              style={{ background: 'var(--chat-accent)', color: '#fff' }}
            >
              {isGenerating ? (
                <>
                  <StopIcon className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <PaperAirplaneIcon className="w-4 h-4" />
                  Generate
                  <span className="text-[10px] opacity-60 ml-1">Ctrl+Enter</span>
                </>
              )}
            </button>

            {error && (
              <p className="mt-2 text-[11px] text-red-400 text-center">{error}</p>
            )}
          </div>
        </div>

        {/* Right: Output */}
        <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
          {/* Output toolbar */}
          <div className="flex items-center gap-2 px-3 py-2 border-b border-chat-border flex-shrink-0">
            <span className="text-[10px] text-chat-text-secondary flex-1">
              {isGenerating
                ? <span className="text-purple-400 flex items-center gap-1.5">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
                    Writing...
                  </span>
                : output
                  ? `${wordCount} word${wordCount !== 1 ? 's' : ''}`
                  : 'Output will appear here'
              }
            </span>
            <button
              onClick={handleCopy}
              disabled={!output}
              title="Copy to clipboard"
              className="p-1.5 rounded-md text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            >
              {copied
                ? <span className="text-[10px] text-green-400 font-medium">Copied!</span>
                : <ClipboardDocumentIcon className="w-4 h-4" />
              }
            </button>
            <button
              onClick={handleDownload}
              disabled={!output}
              title="Download as .txt"
              className="p-1.5 rounded-md text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
            </button>
            <button
              onClick={handleClear}
              disabled={!output && !input}
              title="Clear all"
              className="p-1.5 rounded-md text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            >
              <TrashIcon className="w-4 h-4" />
            </button>
          </div>

          {/* Editable output area */}
          <div className="flex-1 overflow-hidden p-3">
            {isGenerating && !output ? (
              <div className="flex items-start gap-2 p-1">
                <div className="shimmer-bar h-2 w-32 mt-1" />
              </div>
            ) : (
              <textarea
                ref={outputRef}
                value={output}
                onChange={e => setOutput(e.target.value)}
                placeholder="Generated content will appear here. You can edit it directly after generation."
                className="w-full h-full resize-none bg-transparent text-chat-text text-sm leading-relaxed outline-none placeholder-chat-text-secondary/40 font-[inherit]"
                style={{ fontFamily: 'inherit' }}
                spellCheck
              />
            )}
            {/* Streaming cursor */}
            {isGenerating && output && (
              <span className="inline-block w-1.5 h-3.5 bg-purple-400 animate-pulse ml-0.5 align-middle" />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
