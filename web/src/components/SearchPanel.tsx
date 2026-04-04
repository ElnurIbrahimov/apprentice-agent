import { useState, useRef, useEffect, useCallback } from 'react';
import {
  MagnifyingGlassIcon,
  StopIcon,
  ClockIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

const SEARCH_SYSTEM_PROMPT_BASE = `You are a web search assistant. Answer the user's query using ONLY the web search results provided below. Rules:
- Cite sources inline using [1], [2], etc. matching the result numbers
- Be accurate and factual — only state things supported by the results
- Use headers and bullet points for readability
- If the results don't fully answer the query, say so clearly
- Do NOT fabricate information or URLs`;

const QUICK_SUGGESTIONS = [
  'Latest AI news today',
  'How does quantum computing work',
  'Top programming languages 2025',
  'Climate change recent updates',
  'Best practices for REST APIs',
  'Space exploration milestones',
];

interface SearchSource {
  title: string;
  url: string;
  snippet: string;
}

interface SearchEntry {
  query: string;
  result: string;
  sources: SearchSource[];
  timestamp: number;
}

/* ── Simple markdown-like renderer ── */
function FormattedResult({ text }: { text: string }) {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={i} className="text-sm font-semibold text-chat-text mt-4 mb-1">
          {line.slice(4)}
        </h3>
      );
    } else if (line.startsWith('## ')) {
      elements.push(
        <h2 key={i} className="text-base font-semibold text-chat-text mt-5 mb-1.5">
          {line.slice(3)}
        </h2>
      );
    } else if (line.startsWith('# ')) {
      elements.push(
        <h1 key={i} className="text-lg font-bold text-chat-text mt-5 mb-2">
          {line.slice(2)}
        </h1>
      );
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(
        <li key={i} className="ml-4 text-sm text-chat-text leading-relaxed list-disc">
          {line.slice(2)}
        </li>
      );
    } else if (/^\d+\. /.test(line)) {
      elements.push(
        <li key={i} className="ml-4 text-sm text-chat-text leading-relaxed list-decimal">
          {line.replace(/^\d+\. /, '')}
        </li>
      );
    } else if (line.startsWith('**') && line.endsWith('**') && line.length > 4) {
      elements.push(
        <p key={i} className="text-sm font-semibold text-chat-text mt-2">
          {line.slice(2, -2)}
        </p>
      );
    } else if (line.trim() === '') {
      elements.push(<div key={i} className="h-2" />);
    } else {
      // Inline bold: **text**
      const parts = line.split(/(\*\*[^*]+\*\*)/g);
      const rendered = parts.map((part, j) =>
        part.startsWith('**') && part.endsWith('**')
          ? <strong key={j}>{part.slice(2, -2)}</strong>
          : part
      );
      elements.push(
        <p key={i} className="text-sm text-chat-text leading-relaxed">
          {rendered}
        </p>
      );
    }
  }

  return <div className="space-y-0.5">{elements}</div>;
}

/* ── Shimmer skeleton ── */
function ShimmerSkeleton() {
  return (
    <div className="space-y-3 animate-pulse">
      <div className="h-4 rounded bg-surface-2 w-3/4" />
      <div className="h-4 rounded bg-surface-2 w-full" />
      <div className="h-4 rounded bg-surface-2 w-5/6" />
      <div className="h-4 rounded bg-surface-2 w-2/3 mt-4" />
      <div className="h-4 rounded bg-surface-2 w-full" />
      <div className="h-4 rounded bg-surface-2 w-4/5" />
      <div className="h-4 rounded bg-surface-2 w-full" />
      <div className="h-4 rounded bg-surface-2 w-3/5" />
    </div>
  );
}

/* ── Main Component ── */
export function SearchPanel() {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');
  const [sources, setSources] = useState<SearchSource[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isFetchingSources, setIsFetchingSources] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<SearchEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [currentQuery, setCurrentQuery] = useState('');

  const abortRef = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Fetch models
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

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Auto-scroll result area during streaming
  useEffect(() => {
    if (isSearching && resultRef.current) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [result, isSearching]);

  const handleSearch = useCallback(async (searchQuery: string) => {
    const q = searchQuery.trim();
    if (!q || isSearching) return;

    setCurrentQuery(q);
    setResult('');
    setSources([]);
    setError(null);
    setIsSearching(true);
    setIsFetchingSources(true);
    setShowHistory(false);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // Step 1: Fetch real web search results
      let fetchedSources: SearchSource[] = [];
      try {
        const searchRes = await fetch(`/api/search/results?q=${encodeURIComponent(q)}&limit=8`, {
          signal: controller.signal,
        });
        if (searchRes.ok) {
          const searchData = await searchRes.json();
          fetchedSources = searchData.results || [];
          setSources(fetchedSources);
        }
      } catch (e: any) {
        if (e.name === 'AbortError') throw e;
        // Search fetch failed — continue with empty sources, LLM will note no results
      } finally {
        setIsFetchingSources(false);
      }

      // Step 2: Build system prompt with real search context
      let systemPrompt = SEARCH_SYSTEM_PROMPT_BASE;
      if (fetchedSources.length > 0) {
        const context = fetchedSources.map((r, i) =>
          `[${i + 1}] ${r.title}\nURL: ${r.url}\n${r.snippet}`
        ).join('\n\n');
        systemPrompt += `\n\nSearch results for "${q}":\n\n${context}`;
      } else {
        systemPrompt += '\n\nNo web search results available. Answer from your knowledge and clearly state this.';
      }

      // Step 3: Stream LLM summarized response
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: q,
          system_prompt: systemPrompt,
          history: [],
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
                const text = parsed.content || parsed.choices?.[0]?.delta?.content || parsed.chunk || '';
                if (text) {
                  fullResponse += text;
                  setResult(fullResponse);
                }
              } catch {
                // non-JSON line — skip
              }
            }
          }
        }
      } else {
        fullResponse = await res.text();
        setResult(fullResponse);
      }

      // Save to history (last 10)
      if (fullResponse.trim()) {
        setHistory(prev => [
          { query: q, result: fullResponse, sources: fetchedSources, timestamp: Date.now() },
          ...prev,
        ].slice(0, 10));
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Search failed. Make sure the backend is running.');
      }
    } finally {
      setIsSearching(false);
      setIsFetchingSources(false);
      abortRef.current = null;
    }
  }, [isSearching, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsSearching(false);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearch(query);
  };

  const loadHistoryEntry = (entry: SearchEntry) => {
    setQuery(entry.query);
    setCurrentQuery(entry.query);
    setResult(entry.result);
    setSources(entry.sources || []);
    setError(null);
    setShowHistory(false);
  };

  const clearHistory = () => {
    setHistory([]);
    setShowHistory(false);
  };

  const hasResult = result.length > 0;

  return (
    <div className="h-full flex flex-col bg-surface-0 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Web Search</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">AI-powered web search with real-time summarization</p>
      </div>

      {/* Search bar */}
      <div className="px-4 pt-4 pb-2 flex-shrink-0">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-chat-text-secondary pointer-events-none" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search anything..."
              className="w-full pl-9 pr-3 py-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 transition-colors"
              disabled={isSearching}
              autoComplete="off"
            />
          </div>
          <button
            type={isSearching ? 'button' : 'submit'}
            onClick={isSearching ? handleStop : undefined}
            disabled={!isSearching && !query.trim()}
            className="px-3 py-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity flex items-center gap-1.5 text-sm font-medium"
          >
            {isSearching
              ? <><StopIcon className="w-4 h-4" /> Stop</>
              : <><MagnifyingGlassIcon className="w-4 h-4" /> Search</>
            }
          </button>
        </form>

        {/* Controls row */}
        <div className="flex items-center justify-between mt-2">
          {/* Model selector */}
          <div ref={modelMenuRef} className="relative">
            <button
              type="button"
              onClick={() => setShowModelMenu(p => !p)}
              className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
            >
              <span className="max-w-[140px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
              <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showModelMenu && availableModels.length > 0 && (
              <div
                style={{
                  position: 'absolute', top: 28, left: 0, width: 220, maxHeight: 280,
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

          {/* History toggle */}
          {history.length > 0 && (
            <button
              onClick={() => setShowHistory(p => !p)}
              className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
            >
              <ClockIcon className="w-3 h-3" />
              History ({history.length})
            </button>
          )}
        </div>
      </div>

      {/* History dropdown */}
      {showHistory && history.length > 0 && (
        <div className="mx-4 mb-2 rounded-lg border border-chat-border bg-surface-1 overflow-hidden flex-shrink-0">
          <div className="flex items-center justify-between px-3 py-1.5 border-b border-chat-border">
            <span className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">Recent searches</span>
            <button onClick={clearHistory} className="text-[10px] text-chat-text-secondary hover:text-red-400 transition-colors">
              Clear all
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto">
            {history.map((entry, i) => (
              <button
                key={i}
                onClick={() => loadHistoryEntry(entry)}
                className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-2 transition-colors border-b border-chat-border/50 last:border-0"
              >
                <ClockIcon className="w-3 h-3 text-chat-text-secondary flex-shrink-0" />
                <span className="text-xs text-chat-text truncate flex-1">{entry.query}</span>
                <span className="text-[10px] text-chat-text-secondary flex-shrink-0">
                  {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Quick suggestions — show only when no result */}
      {!hasResult && !isSearching && (
        <div className="px-4 pb-3 flex-shrink-0">
          <p className="text-[10px] text-chat-text-secondary mb-2 uppercase tracking-wide font-medium">Quick searches</p>
          <div className="flex flex-wrap gap-2">
            {QUICK_SUGGESTIONS.map((s) => (
              <button
                key={s}
                onClick={() => { setQuery(s); handleSearch(s); }}
                className="px-3 py-1.5 rounded-full text-xs border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-accent transition-colors bg-surface-1"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Results area */}
      <div ref={resultRef} className="flex-1 overflow-y-auto px-4 pb-6">
        {/* Active query label */}
        {(isSearching || hasResult) && currentQuery && (
          <div className="flex items-center gap-2 mb-4">
            <MagnifyingGlassIcon className="w-3.5 h-3.5 text-chat-accent flex-shrink-0" />
            <span className="text-xs font-medium text-chat-text truncate">{currentQuery}</span>
            {hasResult && !isSearching && (
              <button
                onClick={() => { setResult(''); setCurrentQuery(''); setError(null); inputRef.current?.focus(); }}
                className="ml-auto text-chat-text-secondary hover:text-chat-text flex-shrink-0"
                title="Clear result"
              >
                <XMarkIcon className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        )}

        {/* Loading shimmer */}
        {isSearching && !result && (
          <>
            {isFetchingSources && (
              <div className="flex items-center gap-2 mb-3 text-xs text-chat-text-secondary">
                <span className="inline-block w-2 h-2 rounded-full bg-chat-accent animate-pulse" />
                Searching the web…
              </div>
            )}
            {!isFetchingSources && (
              <div className="flex items-center gap-2 mb-3 text-xs text-chat-text-secondary">
                <span className="inline-block w-2 h-2 rounded-full bg-chat-accent animate-pulse" />
                Summarizing results…
              </div>
            )}
            <ShimmerSkeleton />
          </>
        )}

        {/* Streaming / final result */}
        {result && (
          <div className="rounded-lg border border-chat-border bg-surface-1 p-4">
            <FormattedResult text={result} />
            {isSearching && (
              <span className="inline-block w-1.5 h-3.5 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
            )}
          </div>
        )}

        {/* Source cards */}
        {sources.length > 0 && !isSearching && (
          <div className="mt-4">
            <p className="text-[10px] text-chat-text-secondary uppercase tracking-wide font-medium mb-2">Sources</p>
            <div className="space-y-1.5">
              {sources.map((src, i) => (
                <a
                  key={i}
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-2 rounded-lg border border-chat-border bg-surface-1 px-3 py-2 hover:border-chat-accent/50 transition-colors group"
                >
                  <span className="flex-shrink-0 w-4 h-4 mt-0.5 text-[10px] font-bold text-chat-accent flex items-center justify-center">
                    {i + 1}
                  </span>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-chat-text truncate group-hover:text-chat-accent transition-colors">
                      {src.title || src.url}
                    </p>
                    <p className="text-[10px] text-chat-text-secondary truncate">{src.url}</p>
                  </div>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-900/10 p-4 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Empty state */}
        {!hasResult && !isSearching && !error && (
          <div className="flex flex-col items-center justify-center h-40 text-chat-text-secondary text-sm gap-2">
            <MagnifyingGlassIcon className="w-8 h-8 opacity-30" />
            <p className="text-xs opacity-60">Results will appear here</p>
          </div>
        )}
      </div>
    </div>
  );
}
