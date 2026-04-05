import React, { useState, useRef, useEffect, useCallback, Children } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  MagnifyingGlassIcon,
  ClipboardDocumentIcon,
  ClockIcon,
  StopIcon,
  CheckIcon,
  ArrowDownTrayIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
type Depth = 'quick' | 'standard' | 'deep';

interface ResearchSource {
  title: string;
  url: string;
  snippet: string;
  score?: number;
}

interface HistoryEntry {
  topic: string;
  depth: Depth;
  report: string;
  sources: ResearchSource[];
  timestamp: number;
}

/* ── Constants ── */
const DEPTH_OPTIONS: { value: Depth; label: string; description: string }[] = [
  { value: 'quick', label: 'Quick', description: '1–2 paragraphs' },
  { value: 'standard', label: 'Standard', description: 'Full analysis' },
  { value: 'deep', label: 'Deep', description: 'Comprehensive report' },
];

const SYSTEM_PROMPTS: Record<Depth, string> = {
  quick: 'Provide a brief, focused answer to this research question in 1-2 paragraphs. Cite sources using [1], [2], etc. where supported by the search results below.',
  standard:
    'Conduct a thorough analysis of this topic using the search results below. Include: overview, key findings, different perspectives, and a conclusion. Use headers and bullet points. Cite sources using [1], [2], etc.',
  deep:
    'Write a comprehensive research report on this topic using the search results below. Include: executive summary, background, detailed analysis with multiple perspectives, supporting evidence, counterarguments, implications, and conclusion. Cite sources using [1], [2], etc. throughout.',
};

/* ── Helpers ── */
function formatTimestamp(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + '…' : s;
}

/* ── Citation linker ── */
const citationRegex = /\[(\d+)\]/g;
function renderWithCitations(text: string, sourcesLen: number): React.ReactNode {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let match;
  citationRegex.lastIndex = 0;
  while ((match = citationRegex.exec(text)) !== null) {
    const num = parseInt(match[1], 10);
    if (num >= 1 && num <= sourcesLen) {
      if (match.index > last) parts.push(text.slice(last, match.index));
      parts.push(
        <a
          key={match.index}
          href={`#source-${num}`}
          onClick={(e) => { e.preventDefault(); e.stopPropagation(); document.getElementById(`source-${num}`)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }}
          className="text-[10px] font-bold text-chat-accent hover:text-purple-300 cursor-pointer no-underline bg-chat-accent/10 px-0.5 rounded align-super"
        >[{num}]</a>
      );
      last = match.index + match[0].length;
    }
  }
  if (last === 0) return text;
  parts.push(text.slice(last));
  return <>{parts}</>;
}

/* ── Main Component ── */
export function ResearchPanel() {
  const [topic, setTopic] = useState('');
  const [depth, setDepth] = useState<Depth>('standard');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isFetchingSources, setIsFetchingSources] = useState(false);
  const [streamingReport, setStreamingReport] = useState('');
  const [finalReport, setFinalReport] = useState('');
  const [sources, setSources] = useState<ResearchSource[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [activeHistoryIdx, setActiveHistoryIdx] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const reportEndRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Fetch models
  useEffect(() => {
    fetch('/api/models')
      .then((r) => r.json())
      .then((data) => {
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

  // Auto-scroll while streaming
  useEffect(() => {
    if (isGenerating && reportEndRef.current) {
      reportEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamingReport, isGenerating]);

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const handleResearch = useCallback(async () => {
    const trimmed = topic.trim();
    if (!trimmed || isGenerating) return;

    setIsGenerating(true);
    setIsFetchingSources(true);
    setError(null);
    setFinalReport('');
    setStreamingReport('');
    setSources([]);
    setActiveHistoryIdx(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // Step 1: Fetch real web search results
      const searchLimit = depth === 'deep' ? 10 : depth === 'standard' ? 8 : 5;
      let fetchedSources: ResearchSource[] = [];
      try {
        const searchRes = await fetch(
          `/api/search/results?q=${encodeURIComponent(trimmed)}&limit=${searchLimit}`,
          { signal: controller.signal }
        );
        if (searchRes.ok) {
          const searchData = await searchRes.json();
          fetchedSources = searchData.results || [];
          setSources(fetchedSources);
        }
      } catch (e: any) {
        if (e.name === 'AbortError') throw e;
        // Search failed — continue without sources
      } finally {
        setIsFetchingSources(false);
      }

      // Step 2: Build system prompt with real search context
      let systemPrompt = SYSTEM_PROMPTS[depth];
      if (fetchedSources.length > 0) {
        const context = fetchedSources.map((r, i) =>
          `[${i + 1}] ${r.title}\nURL: ${r.url}\n${r.snippet}`
        ).join('\n\n');
        systemPrompt += `\n\nSearch results for "${trimmed}":\n\n${context}`;
      } else {
        systemPrompt += '\n\nNo web search results were available. Use your knowledge and clearly state that no live sources were retrieved.';
      }

      // Step 3: Stream LLM report
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          system_prompt: systemPrompt,
          history: [],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullText = '';

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
                  fullText += text;
                  setStreamingReport(fullText);
                }
              } catch {
                // non-JSON line — skip
              }
            }
          }
        }
      } else {
        fullText = await res.text();
      }

      setFinalReport(fullText);
      setStreamingReport('');

      // Save to history (max 5)
      const entry: HistoryEntry = {
        topic: trimmed,
        depth,
        report: fullText,
        sources: fetchedSources,
        timestamp: Date.now(),
      };
      setHistory((prev) => [entry, ...prev].slice(0, 5));
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Something went wrong. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      setIsFetchingSources(false);
      abortRef.current = null;
    }
  }, [topic, depth, isGenerating, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    // Preserve whatever was streamed so far
    setFinalReport(streamingReport);
    setStreamingReport('');
    setIsGenerating(false);
  }, [streamingReport]);

  const handleExport = useCallback(() => {
    const reportText = activeHistoryIdx !== null
      ? history[activeHistoryIdx]?.report
      : finalReport;
    if (!reportText) return;

    navigator.clipboard.writeText(reportText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch(() => {
      // Fallback for browsers without clipboard API
      const ta = document.createElement('textarea');
      ta.value = reportText;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [finalReport, history, activeHistoryIdx]);

  const handleDownloadMarkdown = useCallback(() => {
    const reportText = activeHistoryIdx !== null ? history[activeHistoryIdx]?.report : finalReport;
    if (!reportText) return;
    const activeSources = activeHistoryIdx !== null ? history[activeHistoryIdx]?.sources ?? [] : sources;
    const activeTopic = activeHistoryIdx !== null ? history[activeHistoryIdx]?.topic : topic;
    const sourcesSection = activeSources.length > 0
      ? '\n\n---\n\n## Sources\n\n' + activeSources.map((s, i) =>
          `[${i + 1}] **${s.title || s.url}**  \n${s.url}${s.snippet ? `  \n> ${s.snippet}` : ''}`
        ).join('\n\n')
      : '';
    const fullMd = `# ${activeTopic}\n\n${reportText}${sourcesSection}`;
    const blob = new Blob([fullMd], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-${(activeTopic || 'report').slice(0, 30).replace(/\s+/g, '-').toLowerCase()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }, [finalReport, sources, topic, history, activeHistoryIdx]);

  const loadHistoryEntry = useCallback((idx: number) => {
    const entry = history[idx];
    if (!entry) return;
    setTopic(entry.topic);
    setDepth(entry.depth);
    setFinalReport(entry.report);
    setSources(entry.sources || []);
    setStreamingReport('');
    setError(null);
    setActiveHistoryIdx(idx);
  }, [history]);

  const displayReport = isGenerating ? streamingReport : finalReport;
  const hasReport = displayReport.length > 0;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Controls + History */}
      <div className="flex flex-col md:w-[340px] md:min-w-[280px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0">
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Research</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Deep analysis on any topic</p>
        </div>

        {/* Controls */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Topic input */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-chat-text-secondary uppercase tracking-wide">
              Research Topic
            </label>
            <textarea
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                  e.preventDefault();
                  handleResearch();
                }
              }}
              placeholder="What do you want to research? (Ctrl+Enter to run)"
              className="w-full p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 leading-relaxed"
              rows={4}
              disabled={isGenerating}
            />
          </div>

          {/* Depth selector */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-chat-text-secondary uppercase tracking-wide">
              Depth
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {DEPTH_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setDepth(opt.value)}
                  disabled={isGenerating}
                  className={`flex flex-col items-center px-2 py-2 rounded-lg border text-xs transition-all disabled:opacity-40 ${
                    depth === opt.value
                      ? 'border-chat-accent bg-chat-accent/10 text-chat-text'
                      : 'border-chat-border bg-surface-1 text-chat-text-secondary hover:border-chat-accent/50 hover:text-chat-text'
                  }`}
                >
                  <span className="font-medium">{opt.label}</span>
                  <span className="text-[10px] opacity-70 mt-0.5">{opt.description}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Model selector */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-chat-text-secondary uppercase tracking-wide">
              Model
            </label>
            <div ref={modelMenuRef} className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu((p) => !p)}
                disabled={isGenerating}
                className="w-full flex items-center justify-between gap-1 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-3 py-2 rounded-lg border border-chat-border bg-surface-1 disabled:opacity-40"
              >
                <span className="truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}</span>
                <svg className="w-3 h-3 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showModelMenu && availableModels.length > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    top: '100%',
                    left: 0,
                    right: 0,
                    marginTop: 4,
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    borderRadius: 10,
                    overflow: 'hidden',
                    zIndex: 50,
                  }}
                >
                  <div style={{ maxHeight: 220, overflowY: 'auto', padding: 4 }}>
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
                    {availableModels.map((m) => (
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

          {/* Run button */}
          <button
            onClick={isGenerating ? handleStop : handleResearch}
            disabled={!isGenerating && !topic.trim()}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all disabled:opacity-40"
            style={{
              background: isGenerating
                ? 'var(--surface-3)'
                : 'var(--accent)',
              color: 'white',
            }}
          >
            {isGenerating ? (
              <>
                <StopIcon className="w-4 h-4" />
                Stop
              </>
            ) : (
              <>
                <MagnifyingGlassIcon className="w-4 h-4" />
                Research
              </>
            )}
          </button>

          {/* Research history */}
          {history.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-chat-text-secondary uppercase tracking-wide">
                <ClockIcon className="w-3 h-3" />
                Recent
              </div>
              <div className="space-y-1">
                {history.map((entry, idx) => (
                  <button
                    key={entry.timestamp}
                    onClick={() => loadHistoryEntry(idx)}
                    className={`w-full text-left px-2.5 py-2 rounded-lg border text-xs transition-all ${
                      activeHistoryIdx === idx
                        ? 'border-chat-accent/50 bg-chat-accent/10 text-chat-text'
                        : 'border-chat-border bg-surface-1 text-chat-text-secondary hover:text-chat-text hover:border-chat-border'
                    }`}
                  >
                    <div className="font-medium truncate">{truncate(entry.topic, 48)}</div>
                    <div className="flex items-center gap-1.5 mt-0.5 opacity-60">
                      <span className="capitalize">{entry.depth}</span>
                      <span>·</span>
                      <span>{formatTimestamp(entry.timestamp)}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Right: Report area */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Toolbar */}
        <div className="flex items-center justify-between px-4 py-2 border-b border-chat-border flex-shrink-0">
          <div className="text-xs text-chat-text-secondary">
            {isGenerating ? (
              <span className="flex items-center gap-2 text-purple-400">
                <span className="inline-block w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
                {isFetchingSources ? 'Searching the web…' : `Researching${depth === 'deep' ? ' (deep mode)' : ''}…`}
              </span>
            ) : hasReport ? (
              <span>
                {Math.round(displayReport.length / 1000)}k chars
                {sources.length > 0 && <span className="ml-2 opacity-60">· {sources.length} sources</span>}
              </span>
            ) : (
              <span>Results will appear here</span>
            )}
          </div>
          {hasReport && (
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
              title="Copy report as Markdown"
            >
              {copied ? (
                <>
                  <CheckIcon className="w-3.5 h-3.5 text-green-400" />
                  <span className="text-green-400">Copied!</span>
                </>
              ) : (
                <>
                  <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                  Export MD
                </>
              )}
            </button>
          )}
          {hasReport && (
            <button
              onClick={handleDownloadMarkdown}
              className="flex items-center gap-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
              title="Download as Markdown file"
            >
              <ArrowDownTrayIcon className="w-3.5 h-3.5" />
              Download
            </button>
          )}
        </div>

        {/* Report content */}
        <div className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="mb-4 px-4 py-3 rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 text-sm">
              {error}
            </div>
          )}

          {!hasReport && !error && !isGenerating && (
            <div className="h-full flex items-center justify-center text-chat-text-secondary">
              <div className="text-center space-y-2">
                <MagnifyingGlassIcon className="w-10 h-10 mx-auto opacity-20" />
                <p className="text-sm">Enter a topic and press Research</p>
                <p className="text-[11px] opacity-60">Supports any question, topic, or analysis task</p>
              </div>
            </div>
          )}

          {hasReport && (
            <div className="max-w-3xl mx-auto">
              {/* Report header */}
              {(finalReport || streamingReport) && topic && (
                <div className="mb-6 pb-4 border-b border-chat-border">
                  <div className="flex items-center gap-2 mb-1">
                    <span
                      className="text-[10px] font-medium px-2 py-0.5 rounded-full uppercase tracking-wide"
                      style={{ background: 'var(--surface-2)', color: 'var(--text-secondary)' }}
                    >
                      {depth}
                    </span>
                    {isGenerating && (
                      <span className="text-[10px] text-purple-400 animate-pulse">Generating…</span>
                    )}
                  </div>
                  <h1 className="text-lg font-semibold text-chat-text leading-snug">{topic}</h1>
                </div>
              )}

              {/* Report body */}
              <div className="text-sm text-chat-text leading-relaxed
                [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-chat-text [&_h1]:mt-4 [&_h1]:mb-2
                [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-chat-text [&_h2]:mt-3 [&_h2]:mb-1.5
                [&_h3]:text-xs [&_h3]:font-semibold [&_h3]:text-chat-text-secondary [&_h3]:mt-2 [&_h3]:mb-1
                [&_ul]:list-disc [&_ul]:pl-4 [&_ul]:space-y-0.5 [&_ol]:list-decimal [&_ol]:pl-4 [&_ol]:space-y-0.5
                [&_li]:text-sm [&_strong]:font-semibold [&_strong]:text-chat-text
                [&_code]:bg-surface-2 [&_code]:px-1 [&_code]:rounded [&_code]:text-xs [&_code]:font-mono
                [&_blockquote]:border-l-2 [&_blockquote]:border-chat-accent [&_blockquote]:pl-3 [&_blockquote]:text-chat-text-secondary [&_blockquote]:italic
                [&_a]:text-chat-accent [&_a]:underline [&_a]:underline-offset-2 [&_p]:mb-2
              ">
                {isGenerating ? (
                  <span className="whitespace-pre-wrap">
                    {displayReport}
                    <span className="inline-block w-1.5 h-4 bg-purple-400 animate-pulse ml-0.5 align-middle rounded-sm" />
                  </span>
                ) : (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      p: ({ children: c }) => (
                        <p className="mb-2">
                          {Children.map(c, (child) =>
                            typeof child === 'string' ? renderWithCitations(child, sources.length) : child
                          )}
                        </p>
                      ),
                    }}
                  >
                    {displayReport}
                  </ReactMarkdown>
                )}
              </div>
              <div ref={reportEndRef} />

              {/* Sources section */}
              {sources.length > 0 && !isGenerating && (
                <div className="mt-8 pt-6 border-t border-chat-border">
                  <p className="text-[11px] font-medium text-chat-text-secondary uppercase tracking-wide mb-3">
                    Sources ({sources.length})
                  </p>
                  <div className="space-y-2">
                    {sources.map((src, i) => (
                      <a
                        key={i}
                        id={`source-${i + 1}`}
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-3 rounded-lg border border-chat-border bg-surface-1 px-3 py-2.5 hover:border-chat-accent/50 transition-colors group scroll-mt-4"
                      >
                        <span className="flex-shrink-0 text-[10px] font-bold text-chat-accent w-4 mt-0.5">
                          [{i + 1}]
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="text-xs font-medium text-chat-text truncate group-hover:text-chat-accent transition-colors">
                            {src.title || src.url}
                          </p>
                          <p className="text-[10px] text-chat-text-secondary truncate mt-0.5">{src.url}</p>
                          {src.snippet && (
                            <p className="text-[11px] text-chat-text-secondary mt-1 line-clamp-2 leading-relaxed">
                              {src.snippet}
                            </p>
                          )}
                          {typeof src.score === 'number' && (
                            <div className="flex items-center gap-1.5 mt-1.5">
                              <div className="flex-1 h-1 rounded-full bg-surface-2 overflow-hidden">
                                <div className="h-full rounded-full bg-gradient-to-r from-purple-500 to-blue-400" style={{ width: `${Math.min(100, Math.round(src.score * 100))}%` }} />
                              </div>
                              <span className="text-[10px] text-chat-text-secondary">{Math.round(src.score * 100)}%</span>
                            </div>
                          )}
                        </div>
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
