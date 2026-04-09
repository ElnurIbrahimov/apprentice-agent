import { useState, useRef, useEffect, useCallback } from 'react';
import {
  MagnifyingGlassIcon,
  PaperAirplaneIcon,
  PlusIcon,
  ArrowPathIcon,
  BookOpenIcon,
  ChatBubbleLeftEllipsisIcon,
  StopIcon,
  ClockIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
type Tab = 'browse' | 'ask' | 'add';

interface MemoryEntry {
  id?: string;
  content: string;
  timestamp?: string | number;
  tags?: string[];
  category?: string;
  relevance?: number;
}

const QA_SYSTEM_PROMPT =
  "You are AURA's knowledge assistant. Answer the user's question based on your stored knowledge and memory. Be specific and cite what you remember. If you don't have stored knowledge about this, say so clearly.";

const ADD_SYSTEM_PROMPT =
  "The user wants to store this information. Acknowledge it and confirm what was noted. Rephrase it concisely for storage.";

/* ── Helpers ── */
function formatTime(ts: string | number | undefined): string {
  if (!ts) return '';
  const d = new Date(typeof ts === 'number' ? ts : ts);
  if (isNaN(d.getTime())) return '';
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/* ── Component ── */
export function WisebasePanel() {
  const [tab, setTab] = useState<Tab>('browse');

  /* Browse state */
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [memoriesLoading, setMemoriesLoading] = useState(false);
  const [memoriesError, setMemoriesError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<MemoryEntry[]>([]);
  const [searching, setSearching] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  /* Q&A state */
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [qaError, setQaError] = useState('');

  /* Add knowledge state */
  const [newFact, setNewFact] = useState('');
  const [addConfirmation, setAddConfirmation] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [addError, setAddError] = useState('');
  const [addSuccess, setAddSuccess] = useState(false);

  /* Model selector */
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  const abortRef = useRef<AbortController | null>(null);
  const answerScrollRef = useRef<HTMLDivElement>(null);

  /* ── Fetch models ── */
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

  /* ── Auto-scroll answer ── */
  useEffect(() => {
    if (isStreaming && answerScrollRef.current) {
      answerScrollRef.current.scrollTop = answerScrollRef.current.scrollHeight;
    }
  }, [answer, isStreaming]);

  /* ── Abort on unmount ── */
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  /* ── Fetch recent memories ── */
  const fetchMemories = useCallback(async () => {
    setMemoriesLoading(true);
    setMemoriesError('');
    try {
      const res = await fetch('/api/memory/recent');
      if (res.status === 404) {
        setMemories([]);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const entries: MemoryEntry[] = (data.memories || data.results || data || []).map((m: any) =>
        typeof m === 'string' ? { content: m } : m
      );
      setMemories(entries);
    } catch (e: any) {
      setMemoriesError('Could not load memories.');
    } finally {
      setMemoriesLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  /* ── Search memories ── */
  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setHasSearched(true);
    setSearchResults([]);
    try {
      const res = await fetch(`/api/memory/search?q=${encodeURIComponent(searchQuery.trim())}`);
      if (res.status === 404) {
        // Endpoint doesn't exist — fall through to empty results
        setSearchResults([]);
        return;
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const entries: MemoryEntry[] = (data.results || data.memories || data || []).map((m: any) =>
        typeof m === 'string' ? { content: m } : m
      );
      setSearchResults(entries);
    } catch {
      setSearchResults([{ content: '⚠ Search failed — check connection', type: 'error' } as any]);
    } finally {
      setSearching(false);
    }
  }, [searchQuery]);

  /* ── Streaming helper (shared by Q&A and Add) ── */
  const streamGenerate = useCallback(
    async (
      message: string,
      systemPrompt: string,
      onChunk: (chunk: string) => void,
      onDone: (full: string) => void,
      onError: (msg: string) => void
    ) => {
      const controller = new AbortController();
      abortRef.current = controller;
      let fullResponse = '';

      try {
        const res = await fetch('/api/generate/raw', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message,
            system_prompt: systemPrompt,
            ...(selectedModel ? { model: selectedModel } : {}),
          }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`API error: ${res.status}`);

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
                  const text =
                    parsed.choices?.[0]?.delta?.content ||
                    parsed.content ||
                    parsed.chunk ||
                    '';
                  if (text) {
                    fullResponse += text;
                    onChunk(fullResponse);
                  }
                } catch {
                  fullResponse += data;
                  onChunk(fullResponse);
                }
              } else if (line.trim() && !line.startsWith(':')) {
                fullResponse += line;
                onChunk(fullResponse);
              }
            }
          }
        } else {
          fullResponse = await res.text();
          onChunk(fullResponse);
        }

        onDone(fullResponse);
      } catch (e: any) {
        if (e.name !== 'AbortError') {
          onError(e.message || 'Request failed');
        }
      } finally {
        abortRef.current = null;
      }
    },
    [selectedModel]
  );

  /* ── Q&A ask ── */
  const handleAsk = useCallback(async () => {
    if (!question.trim() || isStreaming) return;
    setAnswer('');
    setQaError('');
    setIsStreaming(true);

    await streamGenerate(
      question,
      QA_SYSTEM_PROMPT,
      (partial) => setAnswer(partial),
      (_full) => setIsStreaming(false),
      (err) => {
        setQaError(`Error: ${err}`);
        setIsStreaming(false);
      }
    );
  }, [question, isStreaming, streamGenerate]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsStreaming(false);
    setIsAdding(false);
  }, []);

  /* ── Add knowledge ── */
  const handleAdd = useCallback(async () => {
    if (!newFact.trim() || isAdding) return;
    setAddConfirmation('');
    setAddError('');
    setAddSuccess(false);
    setIsAdding(true);

    // Try to POST to memory endpoint first
    let storedViaApi = false;
    try {
      const res = await fetch('/api/memory/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: newFact.trim() }),
      });
      if (res.ok) {
        storedViaApi = true;
        fetchMemories();
      }
    } catch (e) {
      console.warn('[Wisebase] Memory API unavailable:', e);
    }

    // Get AI confirmation/rephrasing regardless
    await streamGenerate(
      newFact,
      ADD_SYSTEM_PROMPT,
      (partial) => setAddConfirmation(partial),
      (_full) => {
        setIsAdding(false);
        setAddSuccess(storedViaApi);
        if (storedViaApi) setNewFact('');
      },
      (err) => {
        setAddError(`Error: ${err}`);
        setIsAdding(false);
      }
    );
  }, [newFact, isAdding, streamGenerate, fetchMemories]);

  /* ── Render ── */
  const displayList = hasSearched ? searchResults : memories;

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-chat-text flex items-center gap-1.5">
              <BookOpenIcon className="w-4 h-4 text-indigo-400" />
              Wisebase
            </h2>
            <p className="text-[10px] text-chat-text-secondary mt-0.5">
              AURA's knowledge — browse, search, ask, and add
            </p>
          </div>
          {/* Model selector */}
          <div ref={modelMenuRef} className="relative">
            <button
              type="button"
              onClick={() => setShowModelMenu((p) => !p)}
              className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
            >
              <span className="max-w-[120px] truncate">
                {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
              </span>
              <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showModelMenu && (
              <div
                style={{
                  position: 'absolute',
                  top: 28,
                  right: 0,
                  width: 220,
                  maxHeight: 280,
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-default)',
                  borderRadius: 10,
                  overflow: 'hidden',
                  zIndex: 50,
                }}
              >
                <div style={{ maxHeight: 280, overflowY: 'auto', padding: 4 }}>
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

        {/* Tabs */}
        <div className="flex gap-0.5 mt-3">
          {(['browse', 'ask', 'add'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1 text-xs rounded-md capitalize transition-colors ${
                tab === t
                  ? 'bg-indigo-600 text-white'
                  : 'text-chat-text-secondary hover:text-chat-text hover:bg-surface-1'
              }`}
            >
              {t === 'browse' && 'Browse'}
              {t === 'ask' && 'Ask'}
              {t === 'add' && '+ Add'}
            </button>
          ))}
        </div>
      </div>

      {/* Tab: Browse */}
      {tab === 'browse' && (
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Search bar */}
          <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
            <div className="flex gap-2">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (!e.target.value.trim()) {
                    setHasSearched(false);
                    setSearchResults([]);
                  }
                }}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                placeholder="Search stored memories..."
                className="flex-1 px-3 py-2 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-xs outline-none focus:border-indigo-500 placeholder-chat-text-secondary/50"
              />
              <button
                onClick={handleSearch}
                disabled={searching || !searchQuery.trim()}
                className="p-2 rounded-lg bg-indigo-600 hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
              >
                {searching ? (
                  <ArrowPathIcon className="w-4 h-4 animate-spin" />
                ) : (
                  <MagnifyingGlassIcon className="w-4 h-4" />
                )}
              </button>
              {hasSearched && (
                <button
                  onClick={() => { setHasSearched(false); setSearchResults([]); setSearchQuery(''); }}
                  className="p-2 rounded-lg bg-surface-1 border border-chat-border text-chat-text-secondary hover:text-chat-text text-xs transition-colors"
                  title="Clear search"
                >
                  ✕
                </button>
              )}
            </div>
            {hasSearched && (
              <p className="text-[10px] text-chat-text-secondary mt-1.5">
                {searchResults.length > 0
                  ? `${searchResults.length} result${searchResults.length !== 1 ? 's' : ''} for "${searchQuery}"`
                  : `No results found for "${searchQuery}"`}
              </p>
            )}
          </div>

          {/* Memory list */}
          <div className="flex-1 overflow-y-auto p-4 space-y-2">
            {/* Loading */}
            {memoriesLoading && !hasSearched && (
              <div className="flex items-center justify-center py-8 gap-2 text-chat-text-secondary text-xs">
                <ArrowPathIcon className="w-4 h-4 animate-spin" />
                Loading memories...
              </div>
            )}

            {/* Error */}
            {memoriesError && !hasSearched && (
              <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
                {memoriesError}
                <button onClick={fetchMemories} className="ml-2 underline hover:no-underline">
                  Retry
                </button>
              </div>
            )}

            {/* Empty state */}
            {!memoriesLoading && !memoriesError && displayList.length === 0 && (
              <div className="empty-state">
                <BookOpenIcon className="empty-state-icon" />
                <p className="empty-state-title">No knowledge stored yet</p>
                <p className="empty-state-desc">Add facts, notes, or documents to build your knowledge base</p>
              </div>
            )}

            {/* Entries */}
            {displayList.map((entry, i) => (
              <div
                key={entry.id ?? i}
                className="rounded-lg p-3 bg-surface-1 border border-chat-border hover:border-indigo-500/30 transition-colors"
              >
                <p className="text-xs text-chat-text leading-relaxed">{entry.content}</p>
                <div className="flex items-center gap-2 mt-1.5 flex-wrap">
                  {entry.timestamp && (
                    <span className="flex items-center gap-0.5 text-[10px] text-chat-text-secondary">
                      <ClockIcon className="w-3 h-3" />
                      {formatTime(entry.timestamp)}
                    </span>
                  )}
                  {entry.category && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-indigo-600/20 text-indigo-400">
                      {entry.category}
                    </span>
                  )}
                  {entry.id && (
                    <button
                      onClick={async () => {
                        try {
                          await fetch(`/api/memory/${entry.id}`, { method: 'DELETE' });
                          fetchMemories();
                        } catch { /* ignore */ }
                      }}
                      className="text-[10px] text-red-400/60 hover:text-red-400 ml-auto transition-colors"
                      title="Delete memory"
                    >
                      Delete
                    </button>
                  )}
                  {entry.relevance !== undefined && (
                    <span className={`text-[10px] text-emerald-400 font-medium ${entry.id ? '' : 'ml-auto'}`}>
                      {Math.round(entry.relevance * 100)}% match
                    </span>
                  )}
                  {entry.tags && entry.tags.length > 0 && (
                    <div className="flex gap-1 flex-wrap">
                      {entry.tags.slice(0, 4).map((tag, j) => (
                        <span key={j} className="text-[10px] text-blue-400">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Refresh button */}
          <div className="px-4 py-2 border-t border-chat-border flex-shrink-0">
            <button
              onClick={fetchMemories}
              disabled={memoriesLoading}
              className="flex items-center gap-1.5 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors"
            >
              <ArrowPathIcon className={`w-3 h-3 ${memoriesLoading ? 'animate-spin' : ''}`} />
              Refresh memories
            </button>
          </div>
        </div>
      )}

      {/* Tab: Ask */}
      {tab === 'ask' && (
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* Answer area */}
          <div ref={answerScrollRef} className="flex-1 overflow-y-auto p-4">
            {!answer && !qaError && !isStreaming && (
              <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary text-center py-8">
                <ChatBubbleLeftEllipsisIcon className="w-8 h-8 mb-2 opacity-30" />
                <p className="text-sm">Ask AURA anything from memory</p>
                <p className="text-xs mt-1 opacity-60">
                  Answers are grounded in stored knowledge
                </p>
              </div>
            )}

            {qaError && (
              <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-3">
                {qaError}
              </div>
            )}

            {(answer || isStreaming) && (
              <div className="rounded-lg p-3 bg-surface-1 border border-chat-border">
                <div className="flex items-center gap-1.5 mb-2">
                  <BookOpenIcon className="w-3.5 h-3.5 text-indigo-400" />
                  <span className="text-[10px] text-indigo-400 font-medium">Knowledge answer</span>
                  {isStreaming && (
                    <span className="ml-auto flex items-center gap-1 text-[10px] text-chat-text-secondary">
                      <div className="shimmer-bar h-1.5 w-12" />
                    </span>
                  )}
                </div>
                <p className="text-sm text-chat-text leading-relaxed whitespace-pre-wrap">
                  {answer}
                  {isStreaming && (
                    <span className="inline-block w-1.5 h-3.5 bg-indigo-400 animate-pulse ml-0.5 align-middle" />
                  )}
                </p>
              </div>
            )}
          </div>

          {/* Question input */}
          <div className="p-3 border-t border-chat-border flex-shrink-0">
            <div className="flex gap-2">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAsk();
                  }
                }}
                placeholder="Ask a question from memory..."
                className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-indigo-500 placeholder-chat-text-secondary/50"
                rows={2}
                disabled={isStreaming}
              />
              <button
                onClick={isStreaming ? handleStop : handleAsk}
                disabled={!isStreaming && !question.trim()}
                className="self-end p-2.5 rounded-lg bg-indigo-600 hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
              >
                {isStreaming ? (
                  <StopIcon className="w-4 h-4" />
                ) : (
                  <PaperAirplaneIcon className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab: Add */}
      {tab === 'add' && (
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {/* Input */}
            <div>
              <label className="text-xs text-chat-text-secondary block mb-1.5">
                Knowledge to store
              </label>
              <textarea
                value={newFact}
                onChange={(e) => {
                  setNewFact(e.target.value);
                  setAddSuccess(false);
                  setAddConfirmation('');
                  setAddError('');
                }}
                placeholder="Enter a fact, memory, or piece of knowledge for AURA to remember..."
                className="w-full p-3 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-indigo-500 placeholder-chat-text-secondary/50 min-h-[100px]"
                disabled={isAdding}
              />
            </div>

            {/* Error */}
            {addError && (
              <div className="text-xs text-red-400 bg-red-500/10 rounded-lg px-3 py-2">
                {addError}
              </div>
            )}

            {/* AI confirmation */}
            {(addConfirmation || isAdding) && (
              <div className="rounded-lg p-3 bg-surface-1 border border-chat-border">
                <div className="flex items-center gap-1.5 mb-2">
                  {addSuccess ? (
                    <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-400" />
                  ) : (
                    <BookOpenIcon className="w-3.5 h-3.5 text-indigo-400" />
                  )}
                  <span className={`text-[10px] font-medium ${addSuccess ? 'text-emerald-400' : isAdding ? 'text-indigo-400' : 'text-amber-400'}`}>
                    {addSuccess ? 'Stored + confirmed' : isAdding ? 'Processing...' : 'Acknowledged (not stored)'}
                  </span>
                  {isAdding && (
                    <div className="ml-auto shimmer-bar h-1.5 w-10" />
                  )}
                </div>
                <p className="text-sm text-chat-text leading-relaxed whitespace-pre-wrap">
                  {addConfirmation}
                  {isAdding && (
                    <span className="inline-block w-1.5 h-3.5 bg-indigo-400 animate-pulse ml-0.5 align-middle" />
                  )}
                </p>
              </div>
            )}
          </div>

          {/* Submit */}
          <div className="p-3 border-t border-chat-border flex-shrink-0">
            <button
              onClick={isAdding ? handleStop : handleAdd}
              disabled={!isAdding && !newFact.trim()}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-indigo-600 hover:opacity-90 disabled:opacity-40 text-white text-sm transition-opacity"
            >
              {isAdding ? (
                <>
                  <StopIcon className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <PlusIcon className="w-4 h-4" />
                  Store Knowledge
                </>
              )}
            </button>
            <p className="text-[10px] text-chat-text-secondary text-center mt-1.5">
              AURA will acknowledge and confirm what was noted
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
