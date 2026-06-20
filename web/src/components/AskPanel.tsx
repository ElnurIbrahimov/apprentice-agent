import { useState, useRef, useEffect, useCallback } from 'react';
import {
  QuestionMarkCircleIcon,
  StopIcon,
  ClipboardDocumentIcon,
  CheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

const ASK_SYSTEM_PROMPT = `Answer the user's question directly and concisely. Be accurate and helpful. If the question is ambiguous, give the most likely interpretation. Format with markdown for readability.`;

const QUICK_TEMPLATES = [
  'Explain ',
  'How does ',
  'What is ',
  'Compare ',
];

interface QAPair {
  question: string;
  answer: string;
  timestamp: number;
}

/* ── Simple markdown-like renderer ── */
function FormattedAnswer({ text }: { text: string }) {
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

/* ── Copy button ── */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // fallback silently
    }
  };

  return (
    <button
      onClick={handleCopy}
      title="Copy answer"
      className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-1.5 py-1 rounded"
      style={{ background: 'transparent' }}
    >
      {copied
        ? <CheckIcon className="w-3.5 h-3.5 text-green-400" />
        : <ClipboardDocumentIcon className="w-3.5 h-3.5" />}
      <span>{copied ? 'Copied' : 'Copy'}</span>
    </button>
  );
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
    </div>
  );
}

/* ── Main Component ── */
export function AskPanel() {
  const [question, setQuestion] = useState('');
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<QAPair[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const streamAreaRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Fetch models
  useEffect(() => {
    apiFetch('/api/models')
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

  // Auto-scroll during streaming
  useEffect(() => {
    if (isAsking && streamAreaRef.current) {
      streamAreaRef.current.scrollTop = streamAreaRef.current.scrollHeight;
    }
  }, [streamingAnswer, isAsking]);

  const handleAsk = useCallback(async (q: string) => {
    const trimmed = q.trim();
    if (!trimmed || isAsking) return;

    setCurrentQuestion(trimmed);
    setStreamingAnswer('');
    setError(null);
    setIsAsking(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          system_prompt: ASK_SYSTEM_PROMPT,
          history: [],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullAnswer = '';

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) {
                  fullAnswer += text;
                  setStreamingAnswer(fullAnswer);
                }
              } catch {
                fullAnswer += data;
                setStreamingAnswer(fullAnswer);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullAnswer += line;
              setStreamingAnswer(fullAnswer);
            }
          }
        }
      } else {
        fullAnswer = await res.text();
        setStreamingAnswer(fullAnswer);
      }

      if (fullAnswer.trim()) {
        setHistory(prev => [
          { question: trimmed, answer: fullAnswer, timestamp: Date.now() },
          ...prev,
        ].slice(0, 10));
        setStreamingAnswer('');
        setCurrentQuestion('');
        setQuestion('');
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Request failed. Make sure the backend is running.');
      }
    } finally {
      setIsAsking(false);
      abortRef.current = null;
    }
  }, [isAsking, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsAsking(false);
    // Preserve whatever streamed so far as a completed entry
    if (streamingAnswer.trim() && currentQuestion) {
      setHistory(prev => [
        { question: currentQuestion, answer: streamingAnswer + ' _(stopped)_', timestamp: Date.now() },
        ...prev,
      ].slice(0, 10));
    }
    setStreamingAnswer('');
    setCurrentQuestion('');
  }, [streamingAnswer, currentQuestion]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleAsk(question);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAsk(question);
    }
  };

  const applyTemplate = (template: string) => {
    setQuestion(template);
    inputRef.current?.focus();
  };

  const removeHistoryEntry = (index: number) => {
    setHistory(prev => prev.filter((_, i) => i !== index));
  };

  const hasStreaming = streamingAnswer.length > 0 || (isAsking && currentQuestion);

  return (
    <div className="h-full flex flex-col bg-surface-0 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Quick Ask</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">Ask a question — get a focused answer instantly</p>
      </div>

      {/* Input area */}
      <div className="px-4 pt-4 pb-2 flex-shrink-0">
        <form onSubmit={handleSubmit} className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything... (Enter to send, Shift+Enter for newline)"
            rows={2}
            className="flex-1 px-3 py-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 transition-colors resize-none"
            disabled={isAsking}
            autoComplete="off"
          />
          <button
            type={isAsking ? 'button' : 'submit'}
            onClick={isAsking ? handleStop : undefined}
            disabled={!isAsking && !question.trim()}
            className="px-3 py-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity flex items-center gap-1.5 text-sm font-medium flex-shrink-0"
          >
            {isAsking
              ? <><StopIcon className="w-4 h-4" /> Stop</>
              : <><QuestionMarkCircleIcon className="w-4 h-4" /> Ask</>
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

          {/* History count */}
          {history.length > 0 && (
            <span className="text-[10px] text-chat-text-secondary px-2 py-1 rounded-md" style={{ background: 'var(--border-subtle)' }}>
              {history.length} Q&A{history.length !== 1 ? 's' : ''}
            </span>
          )}
        </div>

        {/* Quick templates */}
        <div className="flex flex-wrap gap-1.5 mt-3">
          {QUICK_TEMPLATES.map((t) => (
            <button
              key={t}
              onClick={() => applyTemplate(t)}
              disabled={isAsking}
              className="px-3 py-1 rounded-full text-xs border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-accent transition-colors bg-surface-1 disabled:opacity-40"
            >
              {t}...
            </button>
          ))}
        </div>
      </div>

      {/* Scrollable content: streaming + history */}
      <div ref={streamAreaRef} className="flex-1 overflow-y-auto px-4 pb-6 space-y-4">

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-500/30 bg-red-900/10 p-4 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Active streaming answer */}
        {hasStreaming && (
          <div className="rounded-lg border border-chat-accent/30 bg-surface-1 overflow-hidden">
            <div className="px-3 py-2 border-b border-chat-border/50 flex items-center gap-2">
              <QuestionMarkCircleIcon className="w-3.5 h-3.5 text-chat-accent flex-shrink-0" />
              <span className="text-xs font-medium text-chat-text truncate flex-1">{currentQuestion}</span>
            </div>
            <div className="p-4">
              {isAsking && !streamingAnswer && <ShimmerSkeleton />}
              {streamingAnswer && (
                <>
                  <FormattedAnswer text={streamingAnswer} />
                  {isAsking && (
                    <span className="inline-block w-1.5 h-3.5 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!hasStreaming && !error && history.length === 0 && (
          <div className="flex flex-col items-center justify-center h-40 text-chat-text-secondary text-sm gap-2">
            <QuestionMarkCircleIcon className="w-8 h-8 opacity-30" />
            <p className="text-xs opacity-60">Answers will appear here</p>
          </div>
        )}

        {/* Q&A history */}
        {history.length > 0 && (
          <div className="space-y-3">
            {history.map((pair, i) => (
              <div key={pair.timestamp} className="rounded-lg border border-chat-border bg-surface-1 overflow-hidden">
                {/* Question row */}
                <div className="px-3 py-2 border-b border-chat-border/50 flex items-center gap-2">
                  <QuestionMarkCircleIcon className="w-3.5 h-3.5 text-chat-text-secondary flex-shrink-0" />
                  <span className="text-xs font-medium text-chat-text truncate flex-1">{pair.question}</span>
                  <span className="text-[10px] text-chat-text-secondary flex-shrink-0">
                    {new Date(pair.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                  <button
                    onClick={() => removeHistoryEntry(i)}
                    className="text-chat-text-secondary hover:text-chat-text flex-shrink-0 ml-0.5"
                    title="Remove"
                  >
                    <XMarkIcon className="w-3.5 h-3.5" />
                  </button>
                </div>
                {/* Answer */}
                <div className="p-4">
                  <FormattedAnswer text={pair.answer} />
                </div>
                {/* Footer with copy */}
                <div className="px-3 py-1.5 border-t border-chat-border/50 flex justify-end">
                  <CopyButton text={pair.answer} />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
