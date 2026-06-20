import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import {
  PaperAirplaneIcon,
  StopIcon,
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type MathMode = 'solve' | 'stepbystep' | 'graph' | 'simplify';

interface HistoryEntry {
  problem: string;
  mode: MathMode;
  solution: string;
  timestamp: number;
}

/* ── Constants ── */
const MODES: { id: MathMode; label: string }[] = [
  { id: 'solve', label: 'Solve' },
  { id: 'stepbystep', label: 'Step-by-Step' },
  { id: 'graph', label: 'Graph' },
  { id: 'simplify', label: 'Simplify' },
];

const TEMPLATES: { label: string; prompt: string }[] = [
  { label: 'Algebra', prompt: 'Solve for x: 2x^2 - 5x + 3 = 0' },
  { label: 'Calculus', prompt: 'Find the integral of x^2 * sin(x) dx' },
  { label: 'Statistics', prompt: 'Find the mean, variance, and standard deviation of: 4, 8, 15, 16, 23, 42' },
  { label: 'Geometry', prompt: 'Find the area and perimeter of a triangle with sides 5, 12, and 13' },
  { label: 'Linear Algebra', prompt: 'Find the eigenvalues of the matrix [[3, 1], [1, 3]]' },
];

const SYSTEM_PROMPTS: Record<MathMode, string> = {
  solve: 'Solve the following math problem. Show the final answer clearly. Use clear notation.',
  stepbystep:
    'Solve the following math problem step by step. Number each step. Explain your reasoning at each step. Show the final answer at the end.',
  graph:
    'Describe what the graph of the following function/equation looks like. Include: shape, key points (intercepts, maxima, minima, asymptotes), domain, range, and behavior.',
  simplify: 'Simplify the following mathematical expression. Show each simplification step.',
};

/* ── Component ── */
export function MathPanel() {
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<MathMode>('stepbystep');
  const [solution, setSolution] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  // Fetch available models
  useEffect(() => {
    apiFetch('/api/models')
      .then((res) => res.json())
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

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Auto-scroll output during streaming
  useEffect(() => {
    if (isGenerating && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [solution, isGenerating]);

  const handleSolve = useCallback(async (problem: string, targetMode: MathMode) => {
    if (!problem.trim() || isGenerating) return;

    setError(null);
    setSolution('');
    setIsGenerating(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: problem,
          system_prompt: SYSTEM_PROMPTS[targetMode],
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
                const text =
                  parsed.choices?.[0]?.delta?.content ||
                  parsed.content ||
                  parsed.chunk ||
                  '';
                if (text) {
                  fullResponse += text;
                  setSolution(fullResponse);
                }
              } catch {
                fullResponse += data;
                setSolution(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setSolution(fullResponse);
            }
          }
        }
      } else {
        fullResponse = await res.text();
        setSolution(fullResponse);
      }

      // Save to history (keep last 5)
      if (fullResponse.trim()) {
        setHistory((prev) =>
          [
            { problem, mode: targetMode, solution: fullResponse, timestamp: Date.now() },
            ...prev,
          ].slice(0, 5)
        );
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Something went wrong. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [isGenerating, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleSubmit = useCallback(() => {
    handleSolve(input, mode);
  }, [input, mode, handleSolve]);

  const handleTemplateClick = useCallback((prompt: string) => {
    setInput(prompt);
    handleSolve(prompt, mode);
  }, [mode, handleSolve]);

  const handleHistoryClick = useCallback((entry: HistoryEntry) => {
    setInput(entry.problem);
    setMode(entry.mode);
    setSolution(entry.solution);
    setError(null);
  }, []);

  const handleCopy = useCallback(() => {
    if (!solution) return;
    navigator.clipboard.writeText(solution).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [solution]);

  const modeLabel = MODES.find((m) => m.id === mode)?.label ?? mode;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">

      {/* Left: Input panel */}
      <div className="flex flex-col md:w-[380px] md:min-w-[280px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0">

        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Math Solver</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">
            Type a problem or equation — Aura will solve it
          </p>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* Mode selector */}
          <div>
            <p className="text-[10px] text-chat-text-secondary mb-2 uppercase tracking-wide">Mode</p>
            <div className="flex flex-wrap gap-1.5">
              {MODES.map((m) => (
                <button
                  key={m.id}
                  onClick={() => setMode(m.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors border ${
                    mode === m.id
                      ? 'bg-chat-accent text-white border-transparent'
                      : 'text-chat-text-secondary border-chat-border hover:text-chat-text hover:border-chat-accent/40'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Quick templates */}
          <div>
            <p className="text-[10px] text-chat-text-secondary mb-2 uppercase tracking-wide">Quick start</p>
            <div className="flex flex-wrap gap-1.5">
              {TEMPLATES.map((t) => (
                <button
                  key={t.label}
                  onClick={() => handleTemplateClick(t.prompt)}
                  disabled={isGenerating}
                  className="px-2.5 py-1 rounded-md text-[11px] border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-accent/40 transition-colors disabled:opacity-40"
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          {/* Recent history */}
          {history.length > 0 && (
            <div>
              <p className="text-[10px] text-chat-text-secondary mb-2 uppercase tracking-wide">Recent</p>
              <div className="space-y-1.5">
                {history.map((entry) => (
                  <button
                    key={entry.timestamp}
                    onClick={() => handleHistoryClick(entry)}
                    className="w-full text-left px-3 py-2 rounded-lg border border-chat-border bg-surface-1 hover:border-chat-accent/40 transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[10px] text-chat-accent/80 font-medium">
                        {MODES.find((m) => m.id === entry.mode)?.label}
                      </span>
                      <span className="text-[10px] text-chat-text-secondary">
                        {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <p className="text-xs text-chat-text truncate group-hover:text-white transition-colors">
                      {entry.problem}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Input area */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              placeholder="e.g. integral of x^2, or solve 3x + 5 = 14"
              className="flex-1 p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={3}
              disabled={isGenerating}
            />
            <button
              onClick={isGenerating ? handleStop : handleSubmit}
              disabled={!isGenerating && !input.trim()}
              className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
              title={isGenerating ? 'Stop' : `${modeLabel}`}
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
                onClick={() => setShowModelMenu((p) => !p)}
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
                    position: 'absolute',
                    bottom: 28,
                    left: 0,
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
        </div>
      </div>

      {/* Right: Output panel */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">

        {/* Output toolbar */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-chat-border flex-shrink-0">
          <span className="text-xs font-medium text-chat-text-secondary flex-1">
            {isGenerating
              ? <span className="text-purple-400">Solving...</span>
              : solution
              ? <span className="text-chat-text">{modeLabel} — solution ready</span>
              : <span>Solution will appear here</span>
            }
          </span>
          {solution && (
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] border border-chat-border text-chat-text-secondary hover:text-chat-text transition-colors"
              title="Copy solution"
            >
              {copied
                ? <ClipboardDocumentCheckIcon className="w-3.5 h-3.5 text-green-400" />
                : <ClipboardDocumentIcon className="w-3.5 h-3.5" />
              }
              {copied ? 'Copied!' : 'Copy'}
            </button>
          )}
        </div>

        {/* Output body */}
        <div ref={outputRef} className="flex-1 overflow-y-auto p-5">
          {error && (
            <div className="mb-4 px-4 py-3 rounded-lg border border-red-500/30 bg-red-500/10 text-sm text-red-400">
              {error}
            </div>
          )}

          {solution ? (
            <div className="text-sm text-chat-text leading-relaxed prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                {solution}
              </ReactMarkdown>
              {isGenerating && (
                <span className="inline-block w-1.5 h-4 bg-purple-400 animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          ) : !error && (
            <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary text-sm gap-3">
              <div className="text-4xl select-none" aria-hidden>∑</div>
              <div className="text-center">
                <p className="font-medium">No problem yet</p>
                <p className="text-[11px] mt-1 text-chat-text-secondary/70">
                  Type a math problem on the left or pick a quick-start template
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
