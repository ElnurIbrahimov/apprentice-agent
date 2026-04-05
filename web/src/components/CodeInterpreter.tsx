import React, { useState, useRef, useCallback, useEffect } from 'react';
import { PlayIcon, StopIcon, ArrowPathIcon, ChevronDownIcon, ChevronUpIcon, SparklesIcon, XMarkIcon, ClipboardDocumentIcon, ClipboardDocumentCheckIcon, TrashIcon, ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/outline';
import { highlightCode } from '../utils/codeHighlighter';
import { sanitizeHtml } from '../utils/sanitize';
import { execute, resetRuntime, subscribe, isReady, type OutputBlock, type VariableInfo } from '../utils/pyodideExecutor';
import { executeJS } from '../utils/jsExecutor';

interface Exchange {
  id: string;
  code: string;
  outputs: OutputBlock[];
  variables: VariableInfo[];
  phase: 'idle' | 'executing';
  executionTime?: number;
  runMode?: 'browser' | 'server';
  language: 'python' | 'javascript';
  executionOrder?: number;
}

let counter = 0;
function newId() { return `ex-${Date.now()}-${++counter}`; }

// Module-level execution order counter
let _execOrderCounter = 0;

/* ── Output renderer ── */
function OutputRenderer({ block }: { block: OutputBlock }) {
  if (block.type === 'stdout' || block.type === 'stderr') {
    return (
      <pre className={`text-xs font-mono whitespace-pre-wrap px-3 py-1 ${block.type === 'stderr' ? 'text-yellow-400' : 'text-chat-text'}`}>
        {block.text}
      </pre>
    );
  }
  if (block.type === 'image' && block.data) {
    return (
      <div className="px-3 py-2">
        <img src={`data:${block.mime || 'image/png'};base64,${block.data}`} alt="Output" className="max-w-full rounded-lg" />
      </div>
    );
  }
  if (block.type === 'html' && block.content) {
    return (
      <div className="px-3 py-2 overflow-x-auto [&_table]:text-xs [&_table]:border-collapse [&_td]:px-2 [&_td]:py-1 [&_td]:border [&_td]:border-chat-border [&_th]:px-2 [&_th]:py-1 [&_th]:border [&_th]:border-chat-border [&_th]:bg-surface-2 [&_th]:text-chat-text-secondary"
        dangerouslySetInnerHTML={{ __html: sanitizeHtml(block.content) }}
      />
    );
  }
  if (block.type === 'error') {
    return (
      <div className="px-3 py-2">
        <div className="text-xs font-semibold text-red-400">{block.ename}: {block.evalue}</div>
        {block.traceback && (
          <pre className="text-[10px] text-red-300/70 mt-1 whitespace-pre-wrap font-mono">{block.traceback}</pre>
        )}
      </div>
    );
  }
  return null;
}

/* ── Insert cell button ── */
function InsertCellButton({ onClick }: { onClick: () => void }) {
  return (
    <div className="flex justify-center py-1 group">
      <button onClick={onClick} className="w-6 h-6 rounded-full border border-chat-border/30 text-chat-text-secondary/30 hover:border-purple-500/50 hover:text-purple-400 flex items-center justify-center text-xs transition-colors opacity-0 group-hover:opacity-100">+</button>
    </div>
  );
}

/* ── Exchange card ── */
function ExchangeCard({
  exchange,
  onRerun,
  onDelete,
  onMoveUp,
  onMoveDown,
}: {
  exchange: Exchange;
  onRerun: (code: string) => void;
  onDelete: (id: string) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
}) {
  const [codeCopied, setCodeCopied] = useState(false);
  const runBadge = exchange.runMode === 'server'
    ? <span className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400">Server</span>
    : exchange.runMode === 'browser'
    ? <span className="text-[9px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">Browser</span>
    : null;
  const [codeHtml, setCodeHtml] = useState('');
  const [showVars, setShowVars] = useState(false);

  const langLabel = exchange.language === 'javascript' ? 'JavaScript' : 'Python';
  const langColor = exchange.language === 'javascript' ? 'text-yellow-400' : 'text-purple-400';

  useEffect(() => {
    if (exchange.code) {
      const isDark = !document.documentElement.classList.contains('light');
      highlightCode(exchange.code, exchange.language === 'javascript' ? 'javascript' : 'python', isDark ? 'dark' : 'light')
        .then(setCodeHtml)
        .catch(() => {});
    }
  }, [exchange.code, exchange.language]);

  return (
    <div className="border border-chat-border rounded-lg overflow-hidden bg-surface-1">
      {/* Code */}
      <div className="border-b border-chat-border">
        <div className="flex items-center justify-between px-3 py-1.5">
          <div className="flex items-center gap-2">
            {exchange.executionOrder != null && (
              <span className="text-[9px] font-mono text-chat-text-secondary/60 select-none">[{exchange.executionOrder}]</span>
            )}
            <span className={`text-[10px] font-semibold uppercase tracking-wider ${langColor}`}>{langLabel}</span>
            {runBadge}
          </div>
          <div className="flex items-center gap-1.5">
            {exchange.phase === 'executing' && (
              <span className="text-[10px] text-yellow-400 animate-pulse">Running...</span>
            )}
            {exchange.executionTime != null && (
              <span className="text-[10px] text-chat-text-secondary">{exchange.executionTime.toFixed(2)}s</span>
            )}
            {exchange.phase === 'idle' && (
              <>
                <button
                  onClick={() => onRerun(exchange.code)}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-green-600/20 text-green-400 hover:bg-green-600/30 transition-colors flex items-center gap-0.5"
                  title="Load code back into editor"
                >
                  <ArrowPathIcon className="w-3 h-3" />Re-run
                </button>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(exchange.code);
                    setCodeCopied(true);
                    setTimeout(() => setCodeCopied(false), 1500);
                  }}
                  className="p-0.5 rounded text-chat-text-secondary hover:text-chat-text transition-colors"
                  title="Copy code"
                >
                  {codeCopied
                    ? <ClipboardDocumentCheckIcon className="w-3.5 h-3.5 text-green-400" />
                    : <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                  }
                </button>
                <button
                  onClick={onMoveUp}
                  className="p-0.5 rounded text-chat-text-secondary hover:text-chat-text transition-colors"
                  title="Move cell up"
                >
                  <ArrowUpIcon className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={onMoveDown}
                  className="p-0.5 rounded text-chat-text-secondary hover:text-chat-text transition-colors"
                  title="Move cell down"
                >
                  <ArrowDownIcon className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => onDelete(exchange.id)}
                  className="p-0.5 rounded text-chat-text-secondary hover:text-red-400 transition-colors"
                  title="Delete cell"
                >
                  <TrashIcon className="w-3.5 h-3.5" />
                </button>
              </>
            )}
          </div>
        </div>
        {codeHtml ? (
          <div
            dangerouslySetInnerHTML={{ __html: codeHtml }}
            className="shiki-block px-4 py-2 text-sm [&_pre]:!bg-transparent [&_pre]:!m-0 [&_pre]:!p-0 [&_code]:!bg-transparent overflow-x-auto"
          />
        ) : (
          <pre className="px-4 py-2 text-sm font-mono text-chat-text overflow-x-auto">{exchange.code}</pre>
        )}
      </div>

      {/* Outputs */}
      {exchange.outputs.length > 0 && (
        <div className="divide-y divide-chat-border">
          {exchange.outputs.map((block, i) => (
            <OutputRenderer key={i} block={block} />
          ))}
        </div>
      )}

      {/* Variables */}
      {exchange.variables.length > 0 && (
        <div className="border-t border-chat-border">
          <button
            onClick={() => setShowVars(!showVars)}
            className="flex items-center gap-1 px-3 py-1.5 text-[10px] text-chat-text-secondary hover:text-chat-text w-full"
          >
            {showVars ? <ChevronUpIcon className="w-3 h-3" /> : <ChevronDownIcon className="w-3 h-3" />}
            Variables ({exchange.variables.length})
          </button>
          {showVars && (
            <div className="px-3 pb-2 space-y-0.5">
              {exchange.variables.map((v) => (
                <div key={v.name} className="flex gap-2 text-[10px] font-mono">
                  <span className="text-purple-400">{v.name}</span>
                  <span className="text-chat-text-secondary">: {v.type_name} =</span>
                  <span className="text-chat-text truncate">{v.repr}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const CODE_SYSTEM_PROMPT = `You are a Python code generator. Output ONLY executable Python code, no markdown fences, no explanation.
Available libraries: numpy, pandas, matplotlib, scipy, scikit-learn, sympy.
For plots, always call plt.show() at the end.
For DataFrames, print them with print(df).
Write clean, commented code. Output ONLY the Python code.`;

/* ── Main component ── */
export function CodeInterpreter() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [code, setCode] = useState('');
  const [workerReady, setWorkerReady] = useState(isReady());
  const [loading, setLoading] = useState(isReady() ? '' : 'Initializing...');
  const [isExecuting, setIsExecuting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [runMode, setRunMode] = useState<'browser' | 'server'>('browser');
  const [defaultLang, setDefaultLang] = useState<'python' | 'javascript'>('python');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const serverSessionId = useRef<string>(`code-${Date.now().toString(36)}`);

  // Subscribe to global worker events (ready/loading)
  useEffect(() => {
    return subscribe({
      onReady: () => { setWorkerReady(true); setLoading(''); },
      onLoading: (stage) => setLoading(stage),
    });
  }, []);

  // Auto-scroll
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [exchanges]);

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

  // Close model menu on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) setShowModelMenu(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Cleanup abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const handleAskAI = useCallback(async () => {
    const prompt = code.trim();
    if (!prompt || isGenerating) return;

    setIsGenerating(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          system_prompt: CODE_SYSTEM_PROMPT,
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullCode = '';
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
                if (text) { fullCode += text; setCode(fullCode); }
              } catch {
                fullCode += data;
                setCode(fullCode);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullCode += line;
              setCode(fullCode);
            }
          }
        }
      } else {
        fullCode = await res.text();
        setCode(fullCode);
      }

      // Strip markdown fences if present
      let clean = fullCode.trim();
      const fenceMatch = clean.match(/```python?\s*\n([\s\S]*?)```/);
      if (fenceMatch) clean = fenceMatch[1].trim();
      if (clean !== fullCode.trim()) setCode(clean);
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setCode(`# Error generating code: ${e.message}`);
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [code, isGenerating, selectedModel]);

  const handleRun = useCallback(() => {
    const trimmed = code.trim();
    if (!trimmed || (defaultLang === 'python' && !workerReady)) return;

    const id = newId();
    const execOrder = ++_execOrderCounter;
    setIsExecuting(true);

    if (defaultLang === 'javascript') {
      setExchanges((prev) => [
        ...prev.slice(-49),
        { id, code: trimmed, outputs: [], variables: [], phase: 'executing', runMode: 'browser', language: 'javascript', executionOrder: execOrder },
      ]);
      setCode('');

      executeJS(trimmed, {
        onOutput: (block) => {
          setExchanges((prev) =>
            prev.map((ex) => ex.id === id ? { ...ex, outputs: [...ex.outputs, block] } : ex)
          );
        },
        onDone: (_success, executionTime) => {
          setExchanges((prev) =>
            prev.map((ex) => ex.id === id ? { ...ex, phase: 'idle', executionTime } : ex)
          );
          setIsExecuting(false);
        },
      });
      return;
    }

    setExchanges((prev) => [
      ...prev.slice(-49),
      { id, code: trimmed, outputs: [], variables: [], phase: 'executing', runMode: 'browser', language: 'python', executionOrder: execOrder },
    ]);
    setCode('');

    execute(trimmed, {
      onOutput: (block) => {
        setExchanges((prev) =>
          prev.map((ex) => ex.id === id ? { ...ex, outputs: [...ex.outputs, block] } : ex)
        );
      },
      onVariables: (variables) => {
        setExchanges((prev) =>
          prev.map((ex) => ex.id === id ? { ...ex, variables } : ex)
        );
      },
      onDone: (_success, executionTime) => {
        setExchanges((prev) =>
          prev.map((ex) => ex.id === id ? { ...ex, phase: 'idle', executionTime } : ex)
        );
        setIsExecuting(false);
      },
    });
  }, [code, workerReady, defaultLang]);

  const handleRunServer = useCallback(async () => {
    const trimmed = code.trim();
    if (!trimmed) return;

    const execOrder = ++_execOrderCounter;
    setIsExecuting(true);
    const id = newId();
    setExchanges(prev => [...prev.slice(-49), { id, code: trimmed, outputs: [], variables: [], phase: 'executing', runMode: 'server', language: 'python', executionOrder: execOrder }]);
    setCode('');

    try {
      const res = await fetch('/api/code/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: trimmed, session_id: serverSessionId.current }),
      });
      const data = await res.json();

      if (!res.ok) {
        const errMsg = data.detail || `HTTP ${res.status}`;
        setExchanges(prev => prev.map(ex =>
          ex.id === id ? { ...ex, outputs: [{ type: 'error', ename: 'ServerError', evalue: errMsg }], phase: 'idle' } : ex
        ));
        return;
      }

      const outputs: OutputBlock[] = data.outputs ?? [];
      const variables: VariableInfo[] = data.variables ?? [];

      setExchanges(prev => prev.map(ex =>
        ex.id === id ? { ...ex, outputs, variables, phase: 'idle', executionTime: data.execution_time } : ex
      ));
    } catch (e: any) {
      setExchanges(prev => prev.map(ex =>
        ex.id === id ? { ...ex, outputs: [{ type: 'error', ename: 'NetworkError', evalue: e.message }], phase: 'idle' } : ex
      ));
    } finally {
      setIsExecuting(false);
    }
  }, [code]);

  const handleReset = useCallback(() => {
    resetRuntime();
    serverSessionId.current = `code-${Date.now().toString(36)}`;
    setExchanges([]);
  }, []);

  const handleClearExchanges = useCallback(() => {
    setExchanges([]);
  }, []);

  const handleRerun = useCallback((codeToLoad: string) => {
    setCode(codeToLoad);
    textareaRef.current?.focus();
    textareaRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, []);

  const insertCellAt = useCallback((index: number) => {
    const id = `ex-${Date.now()}-${exchanges.length}`;
    setExchanges(prev => {
      const next = [...prev];
      next.splice(index, 0, { id, code: '', outputs: [], variables: [], phase: 'idle' as const, language: defaultLang });
      return next;
    });
  }, [defaultLang, exchanges.length]);

  const deleteCell = useCallback((id: string) => {
    setExchanges(prev => prev.filter(e => e.id !== id));
  }, []);

  const moveCell = useCallback((id: string, direction: 'up' | 'down') => {
    setExchanges(prev => {
      const idx = prev.findIndex(e => e.id === id);
      if (idx < 0) return prev;
      const target = direction === 'up' ? idx - 1 : idx + 1;
      if (target < 0 || target >= prev.length) return prev;
      const next = [...prev];
      [next[idx], next[target]] = [next[target], next[idx]];
      return next;
    });
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border flex-shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-chat-text">Code Interpreter</h2>
          <span className={`text-[10px] px-2 py-0.5 rounded-full ${workerReady ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
            {workerReady ? 'Python Ready' : loading || 'Loading...'}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={handleClearExchanges}
            disabled={exchanges.length === 0}
            className="flex items-center gap-1 px-2 py-1 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] rounded transition-colors disabled:opacity-30"
            title="Clear exchange history"
          >
            <XMarkIcon className="w-3.5 h-3.5" />
            Clear
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-1 px-2 py-1 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] rounded transition-colors"
            title="Reset runtime + clear"
          >
            <ArrowPathIcon className="w-3.5 h-3.5" />
            Reset
          </button>
        </div>
      </div>

      {/* Exchanges */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        {exchanges.length === 0 && workerReady && (
          <div className="text-center text-chat-text-secondary text-sm py-8">
            <p className="mb-4">Write Python code below and press Run.</p>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                { label: 'Hello World', code: 'print("Hello from Pyodide!")' },
                { label: 'Math', code: 'import math\nprint(f"pi = {math.pi}")\nprint(f"20! = {math.factorial(20)}")' },
                { label: 'Chart', code: 'import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\nplt.plot(x, np.sin(x), label="sin(x)")\nplt.plot(x, np.cos(x), label="cos(x)")\nplt.legend()\nplt.title("Trig Functions")\nplt.show()' },
                { label: 'DataFrame', code: 'import pandas as pd\nimport numpy as np\n\ndf = pd.DataFrame({\n    "Name": ["Alice", "Bob", "Charlie", "Diana"],\n    "Score": np.random.randint(60, 100, 4),\n    "Grade": ["A", "B", "A", "B"]\n})\nprint(df)' },
              ].map((q) => (
                <button
                  key={q.label}
                  onClick={() => { setCode(q.code); textareaRef.current?.focus(); }}
                  className="px-3 py-1.5 text-xs rounded-full bg-surface-2 hover:bg-surface-3 border border-chat-border hover:border-purple-500/30 transition-colors"
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {!workerReady && (
          <div className="text-center text-chat-text-secondary text-sm py-8">
            <div className="shimmer-bar h-3 w-48 mx-auto mb-3" />
            <p>{loading || 'Initializing Python runtime...'}</p>
          </div>
        )}

        <InsertCellButton onClick={() => insertCellAt(0)} />
        {exchanges.map((ex, i) => (
          <React.Fragment key={ex.id}>
            <div className="mb-4">
              <ExchangeCard
                exchange={ex}
                onRerun={handleRerun}
                onDelete={deleteCell}
                onMoveUp={() => moveCell(ex.id, 'up')}
                onMoveDown={() => moveCell(ex.id, 'down')}
              />
            </div>
            <InsertCellButton onClick={() => insertCellAt(i + 1)} />
          </React.Fragment>
        ))}
      </div>

      {/* Input */}
      <div className="border-t border-chat-border p-3 flex-shrink-0">
        <div className="flex gap-2">
          <div className="flex-1 flex flex-col">
            <div className="flex gap-1 mb-1.5">
              <button
                onClick={() => setDefaultLang('python')}
                className={`text-[10px] px-2 py-0.5 rounded ${defaultLang === 'python' ? 'bg-green-600/30 text-green-400' : 'text-chat-text-secondary'}`}
              >Python</button>
              <button
                onClick={() => setDefaultLang('javascript')}
                className={`text-[10px] px-2 py-0.5 rounded ${defaultLang === 'javascript' ? 'bg-yellow-600/30 text-yellow-400' : 'text-chat-text-secondary'}`}
              >JavaScript</button>
            </div>
            <textarea
              ref={textareaRef}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = 'auto';
                el.style.height = Math.min(el.scrollHeight, 200) + 'px';
              }}
              onKeyDown={(e) => {
                if (e.key === 'Tab') {
                  e.preventDefault();
                  const ta = e.currentTarget;
                  const start = ta.selectionStart;
                  const end = ta.selectionEnd;
                  setCode(code.slice(0, start) + '    ' + code.slice(end));
                  requestAnimationFrame(() => { ta.selectionStart = ta.selectionEnd = start + 4; });
                }
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  if (runMode === 'server') handleRunServer();
                  else handleRun();
                }
              }}
              placeholder={`Write ${defaultLang === 'javascript' ? 'JavaScript' : 'Python'} code... (Ctrl+Enter to run)`}
              className="p-3 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm font-mono resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 w-full"
              style={{ minHeight: 52, maxHeight: 200, overflow: 'auto' }}
              disabled={runMode === 'browser' && defaultLang === 'python' && !workerReady}
            />
          </div>
          <div className="flex flex-col gap-1.5 self-end">
            <button
              onClick={handleAskAI}
              disabled={!code.trim() || isGenerating || isExecuting}
              className="px-3 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium transition-colors flex items-center gap-1.5"
              title="Generate code from description"
            >
              {isGenerating ? (
                <><StopIcon className="w-3.5 h-3.5" />AI...</>
              ) : (
                <><SparklesIcon className="w-3.5 h-3.5" />Ask AI</>
              )}
            </button>
            <div className="flex items-center gap-1">
              <div className="flex items-center gap-0.5 text-[10px] text-chat-text-secondary border border-chat-border rounded-md overflow-hidden">
                <button
                  onClick={() => setRunMode('browser')}
                  className={`px-2 py-1 transition-colors ${runMode === 'browser' ? 'bg-green-600/30 text-green-400' : 'hover:text-chat-text'}`}
                  title="Run in browser (Pyodide)"
                >
                  Browser
                </button>
                <span className="text-chat-border">|</span>
                <button
                  onClick={() => setRunMode('server')}
                  className={`px-2 py-1 transition-colors ${runMode === 'server' ? 'bg-blue-600/30 text-blue-400' : 'hover:text-chat-text'}`}
                  title="Run on server"
                >
                  Server
                </button>
              </div>
              <button
                onClick={runMode === 'server' ? handleRunServer : handleRun}
                disabled={(runMode === 'browser' && defaultLang === 'python' && (!workerReady || !code.trim())) || (runMode === 'server' && !code.trim()) || (defaultLang === 'javascript' && !code.trim()) || isExecuting || isGenerating}
                className="px-3 py-2 rounded-lg bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium transition-colors flex items-center gap-1.5"
              >
                {isExecuting ? (
                  <><StopIcon className="w-3.5 h-3.5" />Run...</>
                ) : (
                  <><PlayIcon className="w-3.5 h-3.5" />Run</>
                )}
              </button>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-between mt-1.5">
          {/* Model selector */}
          <div ref={modelMenuRef} className="relative">
            <button
              type="button"
              onClick={() => setShowModelMenu(p => !p)}
              className="flex items-center gap-1 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
              style={{ background: 'var(--border-subtle)' }}
            >
              <span className="max-w-[140px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
              <ChevronDownIcon className="w-3 h-3 opacity-70" />
            </button>
            {showModelMenu && availableModels.length > 0 && (
              <div style={{ position: 'absolute', bottom: 28, left: 0, width: 220, maxHeight: 280, background: 'var(--surface-1)', border: '1px solid var(--border-default)', borderRadius: 10, overflow: 'hidden', zIndex: 50 }}>
                <div style={{ maxHeight: 280, overflowY: 'auto', padding: 4 }}>
                  <button onClick={() => { setSelectedModel(null); setShowModelMenu(false); }} className="w-full px-2.5 py-1.5 rounded-lg text-xs text-left" style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}>Auto</button>
                  {availableModels.map(m => (
                    <button key={m} onClick={() => { setSelectedModel(m); setShowModelMenu(false); }} className="w-full px-2.5 py-1.5 rounded-lg text-xs text-left truncate" style={{ color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === m ? 'var(--surface-3)' : 'transparent' }}>{m}</button>
                  ))}
                </div>
              </div>
            )}
          </div>
          <span className="text-[10px] text-chat-text-secondary/50">Ctrl+Enter to run · Ask AI to generate code</span>
        </div>
      </div>
    </div>
  );
}
