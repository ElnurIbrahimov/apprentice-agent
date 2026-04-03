import { useState, useRef, useCallback, useEffect } from 'react';
import { PlayIcon, StopIcon, ArrowPathIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { highlightCode } from '../utils/codeHighlighter';
import { sanitizeHtml } from '../utils/sanitize';
import { execute, resetRuntime, subscribe, isReady, type OutputBlock, type VariableInfo } from '../utils/pyodideExecutor';

interface Exchange {
  id: string;
  code: string;
  outputs: OutputBlock[];
  variables: VariableInfo[];
  phase: 'idle' | 'executing';
  executionTime?: number;
}

let counter = 0;
function newId() { return `ex-${Date.now()}-${++counter}`; }

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

/* ── Exchange card ── */
function ExchangeCard({ exchange }: { exchange: Exchange }) {
  const [codeHtml, setCodeHtml] = useState('');
  const [showVars, setShowVars] = useState(false);

  useEffect(() => {
    if (exchange.code) {
      const isDark = !document.documentElement.classList.contains('light');
      highlightCode(exchange.code, 'python', isDark ? 'dark' : 'light')
        .then(setCodeHtml)
        .catch(() => {});
    }
  }, [exchange.code]);

  return (
    <div className="border border-chat-border rounded-lg overflow-hidden bg-surface-1">
      {/* Code */}
      <div className="border-b border-chat-border">
        <div className="flex items-center justify-between px-3 py-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-purple-400">Python</span>
          {exchange.phase === 'executing' && (
            <span className="text-[10px] text-yellow-400 animate-pulse">Running...</span>
          )}
          {exchange.executionTime != null && (
            <span className="text-[10px] text-chat-text-secondary">{exchange.executionTime.toFixed(2)}s</span>
          )}
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

/* ── Main component ── */
export function CodeInterpreter() {
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [code, setCode] = useState('');
  const [workerReady, setWorkerReady] = useState(isReady());
  const [loading, setLoading] = useState(isReady() ? '' : 'Initializing...');
  const [isExecuting, setIsExecuting] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

  const handleRun = useCallback(() => {
    const trimmed = code.trim();
    if (!trimmed || !workerReady) return;

    const id = newId();
    setIsExecuting(true);

    setExchanges((prev) => [
      ...prev,
      { id, code: trimmed, outputs: [], variables: [], phase: 'executing' },
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
  }, [code, workerReady]);

  const handleReset = useCallback(() => {
    resetRuntime();
    setExchanges([]);
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
        <button
          onClick={handleReset}
          className="flex items-center gap-1 px-2 py-1 text-xs text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] rounded transition-colors"
          title="Reset runtime"
        >
          <ArrowPathIcon className="w-3.5 h-3.5" />
          Reset
        </button>
      </div>

      {/* Exchanges */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {exchanges.length === 0 && workerReady && (
          <div className="text-center text-chat-text-secondary text-sm py-12">
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
                  className="px-3 py-1.5 text-xs rounded-full border border-chat-border hover:border-purple-500/30 hover:text-chat-text transition-colors"
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {!workerReady && (
          <div className="text-center text-chat-text-secondary text-sm py-12">
            <div className="shimmer-bar h-3 w-48 mx-auto mb-3" />
            <p>{loading || 'Initializing Python runtime...'}</p>
          </div>
        )}

        {exchanges.map((ex) => (
          <ExchangeCard key={ex.id} exchange={ex} />
        ))}
      </div>

      {/* Input */}
      <div className="border-t border-chat-border p-3 flex-shrink-0">
        <div className="flex gap-2">
          <textarea
            ref={textareaRef}
            value={code}
            onChange={(e) => setCode(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                handleRun();
              }
            }}
            placeholder="Write Python code... (Ctrl+Enter to run)"
            className="flex-1 p-3 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm font-mono resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
            rows={3}
            disabled={!workerReady}
          />
          <button
            onClick={handleRun}
            disabled={!workerReady || !code.trim() || isExecuting}
            className="self-end px-4 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium transition-colors flex items-center gap-1.5"
          >
            {isExecuting ? (
              <><StopIcon className="w-4 h-4" />Running</>
            ) : (
              <><PlayIcon className="w-4 h-4" />Run</>
            )}
          </button>
        </div>
        <div className="flex items-center justify-between mt-1.5 text-[10px] text-chat-text-secondary/50">
          <span>Ctrl+Enter to run</span>
          <span>numpy, pandas, matplotlib, scipy, scikit-learn available</span>
        </div>
      </div>
    </div>
  );
}
