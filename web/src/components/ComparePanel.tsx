import { useState, useRef, useEffect, useCallback } from 'react';
import {
  PaperAirplaneIcon, StopIcon, ArrowsRightLeftIcon,
  ClipboardDocumentIcon, CheckIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
interface ModelState {
  output: string;
  streaming: boolean;
  elapsedMs: number | null;
  error: string | null;
}

const EMPTY_MODEL_STATE: ModelState = {
  output: '',
  streaming: false,
  elapsedMs: null,
  error: null,
};

const PRESETS = [
  { label: 'Creative writing', prompt: 'Write a short story opening (3 sentences) about an astronaut who discovers something unexpected on Mars.' },
  { label: 'Code generation', prompt: 'Write a Python function that takes a list of integers and returns the two numbers that sum closest to zero.' },
  { label: 'Analysis', prompt: 'What are the 3 most important geopolitical risks the world faces in 2026? Be specific and concise.' },
  { label: 'Explain simply', prompt: 'Explain how transformers work in AI — as if I am a 12-year-old who likes video games.' },
];

/* ── Streaming helper ── */
async function streamGenerate(
  prompt: string,
  model: string,
  signal: AbortSignal,
  onChunk: (text: string) => void,
): Promise<void> {
  const res = await fetch('/api/generate/raw', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: prompt, model }),
    signal,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);

  if (!res.body) {
    const text = await res.text();
    onChunk(text);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const raw = decoder.decode(value, { stream: true });
    for (const line of raw.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
          if (text) onChunk(text);
        } catch {
          onChunk(data);
        }
      } else if (line.trim() && !line.startsWith(':')) {
        onChunk(line);
      }
    }
  }
}

/* ── CopyButton ── */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [text]);
  return (
    <button
      onClick={handleCopy}
      disabled={!text}
      title="Copy"
      className="p-1 rounded text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
    >
      {copied
        ? <CheckIcon className="w-3.5 h-3.5 text-green-400" />
        : <ClipboardDocumentIcon className="w-3.5 h-3.5" />}
    </button>
  );
}

/* ── ModelPane ── */
function ModelPane({
  label,
  model,
  models,
  onModelChange,
  state,
}: {
  label: string;
  model: string;
  models: string[];
  onModelChange: (m: string) => void;
  state: ModelState;
}) {
  const outputRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (state.streaming && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [state.output, state.streaming]);

  return (
    <div className="flex flex-col flex-1 min-w-0 border border-chat-border rounded-xl overflow-hidden" style={{ background: 'var(--surface-1)' }}>
      {/* Pane header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-chat-border flex-shrink-0" style={{ background: 'var(--surface-0)' }}>
        <span className="text-xs font-semibold text-chat-text-secondary uppercase tracking-wide">{label}</span>
        <select
          value={model}
          onChange={e => onModelChange(e.target.value)}
          className="flex-1 text-xs rounded-md px-2 py-1 border border-chat-border text-chat-text outline-none focus:border-chat-accent truncate"
          style={{ background: 'var(--surface-2)' }}
        >
          {models.length === 0
            ? <option value="">Loading models...</option>
            : models.map(m => <option key={m} value={m}>{m}</option>)
          }
        </select>
        {state.elapsedMs !== null && (
          <span className="text-[10px] text-chat-text-secondary whitespace-nowrap">{(state.elapsedMs / 1000).toFixed(1)}s</span>
        )}
        <CopyButton text={state.output} />
      </div>

      {/* Output */}
      <div ref={outputRef} className="flex-1 overflow-y-auto p-3 text-sm text-chat-text leading-relaxed">
        {state.error ? (
          <p className="text-red-400 text-xs">{state.error}</p>
        ) : state.output ? (
          <pre className="whitespace-pre-wrap font-sans">{state.output}</pre>
        ) : state.streaming ? (
          <div className="flex items-center gap-2 text-chat-text-secondary text-xs">
            <span className="inline-block w-1.5 h-3 bg-chat-accent animate-pulse rounded-sm" />
            Generating…
          </div>
        ) : (
          <p className="text-chat-text-secondary text-xs">Response will appear here</p>
        )}
        {state.streaming && state.output && (
          <span className="inline-block w-1.5 h-3 bg-chat-accent animate-pulse rounded-sm ml-0.5 align-middle" />
        )}
      </div>
    </div>
  );
}

/* ── Main Component ── */
export function ComparePanel() {
  const [prompt, setPrompt] = useState('');
  const [models, setModels] = useState<string[]>([]);
  const [modelA, setModelA] = useState('');
  const [modelB, setModelB] = useState('');
  const [stateA, setStateA] = useState<ModelState>(EMPTY_MODEL_STATE);
  const [stateB, setStateB] = useState<ModelState>(EMPTY_MODEL_STATE);
  const [isRunning, setIsRunning] = useState(false);

  const abortA = useRef<AbortController | null>(null);
  const abortB = useRef<AbortController | null>(null);

  /* Fetch models */
  useEffect(() => {
    fetch('/api/models')
      .then(r => r.json())
      .then(data => {
        const all: string[] = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        setModels(all);
        if (all.length >= 1) setModelA(all[0]);
        if (all.length >= 2) setModelB(all[1]);
      })
      .catch(() => {});
  }, []);

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      abortA.current?.abort();
      abortB.current?.abort();
    };
  }, []);

  const handleSwap = useCallback(() => {
    const a = modelA;
    const b = modelB;
    setModelA(b);
    setModelB(a);
    setModelB(modelA);
  }, [modelA, modelB]);

  const handleStop = useCallback(() => {
    abortA.current?.abort();
    abortB.current?.abort();
    setIsRunning(false);
    setStateA(p => ({ ...p, streaming: false }));
    setStateB(p => ({ ...p, streaming: false }));
  }, []);

  const handleRun = useCallback(async () => {
    if (!prompt.trim() || isRunning) return;
    if (!modelA || !modelB) return;

    setStateA({ output: '', streaming: true, elapsedMs: null, error: null });
    setStateB({ output: '', streaming: true, elapsedMs: null, error: null });
    setIsRunning(true);

    abortA.current = new AbortController();
    abortB.current = new AbortController();

    const startA = Date.now();
    const startB = Date.now();

    const runA = streamGenerate(
      prompt,
      modelA,
      abortA.current.signal,
      (chunk) => setStateA(p => ({ ...p, output: p.output + chunk })),
    )
      .then(() => setStateA(p => ({ ...p, streaming: false, elapsedMs: Date.now() - startA })))
      .catch(e => {
        if (e.name !== 'AbortError') {
          setStateA(p => ({ ...p, streaming: false, error: e.message }));
        } else {
          setStateA(p => ({ ...p, streaming: false }));
        }
      });

    const runB = streamGenerate(
      prompt,
      modelB,
      abortB.current.signal,
      (chunk) => setStateB(p => ({ ...p, output: p.output + chunk })),
    )
      .then(() => setStateB(p => ({ ...p, streaming: false, elapsedMs: Date.now() - startB })))
      .catch(e => {
        if (e.name !== 'AbortError') {
          setStateB(p => ({ ...p, streaming: false, error: e.message }));
        } else {
          setStateB(p => ({ ...p, streaming: false }));
        }
      });

    await Promise.allSettled([runA, runB]);
    setIsRunning(false);
  }, [prompt, modelA, modelB, isRunning]);

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface-0)' }}>
      {/* Top: prompt input + controls */}
      <div className="flex-shrink-0 border-b border-chat-border px-4 py-3 space-y-3" style={{ background: 'var(--surface-0)' }}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-chat-text">Compare Models</h2>
          <button
            onClick={handleSwap}
            title="Swap models"
            className="flex items-center gap-1 text-xs text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md border border-chat-border"
            style={{ background: 'var(--surface-2)' }}
          >
            <ArrowsRightLeftIcon className="w-3.5 h-3.5" />
            Swap
          </button>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map(p => (
            <button
              key={p.label}
              onClick={() => setPrompt(p.prompt)}
              className="text-[11px] px-2.5 py-1 rounded-full border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-accent transition-colors"
              style={{ background: 'var(--surface-2)' }}
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Prompt + send */}
        <div className="flex gap-2">
          <textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleRun();
              }
            }}
            placeholder="Enter a prompt to compare both models…"
            rows={2}
            disabled={isRunning}
            className="flex-1 p-2.5 rounded-lg border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
            style={{ background: 'var(--surface-2)' }}
          />
          <button
            onClick={isRunning ? handleStop : handleRun}
            disabled={!isRunning && (!prompt.trim() || !modelA || !modelB)}
            className="self-end p-2.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white transition-opacity"
          >
            {isRunning
              ? <StopIcon className="w-4 h-4" />
              : <PaperAirplaneIcon className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Side-by-side panes */}
      <div className="flex-1 overflow-hidden flex flex-col md:flex-row gap-3 p-3">
        <ModelPane
          label="Model A"
          model={modelA}
          models={models}
          onModelChange={setModelA}
          state={stateA}
        />
        <ModelPane
          label="Model B"
          model={modelB}
          models={models}
          onModelChange={setModelB}
          state={stateB}
        />
      </div>
    </div>
  );
}
