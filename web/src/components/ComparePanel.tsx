import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import {
  PaperAirplaneIcon, StopIcon, ArrowsRightLeftIcon,
  ClipboardDocumentIcon, CheckIcon, EyeIcon, EyeSlashIcon,
  TrashIcon,
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

const VOTES_KEY = 'aura-compare-votes';
const RATINGS_KEY = 'aura-elo-ratings';

interface VoteRecord {
  prompt: string;
  modelA: string;
  modelB: string;
  winner: 'A' | 'B' | 'tie';
  timestamp: number;
}

function calculateElo(votes: VoteRecord[]): Record<string, number> {
  const ratings: Record<string, number> = {};
  const K = 32;
  for (const v of votes) {
    if (!ratings[v.modelA]) ratings[v.modelA] = 1200;
    if (!ratings[v.modelB]) ratings[v.modelB] = 1200;
    const ra = ratings[v.modelA], rb = ratings[v.modelB];
    const ea = 1 / (1 + Math.pow(10, (rb - ra) / 400));
    const eb = 1 - ea;
    const sa = v.winner === 'A' ? 1 : v.winner === 'tie' ? 0.5 : 0;
    const sb = 1 - sa;
    ratings[v.modelA] = Math.round(ra + K * (sa - ea));
    ratings[v.modelB] = Math.round(rb + K * (sb - eb));
  }
  return ratings;
}

function EloLeaderboard() {
  const [sortAsc, setSortAsc] = useState(false);
  const votes: VoteRecord[] = JSON.parse(localStorage.getItem(VOTES_KEY) || '[]');
  const ratings = useMemo(() => calculateElo(votes), [votes]);

  const stats = useMemo(() => {
    const s: Record<string, { wins: number; losses: number; ties: number; comparisons: number }> = {};
    for (const v of votes) {
      for (const m of [v.modelA, v.modelB]) {
        if (!s[m]) s[m] = { wins: 0, losses: 0, ties: 0, comparisons: 0 };
        s[m].comparisons++;
      }
      if (v.winner === 'A') { s[v.modelA].wins++; s[v.modelB].losses++; }
      else if (v.winner === 'B') { s[v.modelB].wins++; s[v.modelA].losses++; }
      else { s[v.modelA].ties++; s[v.modelB].ties++; }
    }
    return s;
  }, [votes]);

  const sorted = useMemo(() =>
    Object.entries(ratings)
      .sort(([, a], [, b]) => sortAsc ? a - b : b - a)
      .map(([model, elo], i) => ({ rank: i + 1, model, elo, ...(stats[model] || { wins: 0, losses: 0, ties: 0, comparisons: 0 }) })),
    [ratings, stats, sortAsc]
  );

  if (sorted.length === 0) {
    return <div className="flex-1 flex items-center justify-center text-chat-text-secondary text-sm">No votes recorded yet. Run a comparison and vote!</div>;
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      <div className="text-[10px] text-chat-text-secondary mb-3">{votes.length} total votes recorded</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="text-chat-text-secondary border-b border-chat-border">
            <th className="text-left py-2 px-1 w-8">#</th>
            <th className="text-left py-2 px-1">Model</th>
            <th className="text-right py-2 px-1 cursor-pointer hover:text-chat-text" onClick={() => setSortAsc(!sortAsc)}>
              ELO {sortAsc ? '↑' : '↓'}
            </th>
            <th className="text-right py-2 px-1">W/L/T</th>
            <th className="text-right py-2 px-1">Win %</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr key={m.model} className="border-b border-chat-border/30 hover:bg-surface-1">
              <td className="py-2 px-1 text-chat-text-secondary">{m.rank}</td>
              <td className="py-2 px-1 text-chat-text font-medium truncate max-w-[180px]">{m.model.split('/').pop()}</td>
              <td className="py-2 px-1 text-right font-mono text-chat-accent font-semibold">{m.elo}</td>
              <td className="py-2 px-1 text-right text-chat-text-secondary">{m.wins}/{m.losses}/{m.ties}</td>
              <td className="py-2 px-1 text-right text-chat-text-secondary">{m.comparisons > 0 ? Math.round(m.wins / m.comparisons * 100) : 0}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VoteHistoryPanel() {
  const [refreshKey, setRefreshKey] = useState(0);
  const votes: VoteRecord[] = JSON.parse(localStorage.getItem(VOTES_KEY) || '[]').reverse();

  const clearHistory = () => {
    localStorage.removeItem(VOTES_KEY);
    localStorage.removeItem(RATINGS_KEY);
    setRefreshKey(k => k + 1);
  };

  if (votes.length === 0) {
    return <div className="flex-1 flex items-center justify-center text-chat-text-secondary text-sm">No vote history yet.</div>;
  }

  return (
    <div className="flex-1 overflow-y-auto p-4" key={refreshKey}>
      <div className="flex items-center justify-between mb-3">
        <span className="text-[10px] text-chat-text-secondary">{votes.length} votes</span>
        <button onClick={clearHistory} className="flex items-center gap-1 text-[10px] text-red-400/60 hover:text-red-400 transition-colors">
          <TrashIcon className="w-3 h-3" />Clear All
        </button>
      </div>
      <div className="space-y-1.5">
        {votes.map((v, i) => (
          <div key={i} className="flex items-center gap-2 text-xs py-1.5 px-2 rounded-lg border border-chat-border/20" style={{ background: 'var(--surface-1)' }}>
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${v.winner === 'A' ? 'bg-green-600/20 text-green-400' : v.winner === 'B' ? 'bg-blue-600/20 text-blue-400' : 'bg-yellow-600/20 text-yellow-400'}`}>
              {v.winner === 'A' ? 'A wins' : v.winner === 'B' ? 'B wins' : 'Tie'}
            </span>
            <span className="text-chat-text-secondary truncate flex-1">{v.prompt.slice(0, 60)}{v.prompt.length > 60 ? '…' : ''}</span>
            <span className="text-[10px] text-chat-text-secondary/50 flex-shrink-0">
              {new Date(v.timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

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
  blindMode,
}: {
  label: string;
  model: string;
  models: string[];
  onModelChange: (m: string) => void;
  state: ModelState;
  blindMode: boolean;
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
        {blindMode ? (
          <span className="flex-1 text-xs text-chat-text-secondary italic px-2 py-1">Hidden</span>
        ) : (
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
        )}
        {state.elapsedMs !== null && (
          <span className="text-[11px] font-medium text-chat-accent whitespace-nowrap">{(state.elapsedMs / 1000).toFixed(1)}s</span>
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
          <div className="empty-state">
            <p className="empty-state-desc">Response will appear here</p>
          </div>
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
  const [blindMode, setBlindMode] = useState(false);
  const [vote, setVote] = useState<'A' | 'B' | 'tie' | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [activeTab, setActiveTab] = useState<'compare' | 'leaderboard' | 'history'>('compare');

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
  }, [modelA, modelB]);

  const handleStop = useCallback(() => {
    abortA.current?.abort();
    abortB.current?.abort();
    setIsRunning(false);
    setStateA(p => ({ ...p, streaming: false }));
    setStateB(p => ({ ...p, streaming: false }));
  }, []);

  const handleVote = useCallback((winner: 'A' | 'B' | 'tie') => {
    setVote(winner);
    if (blindMode) setRevealed(true);

    const votes = JSON.parse(localStorage.getItem(VOTES_KEY) || '[]');
    votes.push({
      prompt, modelA, modelB, winner,
      timestamp: Date.now(),
    });
    const capped = votes.slice(-100);
    localStorage.setItem(VOTES_KEY, JSON.stringify(capped));
    localStorage.setItem(RATINGS_KEY, JSON.stringify(calculateElo(capped)));
  }, [blindMode, prompt, modelA, modelB]);

  const handleRun = useCallback(async () => {
    if (!prompt.trim() || isRunning) return;
    if (!modelA || !modelB) return;

    setStateA({ output: '', streaming: true, elapsedMs: null, error: null });
    setStateB({ output: '', streaming: true, elapsedMs: null, error: null });
    setIsRunning(true);
    setVote(null);
    setRevealed(false);

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

  const isGenerating = stateA.streaming || stateB.streaming;

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface-0)' }}>
      {/* Top: prompt input + controls */}
      <div className="flex-shrink-0 border-b border-chat-border px-4 py-3 space-y-3" style={{ background: 'var(--surface-0)' }}>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-chat-text">Compare Models</h2>
          <div className="flex items-center gap-2">
            {/* Blind mode toggle */}
            <button
              onClick={() => setBlindMode(v => !v)}
              title={blindMode ? 'Blind mode ON — click to reveal' : 'Enable blind mode'}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded-md border transition-colors ${blindMode ? 'border-purple-500 text-purple-400' : 'border-chat-border text-chat-text-secondary hover:text-chat-text'}`}
              style={{ background: 'var(--surface-2)' }}
            >
              {blindMode
                ? <EyeSlashIcon className="w-3.5 h-3.5" />
                : <EyeIcon className="w-3.5 h-3.5" />}
              {blindMode ? 'Blind' : 'Blind'}
            </button>
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

      {/* Tab bar */}
      <div className="flex gap-1 px-4 py-1.5 border-b border-chat-border flex-shrink-0">
        {([['compare', 'Compare'], ['leaderboard', 'Leaderboard'], ['history', 'History']] as const).map(([id, label]) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`text-xs px-3 py-1.5 rounded-md transition-colors ${activeTab === id ? 'bg-chat-accent text-white' : 'text-chat-text-secondary hover:text-chat-text'}`}
            style={activeTab !== id ? { background: 'var(--surface-1)' } : undefined}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === 'leaderboard' && <EloLeaderboard />}
      {activeTab === 'history' && <VoteHistoryPanel />}

      {/* Side-by-side panes */}
      {activeTab === 'compare' && <><div className="flex-1 overflow-hidden flex flex-col md:flex-row gap-3 p-3">
        <ModelPane
          label={blindMode ? 'Response A' : 'Model A'}
          model={modelA}
          models={models}
          onModelChange={setModelA}
          state={stateA}
          blindMode={blindMode}
        />
        <ModelPane
          label={blindMode ? 'Response B' : 'Model B'}
          model={modelB}
          models={models}
          onModelChange={setModelB}
          state={stateB}
          blindMode={blindMode}
        />
      </div>

      {/* Voting row */}
      {!isGenerating && stateA.output && stateB.output && (
        <div className="flex-shrink-0 border-t border-chat-border animate-slide-up-fade" style={{ background: 'var(--surface-0)' }}>
          <div className="flex items-center justify-center gap-3 py-3">
            <button
              onClick={() => handleVote('A')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors animation-delay-100 ${vote === 'A' ? 'bg-green-600 text-white' : 'bg-surface-2 text-chat-text-secondary hover:text-chat-text'}`}
              style={vote !== 'A' ? { background: 'var(--surface-2)' } : undefined}
            >
              A wins — {blindMode ? 'Response A' : modelA || 'Model A'}
            </button>
            <button
              onClick={() => handleVote('tie')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors animation-delay-200 ${vote === 'tie' ? 'bg-yellow-600 text-white' : 'bg-surface-2 text-chat-text-secondary hover:text-chat-text'}`}
              style={vote !== 'tie' ? { background: 'var(--surface-2)' } : undefined}
            >
              Tie
            </button>
            <button
              onClick={() => handleVote('B')}
              className={`px-4 py-2 rounded-lg text-sm transition-colors animation-delay-300 ${vote === 'B' ? 'bg-green-600 text-white' : 'bg-surface-2 text-chat-text-secondary hover:text-chat-text'}`}
              style={vote !== 'B' ? { background: 'var(--surface-2)' } : undefined}
            >
              B wins — {blindMode ? 'Response B' : modelB || 'Model B'}
            </button>
          </div>

          {/* Reveal after voting in blind mode */}
          {blindMode && revealed && (
            <div className="text-center text-xs text-chat-text-secondary pb-3">
              Response A = <span className="text-purple-400">{modelA}</span> · Response B = <span className="text-blue-400">{modelB}</span>
            </div>
          )}
        </div>
      )}
      </>}
    </div>
  );
}
