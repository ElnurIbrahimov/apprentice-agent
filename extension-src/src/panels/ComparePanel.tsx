import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useStore } from '../store';
import { HTTP, apiFetch, getAuthHeaders } from '../api';
import { md } from '../markdown';
import { Copy, Check, Star, Eye, EyeOff, Trophy } from 'lucide-react';

const COMPARE_DEFAULTS = ['minimax-m2.7:cloud', 'qwen3.5:397b-cloud', 'kimi-k2.6:cloud'];

// ELO rating system
const ELO_KEY = 'aura-compare-elo';
const VOTE_KEY = 'aura-compare-votes';
function loadElo(): Record<string, number> {
  try { return JSON.parse(localStorage.getItem(ELO_KEY) || '{}'); } catch { return {}; }
}
function saveElo(elo: Record<string, number>) { localStorage.setItem(ELO_KEY, JSON.stringify(elo)); }
function loadVotes(): Array<{ winner: string; loser: string; prompt: string; date: number }> {
  try { return JSON.parse(localStorage.getItem(VOTE_KEY) || '[]'); } catch { return []; }
}
function saveVotes(v: typeof loadVotes extends () => infer R ? R : never) { localStorage.setItem(VOTE_KEY, JSON.stringify(v.slice(-100))); }
function calcElo(winner: number, loser: number, k = 32): [number, number] {
  const expected = 1 / (1 + Math.pow(10, (loser - winner) / 400));
  return [Math.round(winner + k * (1 - expected)), Math.round(loser + k * (0 - (1 - expected)))];
}

export default function ComparePanel() {
  const { mdlCloudList, mdlLocalList, setMdlLists, activePanel, setPendingCtx, setPanel } = useStore();
  const [selected, setSelected] = useState<Set<string>>(new Set(COMPARE_DEFAULTS));
  const [loaded, setLoaded] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [fastest, setFastest] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [markedBest, setMarkedBest] = useState('');
  const [copied, setCopied] = useState('');
  const [blindMode, setBlindMode] = useState(false);
  const [shuffledResults, setShuffledResults] = useState<any[]>([]);
  const [revealed, setRevealed] = useState(false);
  const [elo, setElo] = useState(loadElo);
  const [showLeaderboard, setShowLeaderboard] = useState(false);

  // Shuffle results when blind mode is on
  useEffect(() => {
    if (blindMode && results.length > 0 && !revealed) {
      // Fisher-Yates unbiased shuffle
      const arr = [...results];
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
      setShuffledResults(arr);
    } else {
      setShuffledResults(results);
    }
  }, [results, blindMode, revealed]);

  const voteForModel = useCallback((winnerModel: string) => {
    // Update ELO for winner vs all others
    const newElo = { ...elo };
    const loserModels = results.filter(r => r.model !== winnerModel && !r.error).map(r => r.model);
    for (const loser of loserModels) {
      const wRating = newElo[winnerModel] || 1200;
      const lRating = newElo[loser] || 1200;
      const [newW, newL] = calcElo(wRating, lRating);
      newElo[winnerModel] = newW;
      newElo[loser] = newL;
    }
    setElo(newElo);
    saveElo(newElo);
    // Save vote
    const prompt = inputRef.current?.value || '';
    const votes = loadVotes();
    for (const loser of loserModels) {
      votes.push({ winner: winnerModel, loser, prompt: prompt.slice(0, 100), date: Date.now() });
    }
    saveVotes(votes);
    setMarkedBest(winnerModel);
    if (blindMode) setRevealed(true);
  }, [elo, results, blindMode]);

  const gridCols = selected.size <= 3 ? '1fr' : '1fr 1fr';

  const copyResponse = useCallback((model: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(model);
      setTimeout(() => setCopied(''), 1500);
    }).catch(() => {});
  }, []);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (activePanel !== 'compare' || loaded) return;
    const load = async () => {
      try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);
        const d = await fetch('http://localhost:11434/api/tags', { signal: controller.signal }).then(r => r.json());
        clearTimeout(timeout);
        const all: string[] = (d.models || []).map((m: any) => m.name);
        setMdlLists(all.filter(n => n.includes(':cloud')), all.filter(n => !n.includes(':cloud')));
      } catch {
        try {
          const d = await fetch(`${HTTP}/api/models/available`, { headers: getAuthHeaders() }).then(r => r.json());
          setMdlLists((d.cloud || []).map((m: any) => m.name), (d.local || []).map((m: any) => m.name));
        } catch {}
      }
      setLoaded(true);
    };
    load();
  }, [activePanel, loaded]);

  const toggle = (model: string) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(model)) next.delete(model);
      else next.add(model);
      return next;
    });
  };

  const run = async () => {
    const prompt = inputRef.current?.value.trim();
    if (!prompt || !selected.size) return;
    setLoading(true);
    setError('');
    setResults([]);
    try {
      const data = await apiFetch(`${HTTP}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt, models: [...selected] }),
      });
      setResults(data.results || []);
      setFastest(data.fastest || '');
    } catch (err: any) {
      setError(err.message || 'Compare failed');
    } finally {
      setLoading(false);
    }
  };

  const allModels = [...mdlCloudList, ...mdlLocalList];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Model chips */}
      <div
        className="flex-shrink-0 p-3 overflow-x-auto"
        style={{ borderBottom: '1px solid var(--b1)' }}
      >
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Models</span>
          <div className="flex gap-2" style={{ alignItems: 'center' }}>
            <button
              onClick={() => setBlindMode(b => !b)}
              title={blindMode ? 'Disable blind mode' : 'Enable blind mode (hide model names)'}
              style={{ background: blindMode ? 'var(--pg2)' : 'none', border: blindMode ? '1px solid var(--p)' : 'none', cursor: 'pointer', color: blindMode ? 'var(--pl)' : 'var(--mu)', fontSize: '11px', fontFamily: 'inherit', borderRadius: 'var(--r-sm)', padding: '2px 6px', display: 'flex', alignItems: 'center', gap: 3 }}
            >
              {blindMode ? <EyeOff size={10} /> : <Eye size={10} />}
              {blindMode ? 'Blind' : 'Blind'}
            </button>
            <button
              onClick={() => setShowLeaderboard(l => !l)}
              title="ELO Leaderboard"
              style={{ background: showLeaderboard ? 'var(--pg2)' : 'none', border: showLeaderboard ? '1px solid var(--p)' : 'none', cursor: 'pointer', color: showLeaderboard ? 'var(--pl)' : 'var(--mu)', fontSize: '11px', fontFamily: 'inherit', borderRadius: 'var(--r-sm)', padding: '2px 6px', display: 'flex', alignItems: 'center', gap: 3 }}
            >
              <Trophy size={10} /> ELO
            </button>
            <button
              onClick={() => setSelected(new Set(allModels))}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', fontSize: '11px', fontFamily: 'inherit' }}
            >
              All
            </button>
            <button
              onClick={() => setSelected(new Set())}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', fontSize: '11px', fontFamily: 'inherit' }}
            >
              Clear
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {allModels.length === 0 && !loaded && (
            <span style={{ color: 'var(--mu)', fontSize: '11px' }}>Loading models…</span>
          )}
          {allModels.length === 0 && loaded && (
            <span style={{ color: 'var(--mu)', fontSize: '11px' }}>No models — is Ollama running?</span>
          )}
          {allModels.map(m => {
            const isCloud = m.includes(':cloud');
            const on = selected.has(m);
            return (
              <button
                key={m}
                onClick={() => toggle(m)}
                title={m}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  padding: '3px 9px',
                  background: on ? 'var(--pg2)' : 'var(--s2)',
                  border: `1px solid ${on ? 'var(--p)' : 'var(--b1)'}`,
                  borderRadius: 'var(--r-pill)',
                  color: on ? 'var(--pl)' : 'var(--mu)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  maxWidth: 150,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                <span style={{ flexShrink: 0 }}>{isCloud ? '☁' : '🖥'}</span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{m.replace(/:cloud$/, '')}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Input */}
      <div className="flex gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <textarea
          ref={inputRef}
          placeholder="Enter prompt to compare across models…"
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); } }}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '7px 10px',
            resize: 'none',
            height: 60,
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={run}
          disabled={loading || !selected.size}
          style={{
            background: loading || !selected.size ? 'var(--s3)' : 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '8px 14px',
            cursor: loading || !selected.size ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
            alignSelf: 'flex-end',
          }}
        >
          {loading ? '…' : 'Run'}
        </button>
      </div>

      {/* ELO Leaderboard */}
      {showLeaderboard && (
        <div className="flex-shrink-0 p-3" style={{ borderBottom: '1px solid var(--b1)', maxHeight: 180, overflowY: 'auto' }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--mu)', marginBottom: 6 }}>🏆 ELO Leaderboard</div>
          {Object.keys(elo).length === 0 ? (
            <div style={{ fontSize: 11, color: 'var(--di)' }}>No votes yet. Compare models and pick a winner!</div>
          ) : (
            Object.entries(elo)
              .sort(([, a], [, b]) => b - a)
              .map(([model, rating], i) => (
                <div key={model} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <span style={{ fontSize: 10, color: i === 0 ? '#fbbf24' : 'var(--di)', width: 16, fontWeight: 600 }}>
                    {i === 0 ? '👑' : `${i + 1}.`}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--tx)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {model.replace(/:cloud$/, '')}
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: rating >= 1200 ? '#34d399' : '#f87171' }}>
                    {rating}
                  </span>
                </div>
              ))
          )}
        </div>
      )}

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-3">
        {error && <div style={{ color: 'var(--rd)', fontSize: '12px', marginBottom: 8 }}>⚠ {error}</div>}

        {loading && (
          <div style={{ display: 'grid', gridTemplateColumns: gridCols, gap: 8 }}>
            {[...selected].map(m => (
              <div key={m} style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: 12, minHeight: 100 }}>
                <div style={{ fontSize: '11px', color: 'var(--mu)', marginBottom: 8 }}>{m.replace(/:cloud$/, '')}</div>
                <div className="dots"><span /><span /><span /></div>
              </div>
            ))}
          </div>
        )}

        {shuffledResults.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: gridCols, gap: 8 }}>
            {shuffledResults.map((r, idx) => {
              const isCloud = r.model.includes(':cloud');
              const isFastest = r.model === fastest && !r.error;
              const isBest = r.model === markedBest;
              const displayName = (blindMode && !revealed) ? `Model ${String.fromCharCode(65 + idx)}` : r.model.replace(/:cloud$/, '');
              const timeLabel = r.elapsed_ms >= 1000 ? (r.elapsed_ms / 1000).toFixed(1) + 's' : r.elapsed_ms + 'ms';
              return (
                <div
                  key={r.model}
                  style={{
                    background: 'var(--s2)',
                    border: `1px solid ${isBest ? 'var(--gr)' : isFastest ? 'rgba(34,197,94,0.3)' : 'var(--b1)'}`,
                    borderRadius: 'var(--r-md)',
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                  }}
                >
                  <div
                    className="flex items-center justify-between"
                    style={{ padding: '8px 10px', borderBottom: '1px solid var(--b1)', background: isBest ? 'rgba(34,197,94,0.08)' : 'var(--s3)' }}
                  >
                    <div className="flex items-center gap-1.5" style={{ fontSize: '11px', color: 'var(--tx)', minWidth: 0 }}>
                      <span>{isCloud ? '☁' : '🖥'}</span>
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{displayName}</span>
                      {isBest && <span style={{ fontSize: '10px', color: 'var(--gr)' }}>★ Best</span>}
                    </div>
                    <span
                      style={{
                        fontSize: '10px',
                        color: isFastest ? 'var(--gr)' : 'var(--mu)',
                        flexShrink: 0,
                        fontWeight: isFastest ? 600 : 400,
                      }}
                    >
                      {isFastest ? '⚡ ' : ''}{timeLabel}
                    </span>
                  </div>
                  <div
                    style={{
                      flex: 1,
                      padding: '8px 10px',
                      fontSize: '11.5px',
                      color: r.error ? 'var(--rd)' : 'var(--tx)',
                      lineHeight: 1.55,
                      maxHeight: 200,
                      overflowY: 'auto',
                    }}
                  >
                    {r.error ? (
                      `Error: ${r.error}`
                    ) : (
                      <div
                        className="md-body"
                        dangerouslySetInnerHTML={{ __html: md(r.response || '') }}
                      />
                    )}
                  </div>
                  {!r.error && (
                    <div className="flex items-center gap-2" style={{ padding: '6px 10px', borderTop: '1px solid var(--b1)' }}>
                      <button
                        onClick={() => {
                          setPendingCtx({ text: r.response || '', title: displayName, url: '' });
                          setPanel('chat');
                        }}
                        style={{ background: 'none', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', color: 'var(--mu)', fontSize: '10px', padding: '3px 8px', cursor: 'pointer', fontFamily: 'inherit' }}
                      >
                        Send to Chat
                      </button>
                      <button
                        onClick={() => copyResponse(r.model, r.response || '')}
                        style={{ background: 'none', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)', color: 'var(--mu)', fontSize: '10px', padding: '3px 8px', cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 3 }}
                      >
                        {copied === r.model ? <><Check size={9} /> Copied</> : <><Copy size={9} /> Copy</>}
                      </button>
                      <button
                        onClick={() => voteForModel(r.model)}
                        disabled={!!markedBest}
                        title={isBest ? 'Winner!' : 'Vote as best response'}
                        style={{
                          background: isBest ? 'rgba(34,197,94,0.15)' : 'none',
                          border: isBest ? '1px solid rgba(34,197,94,0.3)' : '1px solid var(--b1)',
                          borderRadius: 'var(--r-sm)',
                          cursor: markedBest ? 'default' : 'pointer',
                          color: isBest ? '#34d399' : 'var(--mu)',
                          padding: '3px 8px',
                          fontFamily: 'inherit',
                          fontSize: '10px',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 3,
                          opacity: markedBest && !isBest ? 0.4 : 1,
                        }}
                      >
                        {isBest ? <><Trophy size={9} /> Winner</> : <><Star size={9} /> Vote</>}
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {!loading && !results.length && !error && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            Select models above and enter a prompt to compare
          </div>
        )}
      </div>
    </div>
  );
}
