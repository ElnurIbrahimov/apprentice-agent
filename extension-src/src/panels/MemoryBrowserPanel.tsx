/**
 * MemoryBrowserPanel — unified memory browser.
 *
 * Tabs:
 *  - Recent: GET /api/memory/recent
 *  - Search: GET /api/memory/search?q=
 *  - Recalls: GET /api/memory/recalls/recent + /recalls/stats
 *
 * Distinct from WisebasePanel, which fronts /api/knowledge (curated snippets).
 * This fronts /api/memory (episodic/unified memory).
 */

import React, { useCallback, useEffect, useState } from 'react';
import { HardDrive, Search, Clock, Activity, Trash2, X, RefreshCw } from 'lucide-react';
import { memory } from '../api/client';
import type { MemoryItem, MemoryRecallEvent, MemoryRecallStats } from '../api/types';

type Tab = 'recent' | 'search' | 'recalls';

export default function MemoryBrowserPanel() {
  const [tab, setTab] = useState<Tab>('recent');
  const [recent, setRecent] = useState<MemoryItem[]>([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryItem[]>([]);
  const [selected, setSelected] = useState<MemoryItem | null>(null);
  const [recalls, setRecalls] = useState<MemoryRecallEvent[]>([]);
  const [recallStats, setRecallStats] = useState<MemoryRecallStats | null>(null);
  const [loading, setLoading] = useState(false);

  const loadRecent = useCallback(async () => {
    setLoading(true);
    try {
      const r = await memory.recent(30);
      setRecent(r.memories ?? []);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  const search = useCallback(async () => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    setLoading(true);
    try {
      const r = await memory.search(query.trim());
      setResults(r.results ?? []);
    } catch { /* silent */ }
    setLoading(false);
  }, [query]);

  const loadRecalls = useCallback(async () => {
    setLoading(true);
    try {
      const [r, s] = await Promise.all([
        memory.recalls.recent(30),
        memory.recalls.stats(),
      ]);
      setRecalls(r.events ?? []);
      setRecallStats(s);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  const remove = useCallback(async (id: string) => {
    try {
      await memory.remove(id);
      setRecent((r) => r.filter((m) => m.id !== id));
      setResults((r) => r.filter((m) => m.id !== id));
      if (selected?.id === id) setSelected(null);
    } catch { /* silent */ }
  }, [selected]);

  useEffect(() => {
    if (tab === 'recent') loadRecent();
    else if (tab === 'recalls') loadRecalls();
  }, [tab, loadRecent, loadRecalls]);

  const items = tab === 'search' ? results : recent;

  return (
    <div className="panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--b1)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <HardDrive size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Memory</span>
        <button
          onClick={() => (tab === 'recent' ? loadRecent() : tab === 'recalls' ? loadRecalls() : search())}
          aria-label="Refresh"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <RefreshCw size={12} />
        </button>
      </div>

      <div style={{ padding: '8px 14px', display: 'flex', gap: 4 }}>
        <TabButton label="Recent" icon={<Clock size={11} />} active={tab === 'recent'} onClick={() => setTab('recent')} />
        <TabButton label="Search" icon={<Search size={11} />} active={tab === 'search'} onClick={() => setTab('search')} />
        <TabButton label="Recalls" icon={<Activity size={11} />} active={tab === 'recalls'} onClick={() => setTab('recalls')} />
      </div>

      {tab === 'search' && (
        <div style={{ padding: '0 14px 8px' }}>
          <div style={{ display: 'flex', gap: 6 }}>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && search()}
              placeholder="Search memories…"
              style={{
                flex: 1,
                padding: '6px 10px',
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 6,
                color: 'var(--tx)',
                fontSize: 12,
              }}
            />
            <button
              onClick={search}
              style={{ padding: '6px 12px', background: 'var(--p)', border: 'none', borderRadius: 6, color: '#fff', fontSize: 11, cursor: 'pointer' }}
            >
              <Search size={12} />
            </button>
          </div>
        </div>
      )}

      <div style={{ flex: 1, padding: '0 14px 14px', overflowY: 'auto' }}>
        {loading && <div style={{ color: 'var(--mu)', fontSize: 11 }}>Loading…</div>}

        {tab === 'recalls' && recallStats && (
          <div style={{ marginBottom: 10, display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
            <MiniStat label="Total" value={recallStats.total_recalls} />
            <MiniStat label="AMEM" value={recallStats.amem_recalls} />
            <MiniStat label="RAG" value={recallStats.rag_recalls} />
          </div>
        )}

        {tab !== 'recalls' && items.map((m) => (
          <div
            key={m.id}
            onClick={() => setSelected(m)}
            style={{
              background: selected?.id === m.id ? 'var(--p)' : 'var(--s2)',
              color: selected?.id === m.id ? '#fff' : 'var(--tx)',
              border: '1px solid var(--b1)',
              borderRadius: 8,
              padding: '8px 10px',
              marginBottom: 6,
              cursor: 'pointer',
              fontSize: 11,
              lineHeight: 1.4,
              position: 'relative',
            }}
          >
            <div style={{ display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
              {m.content}
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 6, fontSize: 9, color: selected?.id === m.id ? 'rgba(255,255,255,0.7)' : 'var(--mu)' }}>
              {m.source && <span>{m.source}</span>}
              {m.category && <span>· {m.category}</span>}
              {typeof m.importance === 'number' && <span>· imp {(m.importance * 100).toFixed(0)}%</span>}
              {typeof m.score === 'number' && <span>· score {m.score.toFixed(2)}</span>}
            </div>
          </div>
        ))}

        {tab === 'recalls' && recalls.map((e, i) => (
          <div
            key={i}
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 8,
              padding: '8px 10px',
              marginBottom: 6,
              fontSize: 11,
            }}
          >
            <div style={{ color: 'var(--tx)', marginBottom: 4 }}>{e.query}</div>
            <div style={{ fontSize: 9, color: 'var(--mu)' }}>
              {e.source} · {e.memories_retrieved} retrieved · {new Date(e.timestamp * 1000).toLocaleString()}
            </div>
          </div>
        ))}

        {tab !== 'recalls' && !loading && items.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: 11, textAlign: 'center', padding: 20 }}>
            {tab === 'search' && query ? 'No results.' : 'No memories yet.'}
          </div>
        )}
      </div>

      {selected && (
        <div style={{
          position: 'absolute',
          right: 0,
          top: 0,
          bottom: 0,
          width: '70%',
          background: 'var(--bg)',
          border: '1px solid var(--b1)',
          padding: 14,
          overflowY: 'auto',
          boxShadow: '-4px 0 16px rgba(0, 0, 0, 0.3)',
          zIndex: 5,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--mu)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Memory detail
            </span>
            <button onClick={() => setSelected(null)} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
              <X size={14} />
            </button>
          </div>
          <div style={{ fontSize: 12, color: 'var(--tx)', lineHeight: 1.5, whiteSpace: 'pre-wrap', marginBottom: 12 }}>
            {selected.content}
          </div>
          <div style={{ fontSize: 10, color: 'var(--mu)', marginBottom: 12 }}>
            {selected.source && <div>Source: {selected.source}</div>}
            {selected.category && <div>Category: {selected.category}</div>}
            {selected.tags && selected.tags.length > 0 && <div>Tags: {selected.tags.join(', ')}</div>}
            <div>Stored: {new Date(selected.timestamp * 1000).toLocaleString()}</div>
          </div>
          <button
            onClick={() => remove(selected.id)}
            style={{
              padding: '6px 12px',
              background: 'rgba(248, 113, 113, 0.1)',
              border: '1px solid rgba(248, 113, 113, 0.3)',
              borderRadius: 6,
              color: '#f87171',
              fontSize: 11,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <Trash2 size={11} /> Delete
          </button>
        </div>
      )}
    </div>
  );
}

function TabButton({ label, icon, active, onClick }: { label: string; icon: React.ReactNode; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: '5px 8px',
        border: 'none',
        borderRadius: 6,
        background: active ? 'var(--p)' : 'var(--s2)',
        color: active ? '#fff' : 'var(--mu)',
        fontSize: 10,
        fontWeight: 600,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
      }}
    >
      {icon}
      {label}
    </button>
  );
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 6, padding: '6px 8px', textAlign: 'center' }}>
      <div style={{ fontSize: 8, color: 'var(--mu)', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--tx)' }}>{value}</div>
    </div>
  );
}
