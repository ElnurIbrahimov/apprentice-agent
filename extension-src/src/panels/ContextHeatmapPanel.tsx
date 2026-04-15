/**
 * ContextHeatmapPanel — live view of Aura's attention/focus topics.
 *
 * Tabs:
 *  - Heatmap: /api/context/heatmap (weighted chips, size+color per backend)
 *  - Focus:   /api/context/focus   (top items grouped by category)
 *  - Stats:   /api/context/stats   (scalar telemetry)
 *
 * Polls every 5s while mounted.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Flame, RefreshCw, Layers, BarChart2 } from 'lucide-react';
import { context } from '../api/client';
import type { HeatmapResponse, FocusResponse, ContextStats } from '../api/types';

type Tab = 'heatmap' | 'focus' | 'stats';

export default function ContextHeatmapPanel() {
  const [tab, setTab] = useState<Tab>('heatmap');
  const [heatmap, setHeatmap] = useState<HeatmapResponse | null>(null);
  const [focus, setFocus] = useState<FocusResponse | null>(null);
  const [stats, setStats] = useState<ContextStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    const results = await Promise.allSettled([
      context.heatmap(),
      context.focus(20),
      context.stats(),
    ]);
    if (results[0].status === 'fulfilled') setHeatmap(results[0].value);
    if (results[1].status === 'fulfilled') setFocus(results[1].value);
    if (results[2].status === 'fulfilled') setStats(results[2].value);
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAll();
    // 10s interval — was 5s, but fetchAll fans out to 3 endpoints in parallel,
    // so 5s produced 0.6 rps per mounted panel. 10s halves that while still
    // feeling live.
    const id = setInterval(fetchAll, 10000);
    return () => clearInterval(id);
  }, [fetchAll]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Flame size={14} style={{ color: 'var(--p)' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)' }}>Context Heatmap</span>
        </div>
        <button
          onClick={fetchAll}
          aria-label="Refresh"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <RefreshCw size={12} />
        </button>
      </div>

      <div style={{ display: 'flex', gap: 4, padding: 3, background: 'var(--s2)', borderRadius: 8, border: '1px solid var(--b1)' }}>
        <TabButton label="Heatmap" icon={<Flame size={11} />} active={tab === 'heatmap'} onClick={() => setTab('heatmap')} />
        <TabButton label="Focus" icon={<Layers size={11} />} active={tab === 'focus'} onClick={() => setTab('focus')} />
        <TabButton label="Stats" icon={<BarChart2 size={11} />} active={tab === 'stats'} onClick={() => setTab('stats')} />
      </div>

      {loading && (
        <div style={{ color: 'var(--mu)', fontSize: 11 }}>Loading…</div>
      )}

      {tab === 'heatmap' && heatmap && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {heatmap.items.length === 0 && (
            <div style={{ color: 'var(--mu)', fontSize: 11 }}>No active topics.</div>
          )}
          {heatmap.items.map((it, i) => (
            <div
              key={`${it.name}-${i}`}
              title={`${it.category} · weight ${it.weight.toFixed(2)}`}
              style={{
                padding: `${Math.round(4 + it.size * 4)}px ${Math.round(8 + it.size * 6)}px`,
                borderRadius: 12,
                background: it.color,
                opacity: Math.max(0.35, it.opacity),
                fontSize: Math.round(10 + it.size * 4),
                color: '#fff',
                fontWeight: 600,
                letterSpacing: 0.2,
                lineHeight: 1.1,
              }}
            >
              {it.name}
            </div>
          ))}
        </div>
      )}

      {tab === 'focus' && focus && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--mu)' }}>
            {focus.active_count} active · avg intensity {(focus.average_intensity * 100).toFixed(0)}%
          </div>
          {Object.entries(focus.by_category ?? {}).map(([cat, items]) => (
            <div key={cat} style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
              <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 6 }}>
                {cat}
              </div>
              {items.slice(0, 6).map((it) => (
                <div key={it.name} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0' }}>
                  <span style={{ flex: 1, fontSize: 11, color: 'var(--tx)' }}>{it.name}</span>
                  <div style={{ width: 60, height: 3, background: 'var(--b1)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${Math.round(it.intensity * 100)}%`, background: 'var(--p)' }} />
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {tab === 'stats' && stats && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
          <StatCard label="Activations" value={stats.total_activations} />
          <StatCard label="Topics" value={stats.topics_tracked} />
          <StatCard label="Decay cycles" value={stats.decay_cycles} />
          <StatCard label="Current items" value={stats.current_items} />
          <StatCard label="Conv. depth" value={stats.conversation_depth} />
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
        background: active ? 'var(--p)' : 'transparent',
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

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 8, padding: '10px 12px' }}>
      <div style={{ fontSize: 9, color: 'var(--mu)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--tx)' }}>{value}</div>
    </div>
  );
}
