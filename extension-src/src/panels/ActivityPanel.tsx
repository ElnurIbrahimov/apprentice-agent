/**
 * ActivityPanel — browse Aura's activity event timeline.
 *
 * Pulls /api/activity/events?limit=200 and groups by day.
 * Click an event that has a URL to open it in a new tab.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Activity, RefreshCw, ExternalLink } from 'lucide-react';
import { activity as activityApi } from '../api/client';
import type { ActivityEvent } from '../api/types';

function formatDay(ts: number): string {
  const d = new Date(ts * 1000);
  const today = new Date();
  const yesterday = new Date(Date.now() - 86400_000);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: d.getFullYear() !== today.getFullYear() ? 'numeric' : undefined });
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function groupByDay(events: ActivityEvent[]): Record<string, ActivityEvent[]> {
  const out: Record<string, ActivityEvent[]> = {};
  for (const e of events) {
    const day = formatDay(e.timestamp);
    if (!out[day]) out[day] = [];
    out[day].push(e);
  }
  return out;
}

export default function ActivityPanel() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await activityApi.events({ limit: 200 });
      setEvents(r.events ?? []);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const categories = Array.from(new Set(events.map((e) => e.category).filter(Boolean))) as string[];
  const filtered = filter === 'all' ? events : events.filter((e) => e.category === filter);
  const grouped = groupByDay(filtered);

  return (
    <div className="panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--b1)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Activity size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          Activity
          <span style={{ color: 'var(--mu)', fontWeight: 400, marginLeft: 6 }}>{filtered.length}</span>
        </span>
        <button
          onClick={load}
          aria-label="Refresh"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <RefreshCw size={12} />
        </button>
      </div>

      {categories.length > 0 && (
        <div style={{ padding: '8px 14px', display: 'flex', gap: 4, overflowX: 'auto', borderBottom: '1px solid var(--b1)' }}>
          <FilterChip label="All" active={filter === 'all'} onClick={() => setFilter('all')} />
          {categories.map((c) => (
            <FilterChip key={c} label={c} active={filter === c} onClick={() => setFilter(c)} />
          ))}
        </div>
      )}

      <div style={{ flex: 1, padding: 14, overflowY: 'auto' }}>
        {loading && <div style={{ color: 'var(--mu)', fontSize: 11 }}>Loading…</div>}

        {Object.entries(grouped).map(([day, evts]) => (
          <div key={day} style={{ marginBottom: 12 }}>
            <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 6, position: 'sticky', top: -14, background: 'var(--bg)', padding: '4px 0' }}>
              {day}
            </div>
            {evts.map((e, i) => (
              <div
                key={e.id || `${day}-${i}`}
                onClick={() => e.url && window.open(e.url, '_blank', 'noopener')}
                style={{
                  display: 'flex',
                  gap: 8,
                  padding: '6px 8px',
                  borderRadius: 6,
                  cursor: e.url ? 'pointer' : 'default',
                  fontSize: 11,
                  alignItems: 'start',
                }}
              >
                <span style={{ fontSize: 9, color: 'var(--mu)', flexShrink: 0, width: 40 }}>
                  {formatTime(e.timestamp)}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {e.title || e.description || e.kind || '(event)'}
                  </div>
                  {e.category && (
                    <div style={{ fontSize: 9, color: 'var(--mu)' }}>{e.category}</div>
                  )}
                </div>
                {e.url && <ExternalLink size={10} style={{ color: 'var(--mu)', flexShrink: 0, marginTop: 2 }} />}
              </div>
            ))}
          </div>
        ))}

        {!loading && filtered.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: 11, textAlign: 'center', padding: 20 }}>
            No activity yet.
          </div>
        )}
      </div>
    </div>
  );
}

function FilterChip({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '3px 10px',
        background: active ? 'var(--p)' : 'var(--s2)',
        border: '1px solid var(--b1)',
        borderRadius: 12,
        color: active ? '#fff' : 'var(--mu)',
        fontSize: 10,
        fontWeight: 500,
        cursor: 'pointer',
        whiteSpace: 'nowrap',
        flexShrink: 0,
      }}
    >
      {label}
    </button>
  );
}
