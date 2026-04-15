/**
 * CalendarPanel — quick calendar view from /api/calendar/*.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { CalendarDays, Plus, RefreshCw, Loader2 } from 'lucide-react';
import { tools } from '../api/client';
import type { CalendarEvent } from '../api/types';

export default function CalendarPanel() {
  const [today, setToday] = useState<CalendarEvent[]>([]);
  const [upcoming, setUpcoming] = useState<CalendarEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newStart, setNewStart] = useState('');
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const [t, u] = await Promise.allSettled([
      tools.calendar.today(),
      tools.calendar.upcoming(7),
    ]);
    if (t.status === 'fulfilled') setToday(t.value.events ?? []);
    if (u.status === 'fulfilled') setUpcoming(u.value.events ?? []);
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const addEvent = useCallback(async () => {
    if (!newTitle.trim() || !newStart.trim()) return;
    setBusy(true);
    try {
      await tools.calendar.add({ title: newTitle.trim(), start: newStart.trim() });
      setNewTitle('');
      setNewStart('');
      setAdding(false);
      load();
    } catch { /* silent */ }
    setBusy(false);
  }, [newTitle, newStart, load]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <CalendarDays size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Calendar</span>
        <button onClick={() => setAdding(true)} style={{ background: 'none', border: '1px solid var(--b1)', borderRadius: 6, padding: '4px 8px', fontSize: 10, color: 'var(--mu)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}>
          <Plus size={11} /> Add
        </button>
        <button onClick={load} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {adding && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="Title"
            style={{ padding: '6px 10px', background: 'var(--bg)', border: '1px solid var(--b1)', borderRadius: 6, color: 'var(--tx)', fontSize: 11 }}
          />
          <input
            value={newStart}
            onChange={(e) => setNewStart(e.target.value)}
            placeholder="Start (ISO, e.g. 2026-04-20T14:00)"
            style={{ padding: '6px 10px', background: 'var(--bg)', border: '1px solid var(--b1)', borderRadius: 6, color: 'var(--tx)', fontSize: 11 }}
          />
          <div style={{ display: 'flex', gap: 6 }}>
            <button onClick={addEvent} disabled={busy} style={{ padding: '6px 12px', background: 'var(--p)', border: 'none', borderRadius: 6, color: '#fff', fontSize: 11, cursor: 'pointer' }}>
              {busy ? <Loader2 size={11} className="spin" /> : 'Save'}
            </button>
            <button onClick={() => setAdding(false)} style={{ padding: '6px 12px', background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 6, color: 'var(--mu)', fontSize: 11, cursor: 'pointer' }}>
              Cancel
            </button>
          </div>
        </div>
      )}

      <Section title="Today" events={today} loading={loading} />
      <Section title="Next 7 days" events={upcoming} loading={loading} />
    </div>
  );
}

function Section({ title, events, loading }: { title: string; events: CalendarEvent[]; loading: boolean }) {
  return (
    <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10 }}>
      <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 6 }}>
        {title}
      </div>
      {loading && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Loading…</div>}
      {!loading && events.length === 0 && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Nothing scheduled.</div>}
      {events.map((e, i) => (
        <div key={e.id || i} style={{ padding: '6px 0', fontSize: 11, color: 'var(--tx)', borderTop: i > 0 ? '1px solid var(--b1)' : 'none' }}>
          <div>{e.title}</div>
          <div style={{ fontSize: 9, color: 'var(--mu)' }}>{e.start}{e.location ? ` · ${e.location}` : ''}</div>
        </div>
      ))}
    </div>
  );
}
