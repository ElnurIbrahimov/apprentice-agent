import { useState, useEffect, useCallback } from 'react';
import { ChevronDownIcon, ClockIcon } from '@heroicons/react/24/outline';
import { usePolling } from '../hooks/usePolling';
import type { ActivityEvent } from '../types';

// ── Category config ─────────────────────────────────────────────────────────

const CATEGORY_CONFIG: Record<string, {
  icon: string; color: string; bg: string; label: string;
}> = {
  tool:      { icon: '🔧', color: 'text-blue-400',   bg: 'bg-blue-500/10',   label: 'Tools' },
  memory:    { icon: '🧠', color: 'text-purple-400', bg: 'bg-purple-500/10', label: 'Memory' },
  emotion:   { icon: '💜', color: 'text-pink-400',   bg: 'bg-pink-500/10',   label: 'Emotion' },
  proactive: { icon: '⚡', color: 'text-amber-400',  bg: 'bg-amber-500/10',  label: 'Proactive' },
  strategy:  { icon: '🎯', color: 'text-green-400',  bg: 'bg-green-500/10',  label: 'Strategy' },
  system:    { icon: '⚙️', color: 'text-gray-400',   bg: 'bg-gray-500/10',   label: 'System' },
};

// Sub-type icons for proactive daemon events (prevents visual monotony)
const PROACTIVE_SUBTYPE_ICONS: Record<string, string> = {
  prepare: '📋',
  suggest: '💡',
  notify:  '🔔',
  observe: '👁',
  analyze: '📊',
  remind:  '⏰',
  monitor: '📡',
};

const ALL_CATEGORIES = ['all', 'tool', 'memory', 'emotion', 'proactive', 'strategy'] as const;

// ── Helpers ─────────────────────────────────────────────────────────────────

function getTimeGroup(ts: number): string {
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 300) return 'Live';
  if (diff < 86400) return 'Today';
  if (diff < 172800) return 'Yesterday';
  return new Date(ts * 1000).toLocaleDateString();
}

function formatRelTime(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleTimeString();
}

// ── EventCard ────────────────────────────────────────────────────────────────

function getEventIcon(event: ActivityEvent, cfg: typeof CATEGORY_CONFIG[string]): string {
  if (event.category === 'proactive' && event.summary) {
    // Extract sub-type from "Daemon: prepare — ..." format
    const match = event.summary.match(/(?:Daemon:\s*)?(\w+)/i);
    if (match) {
      const subtype = match[1].toLowerCase();
      if (subtype in PROACTIVE_SUBTYPE_ICONS) return PROACTIVE_SUBTYPE_ICONS[subtype];
    }
  }
  return cfg.icon;
}

function EventCard({ event, isExpanded, onToggle }: {
  event: ActivityEvent;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  const cfg = CATEGORY_CONFIG[event.category] ?? CATEGORY_CONFIG.system;
  const icon = getEventIcon(event, cfg);
  return (
    <div className={`rounded-lg border border-chat-border/20 ${cfg.bg} overflow-hidden`}>
      <button
        onClick={onToggle}
        className="w-full px-3 py-2 flex items-center gap-2.5 text-left hover:brightness-110 transition-all"
      >
        <span className={`flex items-center justify-center w-7 h-7 rounded-md text-sm shrink-0 ${cfg.bg}`}>{icon}</span>
        <div className="flex-1 min-w-0">
          <p className="text-xs text-chat-text truncate">{event.summary}</p>
          <p className="text-[10px] text-chat-text-secondary/60 mt-0.5">
            {formatRelTime(event.timestamp)}
            {event.duration_ms != null && ` · ${event.duration_ms}ms`}
          </p>
        </div>
        {event.payload && (
          <ChevronDownIcon
            className={`w-3.5 h-3.5 shrink-0 text-chat-text-secondary/40
              transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          />
        )}
      </button>
      {isExpanded && event.payload && (
        <div className="px-3 pb-2.5">
          <pre className="text-[10px] text-chat-text-secondary/70 bg-chat-bg/60
            rounded p-2 overflow-x-auto whitespace-pre-wrap max-h-32">
            {JSON.stringify(event.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ActivityTimeline() {
  const [events, setEvents] = useState<ActivityEvent[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [lastTimestamp, setLastTimestamp] = useState(0);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [liveCount, setLiveCount] = useState(0);

  // Initial load + filter change: full history, no `after`
  useEffect(() => {
    setEvents([]);
    setLastTimestamp(0);
    setExpanded(new Set());
    const params = new URLSearchParams({ limit: '100' });
    if (filter !== 'all') params.set('categories', filter);
    fetch(`/api/activity/events?${params}`)
      .then(r => r.json())
      .then(data => {
        const evts: ActivityEvent[] = data.events || [];
        setEvents(evts);
        setHasMore((data.count || 0) === 100);
        if (evts.length > 0) setLastTimestamp(evts[0].timestamp);
      })
      .catch(() => {});
  }, [filter]);

  // 3s live poll: only fetch events newer than cursor
  const fetchNew = useCallback(async () => {
    if (lastTimestamp === 0) {
      // Initial poll before first load completes — skip
      return;
    }
    const params = new URLSearchParams({
      limit: '50',
      after: String(lastTimestamp),
    });
    if (filter !== 'all') params.set('categories', filter);
    const res = await fetch(`/api/activity/events?${params}`);
    if (!res.ok) return;
    const data = await res.json();
    const newEvts: ActivityEvent[] = data.events || [];
    if (newEvts.length > 0) {
      setEvents(prev => [...newEvts, ...prev]);
      setLastTimestamp(newEvts[0].timestamp);
      setLiveCount(c => c + newEvts.length);
    }
  }, [lastTimestamp, filter]);

  usePolling(fetchNew, 3000);

  // Load older events
  const loadMore = async () => {
    if (!events.length || loadingMore) return;
    setLoadingMore(true);
    const oldest = events[events.length - 1].timestamp;
    const params = new URLSearchParams({
      limit: '100',
      after: '0',
      before: String(oldest),
    });
    if (filter !== 'all') params.set('categories', filter);
    try {
      const res = await fetch(`/api/activity/events?${params}`);
      if (!res.ok) return;
      const data = await res.json();
      const older: ActivityEvent[] = data.events || [];
      if (older.length > 0) {
        setEvents(prev => [...prev, ...older]);
        setHasMore(older.length === 100);
      } else {
        setHasMore(false);
      }
    } catch {
      // ignore
    } finally {
      setLoadingMore(false);
    }
  };

  const toggleExpand = (id: number) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // Group events by time label
  const groups: { label: string; events: ActivityEvent[] }[] = [];
  let currentLabel = '';
  for (const ev of events) {
    const label = getTimeGroup(ev.timestamp);
    if (label !== currentLabel) {
      currentLabel = label;
      groups.push({ label, events: [ev] });
    } else {
      groups[groups.length - 1].events.push(ev);
    }
  }

  return (
    <div className="h-full flex flex-col bg-chat-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border/30 shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-semibold text-chat-text">Activity Timeline</h2>
          {liveCount > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-green-400">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Live
            </span>
          )}
        </div>
        <span className="text-[10px] text-chat-text-secondary/50">{events.length} events</span>
      </div>

      {/* Filter chips */}
      <div className="flex flex-wrap gap-1.5 px-4 py-2.5 border-b border-chat-border/20 shrink-0">
        {ALL_CATEGORIES.map(cat => {
          const cfg = cat === 'all' ? null : CATEGORY_CONFIG[cat];
          const active = filter === cat;
          return (
            <button
              key={cat}
              onClick={() => setFilter(cat)}
              className={`
                px-2.5 py-1 rounded-full text-[11px] font-medium transition-all
                ${active
                  ? 'bg-chat-accent text-white'
                  : 'bg-chat-border/20 text-chat-text-secondary hover:bg-chat-border/40'}
              `}
            >
              {cfg ? `${cfg.icon} ${cfg.label}` : 'All'}
            </button>
          );
        })}
      </div>

      {/* Event list */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
        {events.length === 0 ? (
          <div className="empty-state">
            <ClockIcon className="empty-state-icon" />
            <p className="empty-state-title">No activity yet</p>
            <p className="empty-state-desc">Events will appear as AURA processes tasks</p>
          </div>
        ) : (
          <>
            {groups.map(group => (
              <div key={group.label}>
                {/* Group label */}
                <div className="flex items-center gap-2 py-2">
                  <span className="text-[10px] font-semibold text-chat-text-secondary/50 uppercase tracking-wider">
                    {group.label}
                  </span>
                  <div className="flex-1 h-px bg-chat-border/20" />
                </div>
                {/* Events */}
                <div className="space-y-1">
                  {group.events.map(ev => (
                    <EventCard
                      key={ev.id}
                      event={ev}
                      isExpanded={expanded.has(ev.id)}
                      onToggle={() => toggleExpand(ev.id)}
                    />
                  ))}
                </div>
              </div>
            ))}

            {/* Load more */}
            {hasMore && (
              <div className="pt-3 pb-2 text-center">
                <button
                  onClick={loadMore}
                  disabled={loadingMore}
                  className="text-xs text-chat-text-secondary/60 hover:text-chat-text-secondary
                    disabled:opacity-40 transition-colors"
                >
                  {loadingMore ? 'Loading…' : 'Load 100 more ↓'}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
