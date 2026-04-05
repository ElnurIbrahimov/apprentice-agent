import { useState, useEffect, useMemo } from 'react';
import { ArrowPathIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';

// ── Types ─────────────────────────────────────────────────────────────────────

type MemorySource = 'episodic' | 'amem' | 'knowledge';

interface UnifiedMemoryEntry {
  id: string;
  source: MemorySource;
  content: string;
  timestamp: number;       // unix seconds
  tags?: string[];
  keywords?: string[];
  importance?: number;     // 0-1
  category?: string;
  nodeType?: string;       // knowledge graph node type
}

// ── Config ────────────────────────────────────────────────────────────────────

const SOURCE_CONFIG: Record<MemorySource, {
  label: string;
  icon: string;
  dotColor: string;
  bgColor: string;
  textColor: string;
  borderColor: string;
}> = {
  episodic:  {
    label: 'Episodic',
    icon: '🧠',
    dotColor: 'bg-purple-500',
    bgColor: 'bg-purple-500/8',
    textColor: 'text-purple-400',
    borderColor: 'border-purple-500/20',
  },
  amem: {
    label: 'AMEM',
    icon: '📝',
    dotColor: 'bg-blue-500',
    bgColor: 'bg-blue-500/8',
    textColor: 'text-blue-400',
    borderColor: 'border-blue-500/20',
  },
  knowledge: {
    label: 'Knowledge',
    icon: '🔗',
    dotColor: 'bg-emerald-500',
    bgColor: 'bg-emerald-500/8',
    textColor: 'text-emerald-400',
    borderColor: 'border-emerald-500/20',
  },
};

const ALL_SOURCES: MemorySource[] = ['episodic', 'amem', 'knowledge'];

// ── Helpers ───────────────────────────────────────────────────────────────────

function toUnixSeconds(value: string | number | undefined | null): number {
  if (!value) return 0;
  if (typeof value === 'number') {
    // Already unix seconds or ms — if > 1e10 it's milliseconds
    return value > 1e10 ? Math.floor(value / 1000) : value;
  }
  const parsed = Date.parse(value);
  return isNaN(parsed) ? 0 : Math.floor(parsed / 1000);
}

function formatRelTime(ts: number): string {
  if (!ts) return 'Unknown time';
  const diffSec = Math.floor(Date.now() / 1000 - ts);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 172800) return 'Yesterday';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function getTimeGroup(ts: number): string {
  if (!ts) return 'Unknown';
  const diffSec = Math.floor(Date.now() / 1000 - ts);
  if (diffSec < 300) return 'Just now';
  if (diffSec < 86400) return 'Today';
  if (diffSec < 172800) return 'Yesterday';
  const d = new Date(ts * 1000);
  return d.toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' });
}

function importanceDots(importance: number): string {
  const filled = Math.round(importance * 5);
  return '●'.repeat(filled) + '○'.repeat(5 - filled);
}

// ── Data fetching ─────────────────────────────────────────────────────────────

async function fetchEpisodic(): Promise<UnifiedMemoryEntry[]> {
  try {
    const res = await fetch('/api/memory/recalls/recent?limit=50');
    if (!res.ok) return [];
    const data = await res.json();
    const items: unknown[] = data.recalls ?? data.memories ?? data.results ?? (Array.isArray(data) ? data : []);
    return items.map((item: unknown, i: number) => {
      const r = item as Record<string, unknown>;
      return {
        id: `episodic-${String(r.id ?? i)}`,
        source: 'episodic' as MemorySource,
        content: String(r.content ?? r.text ?? r.summary ?? ''),
        timestamp: toUnixSeconds(r.timestamp as string | number | undefined),
        importance: typeof r.importance === 'number' ? r.importance : typeof r.relevance === 'number' ? r.relevance : undefined,
        category: r.category as string | undefined,
      };
    }).filter(e => e.content.length > 0);
  } catch {
    return [];
  }
}

async function fetchAMEM(): Promise<UnifiedMemoryEntry[]> {
  try {
    const res = await fetch('/api/features/amem/notes?limit=50');
    if (!res.ok) {
      // Fall back to old endpoint
      const res2 = await fetch('/api/amem/notes?limit=50');
      if (!res2.ok) return [];
      const data2 = await res2.json();
      const notes2: unknown[] = data2.notes ?? [];
      return mapAMEMNotes(notes2);
    }
    const data = await res.json();
    const notes: unknown[] = data.notes ?? (Array.isArray(data) ? data : []);
    return mapAMEMNotes(notes);
  } catch {
    return [];
  }
}

function mapAMEMNotes(notes: unknown[]): UnifiedMemoryEntry[] {
  return notes.map((n: unknown, i: number) => {
    const note = n as Record<string, unknown>;
    return {
      id: `amem-${String(note.id ?? i)}`,
      source: 'amem' as MemorySource,
      content: String(note.content ?? ''),
      timestamp: toUnixSeconds(note.created_at as string | number | undefined ?? note.timestamp as string | number | undefined),
      tags: Array.isArray(note.tags) ? note.tags as string[] : [],
      keywords: Array.isArray(note.keywords) ? note.keywords as string[] : [],
      importance: typeof note.importance === 'number' ? note.importance : undefined,
      category: note.category as string | undefined,
    };
  }).filter(e => e.content.length > 0);
}

async function fetchKnowledge(): Promise<UnifiedMemoryEntry[]> {
  try {
    const res = await fetch('/api/knowledge-graph');
    if (!res.ok) return [];
    const data = await res.json();
    const nodes: unknown[] = data.nodes ?? [];
    return nodes.map((n: unknown, i: number) => {
      const node = n as Record<string, unknown>;
      return {
        id: `knowledge-${String(node.id ?? i)}`,
        source: 'knowledge' as MemorySource,
        content: String(node.label ?? node.content ?? node.name ?? ''),
        timestamp: toUnixSeconds(node.created_at as string | number | undefined ?? node.timestamp as string | number | undefined),
        importance: typeof node.confidence === 'number' ? node.confidence : undefined,
        nodeType: node.type as string | undefined,
      };
    }).filter(e => e.content.length > 0);
  } catch {
    return [];
  }
}

// ── MemoryCard ────────────────────────────────────────────────────────────────

function MemoryCard({ entry }: { entry: UnifiedMemoryEntry }) {
  const [expanded, setExpanded] = useState(false);
  const cfg = SOURCE_CONFIG[entry.source];

  return (
    <div
      className={`relative rounded-xl border ${cfg.borderColor} ${cfg.bgColor} overflow-hidden transition-all duration-150 hover:brightness-110 cursor-pointer`}
      onClick={() => setExpanded(v => !v)}
    >
      <div className="px-3 py-2.5 flex items-start gap-2.5">
        {/* Type icon */}
        <span className="text-base shrink-0 mt-0.5">{cfg.icon}</span>

        <div className="flex-1 min-w-0">
          {/* Content preview */}
          <p className={`text-xs text-chat-text ${expanded ? '' : 'line-clamp-2'} leading-relaxed`}>
            {entry.content}
          </p>

          {/* Tags / keywords row */}
          {(entry.tags?.length || entry.keywords?.length) ? (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {entry.tags?.slice(0, 4).map((tag, i) => (
                <span key={`t${i}`} className={`text-[10px] px-1.5 py-0.5 rounded-full ${cfg.textColor} bg-chat-bg/60`}>
                  #{tag}
                </span>
              ))}
              {entry.keywords?.slice(0, 3).map((kw, i) => (
                <span key={`k${i}`} className="text-[10px] px-1.5 py-0.5 rounded bg-chat-bg/40 text-chat-text-secondary">
                  {kw}
                </span>
              ))}
            </div>
          ) : null}

          {/* Bottom row: source label + importance + timestamp */}
          <div className="flex items-center gap-2 mt-1.5 flex-wrap">
            <span className={`text-[10px] font-semibold uppercase tracking-wider ${cfg.textColor}`}>
              {cfg.label}
            </span>
            {entry.nodeType && (
              <span className="text-[10px] text-chat-text-secondary/60">{entry.nodeType}</span>
            )}
            {entry.category && (
              <span className="text-[10px] text-chat-text-secondary/60">{entry.category}</span>
            )}
            {typeof entry.importance === 'number' && (
              <span className={`text-[10px] font-mono ${cfg.textColor} opacity-60`} title={`Importance: ${Math.round(entry.importance * 100)}%`}>
                {importanceDots(entry.importance)}
              </span>
            )}
            <span className="text-[10px] text-chat-text-secondary/50 ml-auto">
              {formatRelTime(entry.timestamp)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function MemoryTimeline() {
  const [entries, setEntries] = useState<UnifiedMemoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeFilters, setActiveFilters] = useState<Set<MemorySource>>(new Set(ALL_SOURCES));
  const [searchQuery, setSearchQuery] = useState('');

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const [episodic, amem, knowledge] = await Promise.all([
        fetchEpisodic(),
        fetchAMEM(),
        fetchKnowledge(),
      ]);
      const merged = [...episodic, ...amem, ...knowledge].sort((a, b) => b.timestamp - a.timestamp);
      setEntries(merged);
    } catch {
      setError('Failed to load memories. Is the Aura server running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const toggleFilter = (source: MemorySource) => {
    setActiveFilters(prev => {
      const next = new Set(prev);
      if (next.has(source)) {
        // Don't allow deselecting all
        if (next.size === 1) return prev;
        next.delete(source);
      } else {
        next.add(source);
      }
      return next;
    });
  };

  const filtered = useMemo(() => {
    let list = entries.filter(e => activeFilters.has(e.source));
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(e =>
        e.content.toLowerCase().includes(q) ||
        e.tags?.some(t => t.toLowerCase().includes(q)) ||
        e.keywords?.some(k => k.toLowerCase().includes(q)) ||
        e.category?.toLowerCase().includes(q) ||
        e.nodeType?.toLowerCase().includes(q)
      );
    }
    return list;
  }, [entries, activeFilters, searchQuery]);

  // Group by time label (newest first)
  const groups = useMemo(() => {
    const map: { label: string; entries: UnifiedMemoryEntry[] }[] = [];
    let currentLabel = '';
    for (const entry of filtered) {
      const label = getTimeGroup(entry.timestamp);
      if (label !== currentLabel) {
        currentLabel = label;
        map.push({ label, entries: [entry] });
      } else {
        map[map.length - 1].entries.push(entry);
      }
    }
    return map;
  }, [filtered]);

  // Count per source in current entries (unfiltered by source, filtered by search)
  const counts = useMemo(() => {
    const base = searchQuery.trim()
      ? entries.filter(e => {
          const q = searchQuery.toLowerCase();
          return e.content.toLowerCase().includes(q) ||
            e.tags?.some(t => t.toLowerCase().includes(q)) ||
            e.keywords?.some(k => k.toLowerCase().includes(q));
        })
      : entries;
    return {
      episodic: base.filter(e => e.source === 'episodic').length,
      amem: base.filter(e => e.source === 'amem').length,
      knowledge: base.filter(e => e.source === 'knowledge').length,
    };
  }, [entries, searchQuery]);

  return (
    <div className="h-full flex flex-col bg-chat-bg">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border/30 shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-base">🕰️</span>
          <h2 className="text-sm font-semibold text-chat-text">Memory Timeline</h2>
          {!loading && (
            <span className="text-[10px] text-chat-text-secondary/50">{filtered.length} entries</span>
          )}
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="p-1.5 text-chat-text-secondary hover:text-chat-text rounded-lg transition-colors disabled:opacity-40"
          aria-label="Refresh memories"
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Search */}
      <div className="px-4 pt-3 pb-2 shrink-0">
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-chat-text-secondary pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search memories by content, tag, keyword..."
            className="w-full pl-9 pr-4 py-2 rounded-lg text-xs text-chat-text placeholder:text-chat-text-secondary/50 border border-chat-border/30 focus:outline-none focus:border-chat-accent transition-colors"
            style={{ background: 'var(--surface-2)' }}
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-chat-text-secondary/50 hover:text-chat-text text-xs"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* Filter toggles */}
      <div className="flex gap-1.5 px-4 pb-2.5 shrink-0 flex-wrap">
        {ALL_SOURCES.map(source => {
          const cfg = SOURCE_CONFIG[source];
          const active = activeFilters.has(source);
          const count = counts[source];
          return (
            <button
              key={source}
              onClick={() => toggleFilter(source)}
              className={`
                flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium border transition-all
                ${active
                  ? `${cfg.bgColor} ${cfg.textColor} ${cfg.borderColor}`
                  : 'bg-transparent border-chat-border/20 text-chat-text-secondary/50 hover:border-chat-border/40 hover:text-chat-text-secondary'}
              `}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${active ? cfg.dotColor : 'bg-chat-border/40'}`} />
              <span>{cfg.icon} {cfg.label}</span>
              <span className="opacity-60">({count})</span>
            </button>
          );
        })}
      </div>

      {/* Timeline content */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {loading && (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <ArrowPathIcon className="w-6 h-6 text-chat-text-secondary animate-spin" />
            <p className="text-xs text-chat-text-secondary">Loading memories...</p>
          </div>
        )}

        {!loading && error && (
          <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
            <span className="text-3xl">⚠️</span>
            <p className="text-xs text-red-400">{error}</p>
            <button
              onClick={load}
              className="text-xs text-chat-accent hover:underline mt-1"
            >
              Try again
            </button>
          </div>
        )}

        {!loading && !error && filtered.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
            <span className="text-4xl">🧠</span>
            <p className="text-sm font-medium text-chat-text">
              {entries.length === 0
                ? 'No memories yet.'
                : 'No memories match your search.'}
            </p>
            <p className="text-xs text-chat-text-secondary/60 max-w-56 leading-relaxed">
              {entries.length === 0
                ? 'Start chatting and I\'ll remember what matters.'
                : 'Try adjusting your filters or search term.'}
            </p>
          </div>
        )}

        {!loading && !error && groups.length > 0 && (
          <div className="relative">
            {/* Vertical timeline line */}
            <div className="absolute left-[7px] top-6 bottom-0 w-px bg-chat-border/30" aria-hidden="true" />

            <div className="space-y-0">
              {groups.map(group => (
                <div key={group.label} className="mb-4">
                  {/* Group label */}
                  <div className="flex items-center gap-2 mb-3 sticky top-0 py-1" style={{ background: 'var(--bg-base, transparent)', zIndex: 1 }}>
                    {/* Timeline dot for group */}
                    <span className="relative z-10 w-3.5 h-3.5 rounded-full bg-chat-border/60 border-2 border-chat-bg shrink-0" />
                    <span className="text-[10px] font-semibold text-chat-text-secondary/60 uppercase tracking-wider">
                      {group.label}
                    </span>
                    <div className="flex-1 h-px bg-chat-border/20" />
                    <span className="text-[10px] text-chat-text-secondary/40">{group.entries.length}</span>
                  </div>

                  {/* Cards for this group */}
                  <div className="pl-6 space-y-2">
                    {group.entries.map(entry => (
                      <div key={entry.id} className="relative">
                        {/* Connector dot on timeline */}
                        <span
                          className={`absolute -left-[19px] top-3 w-2 h-2 rounded-full ${SOURCE_CONFIG[entry.source].dotColor} border border-chat-bg shrink-0`}
                          aria-hidden="true"
                        />
                        <MemoryCard entry={entry} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
