import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Search, Trash2, ExternalLink, Download, Highlighter, BookOpen, ChevronDown, ChevronRight } from 'lucide-react';
import { HTTP, apiFetch } from '../api';
import { useStore } from '../store';
import { sendMsg } from '../ext';

// ── Types ────────────────────────────────────────────────────────────────────

interface Entry {
  id: string;
  title?: string;
  text?: string;
  source_type?: string;
  created_at?: string;
}

interface HighlightData {
  id: string;
  url: string;
  text: string;
  xpath: string;
  context: string;
  timestamp: number;
  color: string;
  pageTitle: string;
  stale?: boolean;
}

type Tab = 'knowledge' | 'highlights';

// ── Helpers ──────────────────────────────────────────────────────────────────

function truncate(s: string, max: number): string {
  if (s.length <= max) return s;
  return s.slice(0, max) + '...';
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function domainFromUrl(url: string): string {
  try { return new URL(url).hostname; } catch { return url; }
}

// ── Highlights Tab ───────────────────────────────────────────────────────────

function HighlightsTab() {
  const [store, setStore] = useState<Record<string, HighlightData[]>>({});
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<HighlightData[] | null>(null);
  const [expandedUrls, setExpandedUrls] = useState<Set<string>>(new Set());
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const loadAll = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await sendMsg({ type: 'GET_ALL_HIGHLIGHTS' });
      if (resp?.ok) setStore(resp.store || {});
    } catch { /* */ }
    setLoading(false);
  }, []);

  useEffect(() => { loadAll(); }, [loadAll]);

  const handleSearch = (q: string) => {
    setQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(async () => {
      if (q.length >= 2) {
        const resp = await sendMsg({ type: 'SEARCH_HIGHLIGHTS', query: q });
        if (resp?.ok) setSearchResults(resp.highlights || []);
      } else {
        setSearchResults(null);
        if (q.length === 0) loadAll();
      }
    }, 300);
  };

  const deleteHighlight = async (id: string, url: string) => {
    await sendMsg({ type: 'DELETE_HIGHLIGHT', id, url });
    // Update local state
    setStore(prev => {
      const next = { ...prev };
      if (next[url]) {
        next[url] = next[url].filter(h => h.id !== id);
        if (next[url].length === 0) delete next[url];
      }
      return next;
    });
    if (searchResults) {
      setSearchResults(prev => prev ? prev.filter(h => h.id !== id) : null);
    }
  };

  const clearUrl = async (url: string) => {
    await sendMsg({ type: 'CLEAR_URL_HIGHLIGHTS', url });
    setStore(prev => {
      const next = { ...prev };
      delete next[url];
      return next;
    });
  };

  const scrollTo = async (id: string, url: string) => {
    await sendMsg({ type: 'SCROLL_TO_HIGHLIGHT_PAGE', id, url });
  };

  const toggleUrl = (url: string) => {
    setExpandedUrls(prev => {
      const next = new Set(prev);
      if (next.has(url)) next.delete(url);
      else next.add(url);
      return next;
    });
  };

  const exportMarkdown = () => {
    const lines: string[] = ['# AURA Highlights\n'];
    const urls = Object.keys(store).sort();
    for (const url of urls) {
      const highlights = store[url];
      if (!highlights.length) continue;
      const title = highlights[0].pageTitle || domainFromUrl(url);
      lines.push(`\n## ${title}\n`);
      lines.push(`> ${url}\n`);
      for (const hl of highlights) {
        lines.push(`- "${hl.text}" _(${relativeTime(hl.timestamp)})_`);
      }
    }
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `aura-highlights-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // Total count
  let totalCount = 0;
  for (const url of Object.keys(store)) totalCount += store[url].length;

  // Render a single highlight card
  const renderHighlight = (hl: HighlightData, showUrl = false) => (
    <div
      key={hl.id}
      style={{
        background: 'var(--s2)',
        border: hl.stale ? '1px dashed rgba(124, 58, 237, 0.3)' : '1px solid var(--b1)',
        borderRadius: 'var(--r-md)',
        padding: '8px 10px',
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div
          style={{ flex: 1, minWidth: 0, cursor: 'pointer' }}
          onClick={() => scrollTo(hl.id, hl.url)}
          title="Click to scroll to highlight on page"
        >
          <div style={{
            fontSize: '12px',
            color: 'var(--tx)',
            borderLeft: '2px solid rgba(124, 58, 237, 0.5)',
            paddingLeft: 8,
            overflow: 'hidden',
            display: '-webkit-box',
            WebkitLineClamp: 3,
            WebkitBoxOrient: 'vertical' as const,
          }}>
            {truncate(hl.text, 200)}
          </div>
          {showUrl && (
            <div style={{ fontSize: '10px', color: 'var(--pl)', marginTop: 4, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const }}>
              {hl.pageTitle || domainFromUrl(hl.url)}
            </div>
          )}
          <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 3 }}>
            {relativeTime(hl.timestamp)}{hl.stale ? ' (page changed)' : ''}
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); deleteHighlight(hl.id, hl.url); }}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 4, flexShrink: 0 }}
          title="Delete highlight"
        >
          <Trash2 size={12} />
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Search + actions */}
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div className="flex-1 relative">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: 'var(--mu)' }} />
          <input
            value={query}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search highlights..."
            style={{
              width: '100%', background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
              padding: '6px 8px 6px 28px', outline: 'none', fontFamily: 'inherit',
            }}
          />
        </div>
        <button
          onClick={exportMarkdown}
          title="Export as Markdown"
          style={{
            background: 'var(--s2)', border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)', color: 'var(--mu)',
            fontSize: '11px', padding: '5px 8px', cursor: 'pointer',
            display: 'flex', alignItems: 'center', gap: 4,
          }}
        >
          <Download size={12} />
        </button>
      </div>

      {/* Count bar */}
      <div style={{ padding: '6px 12px', fontSize: '10px', color: 'var(--di)', borderBottom: '1px solid var(--b1)' }}>
        {totalCount} highlight{totalCount !== 1 ? 's' : ''} across {Object.keys(store).length} page{Object.keys(store).length !== 1 ? 's' : ''}
        {totalCount >= 900 && (
          <span style={{ color: 'var(--am)', marginLeft: 8 }}>
            Approaching limit (1000 max)
          </span>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {loading && (
          <div className="flex justify-center mt-8">
            <div className="dots"><span /><span /><span /></div>
          </div>
        )}

        {/* Search results mode */}
        {searchResults !== null && (
          <>
            {searchResults.length === 0 && (
              <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
                No highlights match "{query}"
              </div>
            )}
            {searchResults.map(hl => renderHighlight(hl, true))}
          </>
        )}

        {/* Grouped by URL mode */}
        {searchResults === null && !loading && totalCount === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            No highlights yet. Select text on any page and click "Save" to create a highlight.
          </div>
        )}

        {searchResults === null && Object.keys(store)
          .sort((a, b) => {
            const aMax = Math.max(...(store[a]?.map(h => h.timestamp) || [0]));
            const bMax = Math.max(...(store[b]?.map(h => h.timestamp) || [0]));
            return bMax - aMax;
          })
          .map(url => {
            const highlights = store[url];
            if (!highlights || highlights.length === 0) return null;
            const expanded = expandedUrls.has(url);
            const title = highlights[0].pageTitle || domainFromUrl(url);

            return (
              <div key={url}>
                {/* URL group header */}
                <div
                  className="flex items-center gap-2"
                  style={{
                    padding: '6px 8px', cursor: 'pointer',
                    borderRadius: 'var(--r-md)',
                    background: expanded ? 'rgba(124, 58, 237, 0.06)' : 'transparent',
                    transition: 'background 0.12s',
                  }}
                  onClick={() => toggleUrl(url)}
                >
                  {expanded ? <ChevronDown size={13} style={{ color: 'var(--pl)', flexShrink: 0 }} /> : <ChevronRight size={13} style={{ color: 'var(--mu)', flexShrink: 0 }} />}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{
                      fontSize: '12px', fontWeight: 500, color: 'var(--tx)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const,
                    }}>
                      {title}
                    </div>
                    <div style={{
                      fontSize: '10px', color: 'var(--di)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' as const,
                    }}>
                      {domainFromUrl(url)} - {highlights.length} highlight{highlights.length !== 1 ? 's' : ''}
                    </div>
                  </div>
                  <div className="flex items-center gap-1" style={{ flexShrink: 0 }}>
                    <button
                      onClick={(e) => { e.stopPropagation(); window.open(url, '_blank'); }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 3 }}
                      title="Open page"
                    >
                      <ExternalLink size={12} />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); clearUrl(url); }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 3 }}
                      title="Clear all highlights for this page"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                </div>

                {/* Expanded highlights */}
                {expanded && (
                  <div className="flex flex-col gap-1.5" style={{ paddingLeft: 20, paddingTop: 4, paddingBottom: 4 }}>
                    {highlights
                      .sort((a, b) => b.timestamp - a.timestamp)
                      .map(hl => renderHighlight(hl))}
                  </div>
                )}
              </div>
            );
          })}
      </div>
    </>
  );
}

// ── Knowledge Tab ────────────────────────────────────────────────────────────

function KnowledgeTab() {
  const { activePanel } = useStore();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = async (q?: string) => {
    setLoading(true);
    setError('');
    try {
      const url = q
        ? `${HTTP}/api/knowledge/search?q=${encodeURIComponent(q)}&limit=20`
        : `${HTTP}/api/knowledge/list?limit=50`;
      const data = await apiFetch(url);
      const rawItems = data.items || data.entries || data.results || [];
      setEntries(rawItems.map((item: any) => ({
        id: item.episode_id || item.id,
        title: item.title,
        text: item.content || item.text,
        source_type: item.source_type,
        created_at: item.saved_at || item.created_at,
      })));
    } catch (err: any) {
      setError(err.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activePanel === 'wisebase') load();
  }, [activePanel]);

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleSearch = (q: string) => {
    setQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      if (q.length >= 2) load(q);
      else if (q.length === 0) load();
    }, 300);
  };

  const deleteEntry = async (id: string) => {
    try {
      await apiFetch(`${HTTP}/api/knowledge/${id}`, { method: 'DELETE' });
      setEntries(e => e.filter(x => x.id !== id));
    } catch {}
  };

  return (
    <>
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div className="flex-1 relative">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: 'var(--mu)' }} />
          <input
            value={query}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search knowledge base..."
            style={{
              width: '100%', background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
              padding: '6px 8px 6px 28px', outline: 'none', fontFamily: 'inherit',
            }}
          />
        </div>
        <button
          onClick={() => { setQuery(''); load(); }}
          style={{
            background: 'var(--s2)', border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)', color: 'var(--mu)',
            fontSize: '11px', padding: '6px 10px', cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          All
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {loading && (
          <div className="flex justify-center mt-8">
            <div className="dots"><span /><span /><span /></div>
          </div>
        )}
        {error && <div style={{ color: 'var(--rd)', fontSize: '12px' }}>! {error}</div>}
        {!loading && entries.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            {query ? 'No results found.' : 'Your knowledge base is empty.'}
          </div>
        )}
        {entries.map(entry => (
          <div
            key={entry.id}
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', padding: '10px',
            }}
          >
            <div className="flex items-start justify-between gap-2">
              <div style={{ flex: 1, minWidth: 0 }}>
                {entry.title && (
                  <div style={{ fontWeight: 500, fontSize: '12.5px', marginBottom: 4, color: 'var(--pl)' }}>
                    {entry.title}
                  </div>
                )}
                <div style={{
                  fontSize: '12px', color: 'var(--mu)', overflow: 'hidden',
                  display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' as const,
                }}>
                  {entry.text}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 6 }}>
                  {entry.source_type} - {entry.created_at ? new Date(entry.created_at).toLocaleDateString() : ''}
                </div>
              </div>
              <button
                onClick={() => deleteEntry(entry.id)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 4, flexShrink: 0 }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────

export default function WisebasePanel() {
  const [tab, setTab] = useState<Tab>('highlights');

  const tabStyle = (t: Tab): React.CSSProperties => ({
    flex: 1,
    padding: '8px 0',
    fontSize: '12px',
    fontWeight: 500,
    fontFamily: 'inherit',
    cursor: 'pointer',
    background: 'transparent',
    border: 'none',
    borderBottom: tab === t ? '2px solid var(--pl)' : '2px solid transparent',
    color: tab === t ? 'var(--tx)' : 'var(--mu)',
    transition: 'color 0.15s, border-color 0.15s',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
  });

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div className="flex" style={{ borderBottom: '1px solid var(--b1)', flexShrink: 0 }}>
        <button style={tabStyle('highlights')} onClick={() => setTab('highlights')}>
          <Highlighter size={13} />
          Highlights
        </button>
        <button style={tabStyle('knowledge')} onClick={() => setTab('knowledge')}>
          <BookOpen size={13} />
          Knowledge
        </button>
      </div>

      {/* Tab content */}
      {tab === 'highlights' ? <HighlightsTab /> : <KnowledgeTab />}
    </div>
  );
}
