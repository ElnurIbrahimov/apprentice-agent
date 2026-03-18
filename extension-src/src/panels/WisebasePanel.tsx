import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Search, Trash2, ExternalLink, Download, Highlighter, BookOpen,
  ChevronDown, ChevronRight, ChevronLeft, Layers, HelpCircle,
  Check, X, RotateCcw, Sparkles, Share2, Copy, ZoomIn, ZoomOut,
  RefreshCw, Code2,
} from 'lucide-react';
import { HTTP, apiFetch, getAuthHeaders } from '../api';
import { useStore } from '../store';
import { sendMsg } from '../ext';
import ext from '../ext';

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

interface FlashCard {
  front: string;
  back: string;
  known: boolean;
  lastReviewed: number;
  reviewCount: number;
}

interface FlashCardDeck {
  id: string;
  cards: FlashCard[];
  createdAt: number;
  sourceLabel: string;
  contentHash: string;
}

interface QuizQuestion {
  question: string;
  options: string[];
  correct: number;
}

interface QuizResult {
  id: string;
  questions: QuizQuestion[];
  answers: (number | null)[];
  score: number;
  total: number;
  createdAt: number;
  sourceLabel: string;
}

type Tab = 'knowledge' | 'highlights' | 'flashcards' | 'quiz' | 'graph';
type ContentSource = 'highlights' | 'knowledge' | 'custom';

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

// ── Shared AI Helpers ────────────────────────────────────────────────────────

/** Simple djb2 hash for content dedup */
function contentHash(s: string): string {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/** Extract JSON array from AI response — handles markdown fences, preamble */
function extractJsonArray(text: string): any[] | null {
  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) return parsed;
  } catch {}
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    try {
      const parsed = JSON.parse(fenceMatch[1].trim());
      if (Array.isArray(parsed)) return parsed;
    } catch {}
  }
  const bracketStart = text.indexOf('[');
  const bracketEnd = text.lastIndexOf(']');
  if (bracketStart !== -1 && bracketEnd > bracketStart) {
    try {
      const parsed = JSON.parse(text.slice(bracketStart, bracketEnd + 1));
      if (Array.isArray(parsed)) return parsed;
    } catch {}
  }
  return null;
}

/** Send a generation request to /api/chat */
async function aiGenerate(prompt: string, signal?: AbortSignal): Promise<string> {
  const resp = await fetch(`${HTTP}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ message: prompt }),
    signal,
  });
  if (!resp.ok) {
    const d = await resp.json().catch(() => ({}));
    throw new Error((d as any).detail || `HTTP ${resp.status}`);
  }
  const data = await resp.json();
  return data.response || data.text || data.content || data.reply || data.message || '';
}

// ── Content Source Picker ────────────────────────────────────────────────────

function ContentSourcePicker({
  source, setSource, customText, setCustomText,
}: {
  source: ContentSource; setSource: (s: ContentSource) => void;
  customText: string; setCustomText: (t: string) => void;
}) {
  const btnStyle = (s: ContentSource): React.CSSProperties => ({
    flex: 1, padding: '5px 4px', fontSize: '11px', fontFamily: 'inherit',
    cursor: 'pointer',
    background: source === s ? 'var(--pg2)' : 'var(--s2)',
    border: source === s ? '1px solid var(--pl)' : '1px solid var(--b1)',
    borderRadius: 'var(--r-sm)',
    color: source === s ? 'var(--pl)' : 'var(--mu)',
    transition: 'all 0.15s',
  });
  return (
    <div className="flex flex-col gap-2">
      <div className="flex gap-1.5">
        <button style={btnStyle('highlights')} onClick={() => setSource('highlights')}>Highlights</button>
        <button style={btnStyle('knowledge')} onClick={() => setSource('knowledge')}>Knowledge</button>
        <button style={btnStyle('custom')} onClick={() => setSource('custom')}>Custom</button>
      </div>
      {source === 'custom' && (
        <textarea
          value={customText} onChange={e => setCustomText(e.target.value)}
          placeholder="Paste or type content to study..."
          style={{
            width: '100%', minHeight: 80, maxHeight: 160, resize: 'vertical',
            background: 'var(--s2)', border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
            padding: '8px', outline: 'none', fontFamily: 'inherit',
          }}
        />
      )}
    </div>
  );
}

/** Gather content from the selected source */
async function gatherSourceContent(source: ContentSource, customText: string): Promise<string> {
  if (source === 'custom') {
    if (!customText.trim()) throw new Error('Please enter some text first.');
    return customText.trim();
  }
  if (source === 'highlights') {
    const resp = await sendMsg({ type: 'GET_ALL_HIGHLIGHTS' });
    if (!resp?.ok || !resp.store) throw new Error('No highlights found.');
    const allTexts: string[] = [];
    for (const url of Object.keys(resp.store)) {
      for (const hl of (resp.store[url] as HighlightData[])) allTexts.push(hl.text);
    }
    if (allTexts.length === 0) throw new Error('No highlights saved yet.');
    return allTexts.slice(0, 15).join('\n\n').slice(0, 3000);
  }
  // knowledge
  const data = await apiFetch(`${HTTP}/api/knowledge/list?limit=10`);
  const items = data.items || data.entries || data.results || [];
  if (items.length === 0) throw new Error('Knowledge base is empty.');
  return items.map((item: any) => {
    const title = item.title ? `${item.title}: ` : '';
    return title + (item.content || item.text || '');
  }).join('\n\n').slice(0, 3000);
}

// ── FlashCards Tab ───────────────────────────────────────────────────────────

function FlashCardsTab() {
  const [decks, setDecks] = useState<FlashCardDeck[]>([]);
  const [activeDeck, setActiveDeck] = useState<FlashCardDeck | null>(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [source, setSource] = useState<ContentSource>('highlights');
  const [customText, setCustomText] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    ext?.storage?.local?.get(['aura_flashcards'], (d: any) => {
      if (d?.aura_flashcards) setDecks(d.aura_flashcards);
    });
  }, []);

  const saveDecks = useCallback((updated: FlashCardDeck[]) => {
    setDecks(updated);
    ext?.storage?.local?.set({ aura_flashcards: updated });
  }, []);

  const generateFlashcards = useCallback(async () => {
    setLoading(true);
    setError('');
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const content = await gatherSourceContent(source, customText);
      const hash = contentHash(content);
      const existing = decks.find(d => d.contentHash === hash);
      if (existing) {
        setActiveDeck(existing);
        setCurrentIndex(0);
        setFlipped(false);
        setLoading(false);
        return;
      }
      const prompt = `Generate 10 flashcards (question and answer pairs) from the following content. The flashcards should test key concepts, definitions, and important details.\n\nReturn ONLY a JSON array, no other text: [{"front": "question or term", "back": "answer or definition"}]\n\nContent:\n${content}`;
      const responseText = await aiGenerate(prompt, ctrl.signal);
      const parsed = extractJsonArray(responseText);
      if (!parsed || parsed.length === 0) throw new Error('Could not parse flashcards from AI response. Try again.');
      const cards: FlashCard[] = parsed
        .filter((c: any) => c.front && c.back).slice(0, 15)
        .map((c: any) => ({ front: String(c.front), back: String(c.back), known: false, lastReviewed: 0, reviewCount: 0 }));
      if (cards.length === 0) throw new Error('AI returned empty flashcards. Try with different content.');
      const sourceLabel = source === 'custom' ? 'Custom text' : source === 'highlights' ? 'Highlights' : 'Knowledge';
      const deck: FlashCardDeck = { id: Date.now().toString(36), cards, createdAt: Date.now(), sourceLabel, contentHash: hash };
      const updated = [deck, ...decks].slice(0, 20);
      saveDecks(updated);
      setActiveDeck(deck);
      setCurrentIndex(0);
      setFlipped(false);
    } catch (err: any) {
      if (err.name !== 'AbortError') setError(err.message || 'Generation failed');
    } finally { setLoading(false); abortRef.current = null; }
  }, [source, customText, decks, saveDecks]);

  const markCard = useCallback((known: boolean) => {
    if (!activeDeck) return;
    const updated = { ...activeDeck, cards: [...activeDeck.cards] };
    updated.cards[currentIndex] = {
      ...updated.cards[currentIndex], known,
      lastReviewed: Date.now(),
      reviewCount: updated.cards[currentIndex].reviewCount + 1,
    };
    setActiveDeck(updated);
    saveDecks(decks.map(d => d.id === updated.id ? updated : d));
    if (currentIndex < updated.cards.length - 1) {
      setTimeout(() => { setCurrentIndex(i => i + 1); setFlipped(false); }, 300);
    }
  }, [activeDeck, currentIndex, decks, saveDecks]);

  const deleteDeck = useCallback((id: string) => {
    saveDecks(decks.filter(d => d.id !== id));
    if (activeDeck?.id === id) { setActiveDeck(null); setCurrentIndex(0); }
  }, [decks, activeDeck, saveDecks]);

  const startStudy = useCallback((deck: FlashCardDeck) => {
    // Sort: unknown first, less reviewed first, older review first
    const indices = deck.cards.map((_, i) => i);
    indices.sort((a, b) => {
      const ca = deck.cards[a], cb = deck.cards[b];
      if (ca.known !== cb.known) return ca.known ? 1 : -1;
      if (ca.reviewCount !== cb.reviewCount) return ca.reviewCount - cb.reviewCount;
      return ca.lastReviewed - cb.lastReviewed;
    });
    setActiveDeck({ ...deck, cards: indices.map(i => deck.cards[i]) });
    setCurrentIndex(0);
    setFlipped(false);
  }, []);

  // ── Active card review mode ──
  if (activeDeck) {
    const card = activeDeck.cards[currentIndex];
    const knownCount = activeDeck.cards.filter(c => c.known).length;
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
          <button onClick={() => { setActiveDeck(null); setCurrentIndex(0); }}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', display: 'flex', alignItems: 'center', gap: 4, fontSize: '12px', fontFamily: 'inherit' }}>
            <ChevronLeft size={14} /> Back
          </button>
          <div style={{ fontSize: '11px', color: 'var(--di)' }}>{knownCount}/{activeDeck.cards.length} mastered</div>
        </div>
        {/* Card counter */}
        <div style={{ textAlign: 'center', padding: '8px 12px 4px', fontSize: '11px', color: 'var(--mu)' }}>
          Card {currentIndex + 1} of {activeDeck.cards.length}
        </div>
        {/* Flip card */}
        <div className="flex-1 flex items-center justify-center p-4" style={{ minHeight: 0 }}>
          <div style={{ width: '100%', maxWidth: 320, height: 200, perspective: 800, cursor: 'pointer' }}
            onClick={() => setFlipped(f => !f)}>
            <div style={{
              width: '100%', height: '100%', position: 'relative',
              transformStyle: 'preserve-3d',
              transition: 'transform 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
              transform: flipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
            }}>
              {/* Front */}
              <div style={{
                position: 'absolute', inset: 0, backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                background: 'linear-gradient(135deg, var(--s2), var(--s3))',
                border: '1px solid var(--b1)', borderRadius: 'var(--r-lg)', padding: 20, boxShadow: 'var(--sh-sm)',
              }}>
                <div style={{ fontSize: '10px', color: 'var(--di)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Question</div>
                <div style={{ fontSize: '14px', color: 'var(--tx)', textAlign: 'center', lineHeight: 1.5, overflow: 'auto', maxHeight: '100%' }}>{card.front}</div>
                <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 12 }}>tap to flip</div>
              </div>
              {/* Back */}
              <div style={{
                position: 'absolute', inset: 0, backfaceVisibility: 'hidden', WebkitBackfaceVisibility: 'hidden',
                transform: 'rotateY(180deg)',
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.08), var(--s3))',
                border: '1px solid rgba(124, 58, 237, 0.25)', borderRadius: 'var(--r-lg)', padding: 20, boxShadow: 'var(--sh-sm)',
              }}>
                <div style={{ fontSize: '10px', color: 'var(--pl)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Answer</div>
                <div style={{ fontSize: '13px', color: 'var(--tx)', textAlign: 'center', lineHeight: 1.5, overflow: 'auto', maxHeight: '100%' }}>{card.back}</div>
              </div>
            </div>
          </div>
        </div>
        {/* Controls */}
        <div className="flex flex-col gap-2 p-3 flex-shrink-0" style={{ borderTop: '1px solid var(--b1)' }}>
          <div className="flex gap-2">
            <button onClick={() => markCard(false)} style={{
              flex: 1, padding: '8px 0', fontSize: '12px', fontFamily: 'inherit', cursor: 'pointer',
              borderRadius: 'var(--r-md)', background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)', color: 'var(--rd)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
            }}>
              <X size={13} /> Still learning
            </button>
            <button onClick={() => markCard(true)} style={{
              flex: 1, padding: '8px 0', fontSize: '12px', fontFamily: 'inherit', cursor: 'pointer',
              borderRadius: 'var(--r-md)', background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.3)', color: 'var(--gr)',
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
            }}>
              <Check size={13} /> Know it
            </button>
          </div>
          <div className="flex items-center justify-between">
            <button onClick={() => { setCurrentIndex(i => Math.max(0, i - 1)); setFlipped(false); }}
              disabled={currentIndex === 0}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                color: currentIndex === 0 ? 'var(--di)' : 'var(--mu)',
                padding: '5px 12px', cursor: currentIndex === 0 ? 'default' : 'pointer',
                fontSize: '11px', fontFamily: 'inherit',
              }}>Previous</button>
            <div style={{ display: 'flex', gap: 3 }}>
              {activeDeck.cards.map((c, i) => (
                <div key={i} style={{
                  width: 6, height: 6, borderRadius: '50%',
                  background: i === currentIndex ? 'var(--pl)' : c.known ? 'var(--gr)' : 'var(--b2)',
                  transition: 'background 0.2s',
                }} />
              ))}
            </div>
            <button onClick={() => { setCurrentIndex(i => Math.min(activeDeck.cards.length - 1, i + 1)); setFlipped(false); }}
              disabled={currentIndex === activeDeck.cards.length - 1}
              style={{
                background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
                color: currentIndex === activeDeck.cards.length - 1 ? 'var(--di)' : 'var(--mu)',
                padding: '5px 12px',
                cursor: currentIndex === activeDeck.cards.length - 1 ? 'default' : 'pointer',
                fontSize: '11px', fontFamily: 'inherit',
              }}>Next</button>
          </div>
        </div>
      </div>
    );
  }

  // ── Deck list / generation view ──
  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: 12 }}>
          <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)', marginBottom: 8 }}>Generate FlashCards</div>
          <ContentSourcePicker source={source} setSource={setSource} customText={customText} setCustomText={setCustomText} />
          <button onClick={generateFlashcards} disabled={loading} style={{
            width: '100%', marginTop: 10, padding: '8px 0', fontSize: '12px', fontWeight: 500, fontFamily: 'inherit',
            cursor: loading ? 'default' : 'pointer', borderRadius: 'var(--r-md)',
            background: loading ? 'var(--s3)' : 'var(--pg2)', border: '1px solid var(--pl)', color: 'var(--pl)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            opacity: loading ? 0.7 : 1, transition: 'opacity 0.15s',
          }}>
            {loading
              ? <><div className="dots" style={{ transform: 'scale(0.7)' }}><span /><span /><span /></div>Generating flashcards...</>
              : <><Sparkles size={13} /> Generate FlashCards</>}
          </button>
          {error && <div style={{ color: 'var(--rd)', fontSize: '11px', marginTop: 6 }}>{error}</div>}
        </div>

        {decks.length > 0 && (
          <div>
            <div style={{ fontSize: '11px', color: 'var(--di)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Saved Decks ({decks.length})
            </div>
            <div className="flex flex-col gap-1.5">
              {decks.map(deck => {
                const knownCount = deck.cards.filter(c => c.known).length;
                const pct = Math.round((knownCount / deck.cards.length) * 100);
                return (
                  <div key={deck.id} onClick={() => startStudy(deck)} style={{
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-md)', padding: '8px 10px', cursor: 'pointer', transition: 'border-color 0.15s',
                  }}>
                    <div className="flex items-center justify-between">
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: '12px', color: 'var(--tx)', fontWeight: 500 }}>
                          {deck.cards.length} cards <span style={{ color: 'var(--di)', fontWeight: 400 }}>from {deck.sourceLabel}</span>
                        </div>
                        <div className="flex items-center gap-2" style={{ marginTop: 4 }}>
                          <div style={{ flex: 1, height: 3, background: 'var(--b1)', borderRadius: 2, overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: pct === 100 ? 'var(--gr)' : 'var(--pl)', transition: 'width 0.3s' }} />
                          </div>
                          <span style={{ fontSize: '10px', color: 'var(--mu)', whiteSpace: 'nowrap' }}>{pct}%</span>
                        </div>
                        <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 3 }}>{relativeTime(deck.createdAt)}</div>
                      </div>
                      <button onClick={(e) => { e.stopPropagation(); deleteDeck(deck.id); }}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 4, flexShrink: 0 }}
                        title="Delete deck"><Trash2 size={12} /></button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {decks.length === 0 && !loading && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 16 }}>
            Generate flashcards from your highlights, knowledge, or custom text to start studying.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Quiz Tab ─────────────────────────────────────────────────────────────────

function QuizTab() {
  const [quizHistory, setQuizHistory] = useState<QuizResult[]>([]);
  const [activeQuiz, setActiveQuiz] = useState<{
    questions: QuizQuestion[]; currentIndex: number;
    answers: (number | null)[]; answered: boolean; sourceLabel: string;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [source, setSource] = useState<ContentSource>('highlights');
  const [customText, setCustomText] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    ext?.storage?.local?.get(['aura_quiz_history'], (d: any) => {
      if (d?.aura_quiz_history) setQuizHistory(d.aura_quiz_history);
    });
  }, []);

  const saveHistory = useCallback((updated: QuizResult[]) => {
    setQuizHistory(updated);
    ext?.storage?.local?.set({ aura_quiz_history: updated });
  }, []);

  const generateQuiz = useCallback(async () => {
    setLoading(true);
    setError('');
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const content = await gatherSourceContent(source, customText);
      const prompt = `Generate 10 multiple-choice questions from the following content. Each question should have exactly 4 options (A, B, C, D) with one correct answer.\n\nReturn ONLY a JSON array, no other text: [{"question": "...", "options": ["A. ...", "B. ...", "C. ...", "D. ..."], "correct": 0}]\nWhere "correct" is the zero-based index of the correct option (0=A, 1=B, 2=C, 3=D).\n\nContent:\n${content}`;
      const responseText = await aiGenerate(prompt, ctrl.signal);
      const parsed = extractJsonArray(responseText);
      if (!parsed || parsed.length === 0) throw new Error('Could not parse quiz from AI response. Try again.');
      const questions: QuizQuestion[] = parsed
        .filter((q: any) => q.question && Array.isArray(q.options) && q.options.length >= 4 && typeof q.correct === 'number')
        .slice(0, 15)
        .map((q: any) => ({ question: String(q.question), options: q.options.slice(0, 4).map(String), correct: Math.min(Math.max(0, q.correct), 3) }));
      if (questions.length === 0) throw new Error('AI returned invalid quiz format. Try again.');
      const sourceLabel = source === 'custom' ? 'Custom text' : source === 'highlights' ? 'Highlights' : 'Knowledge';
      setActiveQuiz({ questions, currentIndex: 0, answers: new Array(questions.length).fill(null), answered: false, sourceLabel });
    } catch (err: any) {
      if (err.name !== 'AbortError') setError(err.message || 'Generation failed');
    } finally { setLoading(false); abortRef.current = null; }
  }, [source, customText]);

  const selectAnswer = useCallback((optionIndex: number) => {
    if (!activeQuiz || activeQuiz.answered) return;
    const answers = [...activeQuiz.answers];
    answers[activeQuiz.currentIndex] = optionIndex;
    setActiveQuiz({ ...activeQuiz, answers, answered: true });
  }, [activeQuiz]);

  const nextQuestion = useCallback(() => {
    if (!activeQuiz) return;
    const nextIdx = activeQuiz.currentIndex + 1;
    if (nextIdx >= activeQuiz.questions.length) {
      const score = activeQuiz.answers.reduce((acc: number, a, i) => acc + (a === activeQuiz.questions[i].correct ? 1 : 0), 0);
      const result: QuizResult = {
        id: Date.now().toString(36), questions: activeQuiz.questions, answers: activeQuiz.answers,
        score, total: activeQuiz.questions.length, createdAt: Date.now(), sourceLabel: activeQuiz.sourceLabel,
      };
      saveHistory([result, ...quizHistory].slice(0, 20));
      setActiveQuiz({ ...activeQuiz, currentIndex: nextIdx });
      return;
    }
    setActiveQuiz({ ...activeQuiz, currentIndex: nextIdx, answered: false });
  }, [activeQuiz, quizHistory, saveHistory]);

  const deleteResult = useCallback((id: string) => {
    saveHistory(quizHistory.filter(r => r.id !== id));
  }, [quizHistory, saveHistory]);

  // ── Results screen ──
  if (activeQuiz && activeQuiz.currentIndex >= activeQuiz.questions.length) {
    const score = activeQuiz.answers.reduce((acc: number, a, i) => acc + (a === activeQuiz.questions[i].correct ? 1 : 0), 0);
    const pct = Math.round((score / activeQuiz.questions.length) * 100);
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center justify-between p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
          <button onClick={() => setActiveQuiz(null)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', display: 'flex', alignItems: 'center', gap: 4, fontSize: '12px', fontFamily: 'inherit' }}>
            <ChevronLeft size={14} /> Back
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 flex flex-col items-center gap-4">
          <div style={{
            width: 100, height: 100, borderRadius: '50%',
            display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            background: pct >= 70 ? 'rgba(16,185,129,0.1)' : pct >= 40 ? 'rgba(245,158,11,0.1)' : 'rgba(239,68,68,0.1)',
            border: `2px solid ${pct >= 70 ? 'var(--gr)' : pct >= 40 ? '#f59e0b' : 'var(--rd)'}`, marginTop: 8,
          }}>
            <div style={{ fontSize: '28px', fontWeight: 700, color: pct >= 70 ? 'var(--gr)' : pct >= 40 ? '#f59e0b' : 'var(--rd)' }}>{pct}%</div>
            <div style={{ fontSize: '10px', color: 'var(--mu)' }}>{score}/{activeQuiz.questions.length}</div>
          </div>
          <div style={{ fontSize: '14px', fontWeight: 500, color: 'var(--tx)' }}>
            {pct >= 90 ? 'Excellent!' : pct >= 70 ? 'Good job!' : pct >= 40 ? 'Keep studying!' : 'Needs more review'}
          </div>
          <div className="flex flex-col gap-2 w-full">
            {activeQuiz.questions.map((q, i) => {
              const userAnswer = activeQuiz.answers[i];
              const isCorrect = userAnswer === q.correct;
              return (
                <div key={i} style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: '8px 10px' }}>
                  <div className="flex items-start gap-2">
                    <div style={{
                      width: 18, height: 18, borderRadius: '50%', flexShrink: 0, marginTop: 1,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: isCorrect ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)',
                    }}>
                      {isCorrect ? <Check size={10} style={{ color: 'var(--gr)' }} /> : <X size={10} style={{ color: 'var(--rd)' }} />}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: '11px', color: 'var(--tx)', marginBottom: 2 }}>{q.question}</div>
                      {!isCorrect && userAnswer !== null && (
                        <div style={{ fontSize: '10px', color: 'var(--rd)' }}>Your answer: {q.options[userAnswer]}</div>
                      )}
                      <div style={{ fontSize: '10px', color: 'var(--gr)' }}>Correct: {q.options[q.correct]}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          <button onClick={() => {
            setActiveQuiz({
              questions: activeQuiz.questions, currentIndex: 0,
              answers: new Array(activeQuiz.questions.length).fill(null),
              answered: false, sourceLabel: activeQuiz.sourceLabel,
            });
          }} style={{
            padding: '8px 20px', fontSize: '12px', fontFamily: 'inherit', cursor: 'pointer',
            borderRadius: 'var(--r-md)', background: 'var(--pg2)', border: '1px solid var(--pl)',
            color: 'var(--pl)', display: 'flex', alignItems: 'center', gap: 5,
          }}>
            <RotateCcw size={12} /> Retry Quiz
          </button>
        </div>
      </div>
    );
  }

  // ── Active quiz question ──
  if (activeQuiz) {
    const q = activeQuiz.questions[activeQuiz.currentIndex];
    const userAnswer = activeQuiz.answers[activeQuiz.currentIndex];
    const answeredCount = activeQuiz.answers.filter(a => a !== null).length;
    const correctSoFar = activeQuiz.answers.reduce((acc: number, a, i) => {
      if (a === null) return acc;
      return acc + (a === activeQuiz.questions[i].correct ? 1 : 0);
    }, 0);
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center justify-between p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
          <button onClick={() => setActiveQuiz(null)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', display: 'flex', alignItems: 'center', gap: 4, fontSize: '12px', fontFamily: 'inherit' }}>
            <ChevronLeft size={14} /> Quit
          </button>
          <div style={{ fontSize: '11px', color: 'var(--mu)' }}>{correctSoFar}/{answeredCount} correct</div>
        </div>
        {/* Progress bar */}
        <div style={{ height: 3, background: 'var(--b1)' }}>
          <div style={{ width: `${((activeQuiz.currentIndex + 1) / activeQuiz.questions.length) * 100}%`, height: '100%', background: 'var(--pl)', transition: 'width 0.3s' }} />
        </div>
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
          <div style={{ fontSize: '10px', color: 'var(--di)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Question {activeQuiz.currentIndex + 1} of {activeQuiz.questions.length}
          </div>
          <div style={{ fontSize: '14px', color: 'var(--tx)', fontWeight: 500, lineHeight: 1.5, marginBottom: 4 }}>{q.question}</div>
          <div className="flex flex-col gap-2">
            {q.options.map((opt, oi) => {
              const isSelected = userAnswer === oi;
              const isCorrect = oi === q.correct;
              const isAnswered = activeQuiz.answered;
              let bg = 'var(--s2)', borderColor = 'var(--b1)', textColor = 'var(--tx)';
              if (isAnswered) {
                if (isCorrect) { bg = 'rgba(16,185,129,0.1)'; borderColor = 'rgba(16,185,129,0.5)'; textColor = 'var(--gr)'; }
                else if (isSelected) { bg = 'rgba(239,68,68,0.1)'; borderColor = 'rgba(239,68,68,0.5)'; textColor = 'var(--rd)'; }
              } else if (isSelected) { borderColor = 'var(--pl)'; bg = 'var(--pg)'; }
              return (
                <button key={oi} onClick={() => selectAnswer(oi)} disabled={isAnswered} style={{
                  width: '100%', textAlign: 'left', padding: '10px 12px', fontSize: '12px', fontFamily: 'inherit',
                  cursor: isAnswered ? 'default' : 'pointer', borderRadius: 'var(--r-md)',
                  background: bg, border: `1px solid ${borderColor}`, color: textColor,
                  transition: 'all 0.15s', display: 'flex', alignItems: 'center', gap: 8,
                }}>
                  <div style={{
                    width: 22, height: 22, borderRadius: '50%', flexShrink: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: isAnswered && isCorrect ? 'rgba(16,185,129,0.2)' : isAnswered && isSelected && !isCorrect ? 'rgba(239,68,68,0.2)' : 'var(--b1)',
                    fontSize: '10px', fontWeight: 600,
                    color: isAnswered && isCorrect ? 'var(--gr)' : isAnswered && isSelected && !isCorrect ? 'var(--rd)' : 'var(--mu)',
                  }}>{String.fromCharCode(65 + oi)}</div>
                  <span style={{ lineHeight: 1.4 }}>{opt}</span>
                  {isAnswered && isCorrect && <Check size={14} style={{ marginLeft: 'auto', color: 'var(--gr)' }} />}
                  {isAnswered && isSelected && !isCorrect && <X size={14} style={{ marginLeft: 'auto', color: 'var(--rd)' }} />}
                </button>
              );
            })}
          </div>
        </div>
        {activeQuiz.answered && (
          <div className="p-3 flex-shrink-0" style={{ borderTop: '1px solid var(--b1)' }}>
            <button onClick={nextQuestion} style={{
              width: '100%', padding: '8px 0', fontSize: '12px', fontWeight: 500, fontFamily: 'inherit',
              cursor: 'pointer', borderRadius: 'var(--r-md)', background: 'var(--pg2)',
              border: '1px solid var(--pl)', color: 'var(--pl)',
            }}>
              {activeQuiz.currentIndex + 1 >= activeQuiz.questions.length ? 'See Results' : 'Next Question'}
            </button>
          </div>
        )}
      </div>
    );
  }

  // ── Quiz generation / history view ──
  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: 12 }}>
          <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)', marginBottom: 8 }}>Generate Quiz</div>
          <ContentSourcePicker source={source} setSource={setSource} customText={customText} setCustomText={setCustomText} />
          <button onClick={generateQuiz} disabled={loading} style={{
            width: '100%', marginTop: 10, padding: '8px 0', fontSize: '12px', fontWeight: 500, fontFamily: 'inherit',
            cursor: loading ? 'default' : 'pointer', borderRadius: 'var(--r-md)',
            background: loading ? 'var(--s3)' : 'var(--pg2)', border: '1px solid var(--pl)', color: 'var(--pl)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
            opacity: loading ? 0.7 : 1, transition: 'opacity 0.15s',
          }}>
            {loading
              ? <><div className="dots" style={{ transform: 'scale(0.7)' }}><span /><span /><span /></div>Generating quiz...</>
              : <><HelpCircle size={13} /> Generate Quiz</>}
          </button>
          {error && <div style={{ color: 'var(--rd)', fontSize: '11px', marginTop: 6 }}>{error}</div>}
        </div>

        {quizHistory.length > 0 && (
          <div>
            <div style={{ fontSize: '11px', color: 'var(--di)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Past Quizzes ({quizHistory.length})
            </div>
            <div className="flex flex-col gap-1.5">
              {quizHistory.map(result => {
                const pct = Math.round((result.score / result.total) * 100);
                return (
                  <div key={result.id} style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)', padding: '8px 10px' }}>
                    <div className="flex items-center justify-between">
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="flex items-center gap-2">
                          <div style={{ fontSize: '14px', fontWeight: 600, color: pct >= 70 ? 'var(--gr)' : pct >= 40 ? '#f59e0b' : 'var(--rd)' }}>{pct}%</div>
                          <div style={{ fontSize: '12px', color: 'var(--tx)' }}>{result.score}/{result.total} correct</div>
                        </div>
                        <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 2 }}>{result.sourceLabel} - {relativeTime(result.createdAt)}</div>
                      </div>
                      <div className="flex items-center gap-1" style={{ flexShrink: 0 }}>
                        <button onClick={() => {
                          setActiveQuiz({
                            questions: result.questions, currentIndex: 0,
                            answers: new Array(result.questions.length).fill(null),
                            answered: false, sourceLabel: result.sourceLabel,
                          });
                        }} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
                          title="Retry quiz"><RotateCcw size={12} /></button>
                        <button onClick={() => deleteResult(result.id)}
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--di)', padding: 4 }}
                          title="Delete quiz"><Trash2 size={12} /></button>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {quizHistory.length === 0 && !loading && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 16 }}>
            Generate a quiz from your highlights, knowledge, or custom text to test yourself.
          </div>
        )}
      </div>
    </div>
  );
}

// ── Graph Tab ─────────────────────────────────────────────────────────────────

const GRAPH_SYSTEM_PROMPT = `You are a knowledge graph extraction expert. Analyze the provided content and extract the most important concepts and their relationships as a Mermaid graph definition.

Rules:
- Use \`graph TD\` (top-down) syntax
- Maximum 20 nodes, minimum 3 nodes
- Use short, clear labels (2-4 words max per node)
- Label edges with the relationship type
- Use meaningful node IDs (A, B, C, etc.)
- Group related concepts visually
- Return ONLY the Mermaid definition, no markdown fences, no explanation
- Use subgraph blocks to group related clusters when there are 8+ nodes

Example output:
graph TD
  A[Machine Learning] -->|uses| B[Neural Networks]
  A -->|requires| C[Training Data]
  B -->|type of| D[Deep Learning]
  C -->|processed by| E[Feature Engineering]`;

function escHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function stripFences(s: string): string {
  return s.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
}

function buildMermaidSrcdoc(mermaidCode: string, zoom: number): string {
  const escaped = escHtml(mermaidCode);
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"><\/script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e8e6f0;min-height:100vh;overflow:auto}
.mermaid{display:flex;justify-content:center;padding:24px;transform:scale(${zoom});transform-origin:top center;transition:transform 0.2s ease}
.mermaid svg{max-width:100%}
.node rect,.node circle,.node polygon{cursor:pointer !important}
.node:hover rect,.node:hover circle,.node:hover polygon{filter:brightness(1.3) !important}
</style>
</head>
<body>
<pre class="mermaid">
${escaped}
</pre>
<script>
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  themeVariables: {
    primaryColor: '#7c3aed',
    primaryTextColor: '#e8e6f0',
    primaryBorderColor: '#5b21b6',
    lineColor: '#6d28d9',
    secondaryColor: '#1e1b4b',
    tertiaryColor: '#0f0a2e',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    fontSize: '13px',
    nodeBorder: '#5b21b6',
    mainBkg: '#1e1b4b',
    clusterBkg: 'rgba(124,58,237,0.08)',
    clusterBorder: 'rgba(124,58,237,0.25)',
    edgeLabelBackground: '#0f0a2e',
    nodeTextColor: '#e8e6f0',
  },
  flowchart: { htmlLabels: true, curve: 'basis', padding: 12 },
  securityLevel: 'loose',
});

// Post node click events to parent
document.addEventListener('click', function(e) {
  var node = e.target.closest('.node');
  if (node) {
    var label = node.querySelector('.nodeLabel');
    if (label) {
      parent.postMessage({ type: 'graph-node-click', label: label.textContent.trim() }, '*');
    }
  }
});

// Notify parent when render completes
mermaid.run().then(function() {
  parent.postMessage({ type: 'graph-rendered' }, '*');
}).catch(function(err) {
  parent.postMessage({ type: 'graph-error', msg: err.message || String(err) }, '*');
});
<\/script>
</body>
</html>`;
}

function GraphTab() {
  const { getModel } = useStore();
  const [mermaidCode, setMermaidCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [zoom, setZoom] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const [nodeCount, setNodeCount] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Listen for messages from the iframe
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === 'graph-node-click') {
        // When a node is clicked, copy the label as a search term
        setStatus(`Node: ${e.data.label}`);
        setTimeout(() => setStatus(''), 2000);
      }
      if (e.data?.type === 'graph-rendered') {
        setStatus('Graph rendered');
        setTimeout(() => setStatus(''), 1500);
      }
      if (e.data?.type === 'graph-error') {
        setError(`Render error: ${e.data.msg}`);
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, []);

  // Update iframe when mermaidCode or zoom changes
  useEffect(() => {
    if (iframeRef.current && mermaidCode) {
      iframeRef.current.srcdoc = buildMermaidSrcdoc(mermaidCode, zoom);
    }
  }, [mermaidCode, zoom]);

  // Count nodes from mermaid code
  useEffect(() => {
    if (!mermaidCode) { setNodeCount(0); return; }
    // Count unique node definitions like A[Label] or A(Label) or A{Label}
    const nodeMatches = mermaidCode.match(/\b[A-Za-z_]\w*\s*[\[\(\{]/g);
    const unique = new Set(nodeMatches?.map(m => m.replace(/[\[\(\{\s]/g, '')) || []);
    setNodeCount(unique.size);
  }, [mermaidCode]);

  const gatherContent = async (): Promise<string> => {
    const parts: string[] = [];

    // Get highlights
    try {
      const resp = await sendMsg({ type: 'GET_ALL_HIGHLIGHTS' });
      if (resp?.ok && resp.store) {
        const allHighlights: string[] = [];
        for (const url of Object.keys(resp.store)) {
          const hls = resp.store[url] as HighlightData[];
          const pageTitle = hls[0]?.pageTitle || url;
          for (const hl of hls.slice(0, 10)) { // max 10 per page
            allHighlights.push(`[${pageTitle}] ${hl.text}`);
          }
        }
        if (allHighlights.length > 0) {
          parts.push('## Highlights\n' + allHighlights.slice(0, 30).join('\n'));
        }
      }
    } catch { /* ignore */ }

    // Get knowledge entries
    try {
      const url = `${HTTP}/api/knowledge/list?limit=30`;
      const data = await apiFetch(url);
      const rawItems = data.items || data.entries || data.results || [];
      const knowledgeParts: string[] = [];
      for (const item of rawItems) {
        const title = item.title || '';
        const text = item.content || item.text || '';
        if (title || text) {
          knowledgeParts.push(`${title}: ${truncate(text, 200)}`);
        }
      }
      if (knowledgeParts.length > 0) {
        parts.push('## Knowledge Base\n' + knowledgeParts.join('\n'));
      }
    } catch { /* ignore */ }

    return parts.join('\n\n');
  };

  const generateGraph = useCallback(async (extraContext?: string) => {
    setLoading(true);
    setError('');
    setStatus('Gathering content...');
    setMermaidCode('');

    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      const content = await gatherContent();
      if (!content && !extraContext) {
        setError('No content found. Add some highlights or knowledge entries first.');
        setLoading(false);
        setStatus('');
        return;
      }

      const fullContent = [content, extraContext].filter(Boolean).join('\n\n## Additional Context\n');
      setStatus('AI is extracting concepts...');

      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          message: `${GRAPH_SYSTEM_PROMPT}\n\nAnalyze this content and extract a knowledge graph:\n\n${fullContent}`,
          model: getModel('artifacts') || undefined,
        }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        throw new Error((d as any).detail || resp.statusText);
      }

      const data = await resp.json();
      const responseText = data.response || data.text || data.content || data.reply || data.message || '';
      const cleaned = stripFences(responseText).trim();

      if (!cleaned || !cleaned.includes('graph')) {
        throw new Error('AI did not return a valid Mermaid graph. Try again.');
      }

      setMermaidCode(cleaned);
      setZoom(1);
      setStatus('');
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Failed to generate graph');
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }, [getModel]);

  const copyMermaid = () => {
    if (!mermaidCode) return;
    navigator.clipboard.writeText(mermaidCode).then(() => {
      setStatus('Mermaid code copied!');
      setTimeout(() => setStatus(''), 1500);
    });
  };

  const downloadSvg = () => {
    if (!iframeRef.current) return;
    try {
      const svgEl = iframeRef.current.contentDocument?.querySelector('.mermaid svg');
      if (!svgEl) { setError('No SVG found — wait for graph to render'); return; }
      const svgData = new XMLSerializer().serializeToString(svgEl);
      const blob = new Blob([svgData], { type: 'image/svg+xml' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `aura-knowledge-graph-${new Date().toISOString().slice(0, 10)}.svg`;
      a.click();
      URL.revokeObjectURL(a.href);
      setStatus('SVG downloaded');
      setTimeout(() => setStatus(''), 1500);
    } catch {
      setError('Could not export SVG');
    }
  };

  const btnStyle: React.CSSProperties = {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
    background: 'var(--s2)', border: '1px solid var(--b1)',
    borderRadius: 'var(--r-md)', color: 'var(--mu)',
    padding: '5px 8px', cursor: 'pointer', fontSize: '11px', fontFamily: 'inherit',
    transition: 'all 0.15s ease',
  };

  // Empty state — no graph generated yet
  if (!mermaidCode && !loading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6" style={{ gap: 16 }}>
        <div style={{
          width: 56, height: 56, borderRadius: '50%',
          background: 'rgba(124,58,237,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Share2 size={24} style={{ color: 'var(--pl)' }} />
        </div>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--tx)', marginBottom: 6 }}>
            Knowledge Graph
          </div>
          <div style={{ fontSize: '11.5px', color: 'var(--mu)', lineHeight: 1.5, maxWidth: 240 }}>
            AI extracts concepts from your highlights and knowledge entries, then visualizes them as an interactive graph.
          </div>
        </div>
        <button
          onClick={() => generateGraph()}
          disabled={loading}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'var(--pl)', color: '#fff', border: 'none',
            borderRadius: 'var(--r-md)', padding: '8px 16px',
            fontSize: '12px', fontWeight: 500, cursor: 'pointer',
            fontFamily: 'inherit', transition: 'opacity 0.15s',
            opacity: loading ? 0.6 : 1,
          }}
        >
          <Sparkles size={14} />
          Generate Knowledge Graph
        </button>
        {error && (
          <div style={{ color: 'var(--rd)', fontSize: '11px', textAlign: 'center', maxWidth: 280 }}>
            {error}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 p-2 flex-shrink-0 flex-wrap" style={{ borderBottom: '1px solid var(--b1)' }}>
        <button onClick={() => generateGraph()} style={btnStyle} title="Regenerate graph" disabled={loading}>
          <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          {loading ? 'Generating...' : 'Regenerate'}
        </button>
        <button onClick={copyMermaid} style={btnStyle} title="Copy Mermaid code">
          <Copy size={12} /> Copy
        </button>
        <button onClick={downloadSvg} style={btnStyle} title="Download as SVG">
          <Download size={12} /> SVG
        </button>
        <button onClick={() => setShowCode(!showCode)} style={{
          ...btnStyle,
          background: showCode ? 'rgba(124,58,237,0.1)' : 'var(--s2)',
          borderColor: showCode ? 'rgba(124,58,237,0.3)' : 'var(--b1)',
          color: showCode ? 'var(--pl)' : 'var(--mu)',
        }} title="Toggle Mermaid source">
          <Code2 size={12} /> Code
        </button>
        <div style={{ flex: 1 }} />
        <button onClick={() => setZoom(z => Math.max(0.3, z - 0.15))} style={btnStyle} title="Zoom out">
          <ZoomOut size={12} />
        </button>
        <span style={{ fontSize: '10px', color: 'var(--di)', minWidth: 32, textAlign: 'center' }}>
          {Math.round(zoom * 100)}%
        </span>
        <button onClick={() => setZoom(z => Math.min(2.5, z + 0.15))} style={btnStyle} title="Zoom in">
          <ZoomIn size={12} />
        </button>
      </div>

      {/* Status / error bar */}
      {(status || error) && (
        <div style={{
          padding: '4px 12px', fontSize: '10px',
          borderBottom: '1px solid var(--b1)',
          background: error ? 'rgba(239,68,68,0.06)' : 'rgba(124,58,237,0.04)',
          color: error ? 'var(--rd)' : 'var(--pl)',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span>{error || status}</span>
          {nodeCount > 0 && !error && (
            <span style={{ color: 'var(--di)' }}>{nodeCount} nodes</span>
          )}
        </div>
      )}
      {!status && !error && nodeCount > 0 && (
        <div style={{
          padding: '4px 12px', fontSize: '10px',
          borderBottom: '1px solid var(--b1)',
          color: 'var(--di)', textAlign: 'right',
        }}>
          {nodeCount} nodes
        </div>
      )}

      {/* Loading overlay */}
      {loading && (
        <div className="flex flex-col items-center justify-center p-8" style={{ gap: 12 }}>
          <div className="dots"><span /><span /><span /></div>
          <div style={{ fontSize: '11px', color: 'var(--mu)' }}>
            Extracting concepts and relationships...
          </div>
        </div>
      )}

      {/* Code view */}
      {showCode && mermaidCode && (
        <div style={{
          maxHeight: 160, overflow: 'auto', flexShrink: 0,
          borderBottom: '1px solid var(--b1)',
        }}>
          <pre style={{
            margin: 0, padding: '10px 12px',
            fontSize: '11px', lineHeight: 1.5,
            color: 'var(--mu)', background: 'var(--s2)',
            fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          }}>
            {mermaidCode}
          </pre>
        </div>
      )}

      {/* Graph iframe */}
      {mermaidCode && !loading && (
        <div style={{ flex: 1, minHeight: 0, position: 'relative' }}>
          <iframe
            ref={iframeRef}
            sandbox="allow-scripts"
            style={{
              width: '100%', height: '100%',
              border: 'none', background: '#0a0a0f',
            }}
            title="Knowledge Graph"
          />
        </div>
      )}
    </div>
  );
}

// ── Main Panel ───────────────────────────────────────────────────────────────

export default function WisebasePanel() {
  const [tab, setTab] = useState<Tab>('highlights');

  const tabStyle = (t: Tab): React.CSSProperties => ({
    flex: 1,
    padding: '7px 0',
    fontSize: '10.5px',
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
    gap: 3,
  });

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tab bar */}
      <div className="flex" style={{ borderBottom: '1px solid var(--b1)', flexShrink: 0 }}>
        <button style={tabStyle('highlights')} onClick={() => setTab('highlights')}>
          <Highlighter size={11} />
          Highlights
        </button>
        <button style={tabStyle('knowledge')} onClick={() => setTab('knowledge')}>
          <BookOpen size={11} />
          Knowledge
        </button>
        <button style={tabStyle('flashcards')} onClick={() => setTab('flashcards')}>
          <Layers size={11} />
          Cards
        </button>
        <button style={tabStyle('quiz')} onClick={() => setTab('quiz')}>
          <HelpCircle size={11} />
          Quiz
        </button>
        <button style={tabStyle('graph')} onClick={() => setTab('graph')}>
          <Share2 size={11} />
          Graph
        </button>
      </div>

      {/* Tab content */}
      {tab === 'highlights' && <HighlightsTab />}
      {tab === 'knowledge' && <KnowledgeTab />}
      {tab === 'flashcards' && <FlashCardsTab />}
      {tab === 'quiz' && <QuizTab />}
      {tab === 'graph' && <GraphTab />}
    </div>
  );
}
