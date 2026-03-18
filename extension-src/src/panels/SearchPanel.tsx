import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Search, Brain, Globe, Layers, Sparkles, Check, X, ExternalLink, Clock, Trash2 } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
import { md } from '../markdown';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface Source {
  url: string;
  title: string;
  snippet?: string;
  favicon?: string;
  domain?: string;
}

interface SearchResult {
  query: string;
  answer: string;
  sources: Source[];
  relatedSearches?: string[];
  timestamp: number;
}

type StepStatus = 'pending' | 'active' | 'done';

interface PipelineStep {
  id: string;
  label: string;
  detail?: string;
  status: StepStatus;
  icon: 'brain' | 'globe' | 'layers' | 'sparkles';
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const RECENT_SEARCHES_KEY = 'aura_recent_searches';
const MAX_RECENT = 5;

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function getDomain(url: string): string {
  try { return new URL(url).hostname.replace(/^www\./, ''); } catch { return url; }
}

function getFavicon(url: string): string {
  try {
    const u = new URL(url);
    return `https://www.google.com/s2/favicons?sz=16&domain=${u.hostname}`;
  } catch { return ''; }
}

function loadRecentSearches(): string[] {
  try {
    const raw = localStorage.getItem(RECENT_SEARCHES_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveRecentSearches(searches: string[]) {
  try { localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(searches.slice(0, MAX_RECENT))); } catch {}
}

function addRecentSearch(query: string, prev: string[]): string[] {
  const filtered = prev.filter(s => s.toLowerCase() !== query.toLowerCase());
  return [query, ...filtered].slice(0, MAX_RECENT);
}

/* ------------------------------------------------------------------ */
/*  Step Icon                                                          */
/* ------------------------------------------------------------------ */

function StepIcon({ type, size = 14 }: { type: PipelineStep['icon']; size?: number }) {
  const props = { size, strokeWidth: 1.8 };
  switch (type) {
    case 'brain': return <Brain {...props} />;
    case 'globe': return <Globe {...props} />;
    case 'layers': return <Layers {...props} />;
    case 'sparkles': return <Sparkles {...props} />;
  }
}

/* ------------------------------------------------------------------ */
/*  Pipeline Steps UI                                                  */
/* ------------------------------------------------------------------ */

function PipelineView({ steps }: { steps: PipelineStep[] }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 0, padding: '16px 0' }}>
      {steps.map((step, i) => (
        <div key={step.id} style={{ display: 'flex', gap: 12, alignItems: 'flex-start', position: 'relative' }}>
          {/* Vertical connector line */}
          {i < steps.length - 1 && (
            <div style={{
              position: 'absolute',
              left: 13,
              top: 28,
              width: 1,
              height: 'calc(100% - 4px)',
              background: step.status === 'done' ? 'var(--gr)' : 'var(--b1)',
              transition: 'background 0.4s ease',
            }} />
          )}
          {/* Circle indicator */}
          <div style={{
            width: 27,
            height: 27,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            background: step.status === 'done'
              ? 'rgba(16, 185, 129, 0.15)'
              : step.status === 'active'
                ? 'var(--pg2)'
                : 'var(--s2)',
            border: `1.5px solid ${
              step.status === 'done'
                ? 'var(--gr)'
                : step.status === 'active'
                  ? 'var(--p)'
                  : 'var(--b1)'
            }`,
            color: step.status === 'done'
              ? 'var(--gr)'
              : step.status === 'active'
                ? 'var(--pl)'
                : 'var(--mu)',
            transition: 'all 0.3s ease',
            animation: step.status === 'active' ? 'searchStepPulse 2s ease-in-out infinite' : 'none',
          }}>
            {step.status === 'done' ? (
              <Check size={13} strokeWidth={2.5} />
            ) : step.status === 'active' ? (
              <div style={{ animation: 'searchStepSpin 1.2s linear infinite', display: 'flex' }}>
                <StepIcon type={step.icon} size={13} />
              </div>
            ) : (
              <StepIcon type={step.icon} size={13} />
            )}
          </div>
          {/* Label + detail */}
          <div style={{ paddingTop: 3, paddingBottom: 16, minHeight: 27 }}>
            <div style={{
              fontSize: '12.5px',
              fontWeight: step.status === 'active' ? 600 : 500,
              color: step.status === 'pending' ? 'var(--mu)' : 'var(--tx)',
              transition: 'color 0.3s ease',
            }}>
              {step.label}
            </div>
            {step.detail && step.status !== 'pending' && (
              <div style={{
                fontSize: '11px',
                color: 'var(--mu)',
                marginTop: 2,
                animation: 'fadeIn 0.3s ease',
              }}>
                {step.detail}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Source Card                                                        */
/* ------------------------------------------------------------------ */

function SourceCard({ source, index }: { source: Source; index: number }) {
  const domain = source.domain || getDomain(source.url);
  const favicon = source.favicon || getFavicon(source.url);

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="search-source-card"
      style={{
        display: 'flex',
        gap: 10,
        padding: '10px 12px',
        background: 'var(--s2)',
        border: '1px solid var(--b1)',
        borderRadius: 'var(--r-md)',
        textDecoration: 'none',
        transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        cursor: 'pointer',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLElement).style.transform = 'translateY(-1px)';
        (e.currentTarget as HTMLElement).style.boxShadow = '0 4px 16px rgba(0,0,0,0.2)';
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--b2)';
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
        (e.currentTarget as HTMLElement).style.boxShadow = 'none';
        (e.currentTarget as HTMLElement).style.borderColor = 'var(--b1)';
      }}
    >
      {/* Number badge */}
      <div style={{
        width: 22,
        height: 22,
        borderRadius: '50%',
        background: 'var(--pg)',
        border: '1px solid rgba(124, 58, 237, 0.15)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '10px',
        fontWeight: 700,
        color: 'var(--pl)',
        flexShrink: 0,
        marginTop: 1,
      }}>
        {index + 1}
      </div>
      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
          {favicon && <img src={favicon} width={14} height={14} alt="" style={{ borderRadius: 2, flexShrink: 0 }} />}
          <span style={{
            fontSize: '12px',
            fontWeight: 600,
            color: 'var(--pl)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {source.title || domain}
          </span>
        </div>
        {source.snippet && (
          <div style={{
            fontSize: '11px',
            color: 'var(--mu)',
            lineHeight: 1.5,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
          }}>
            {source.snippet}
          </div>
        )}
        <div style={{
          fontSize: '10px',
          color: 'var(--di)',
          marginTop: 3,
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}>
          {domain}
          <ExternalLink size={9} />
        </div>
      </div>
    </a>
  );
}

/* ------------------------------------------------------------------ */
/*  Result Thread Entry                                                */
/* ------------------------------------------------------------------ */

function ResultEntry({ result, isLatest }: { result: SearchResult; isLatest: boolean }) {
  const sourcesRef = useRef<HTMLDivElement>(null);

  // Make [1], [2] etc. clickable to scroll to sources
  const renderAnswer = useCallback((html: string) => {
    return html.replace(
      /\[(\d+)\]/g,
      '<a class="search-cite" data-cite="$1" style="color:var(--pl);cursor:pointer;font-weight:600;font-size:11px;vertical-align:super;text-decoration:none;">[$1]</a>'
    );
  }, []);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const el = (e.target as HTMLElement).closest('.search-cite');
      if (!el || !sourcesRef.current) return;
      sourcesRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, []);

  return (
    <div style={{ animation: isLatest ? 'fadeIn 0.3s ease' : 'none' }}>
      {/* Query */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '10px 0 8px',
      }}>
        <Search size={13} style={{ color: 'var(--pl)', flexShrink: 0 }} />
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>
          {result.query}
        </span>
      </div>

      {/* Answer */}
      {result.answer && (
        <div style={{
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-lg)',
          padding: '14px 16px',
          marginBottom: 12,
        }}>
          <div
            className="md-body"
            style={{ fontSize: '12.5px', lineHeight: 1.7 }}
            dangerouslySetInnerHTML={{ __html: renderAnswer(md(result.answer)) }}
          />
        </div>
      )}

      {/* Sources */}
      {result.sources.length > 0 && (
        <div ref={sourcesRef} style={{ marginBottom: 12 }}>
          <div style={{
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--mu)',
            marginBottom: 8,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}>
            <Globe size={11} />
            Sources ({result.sources.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {result.sources.map((src, i) => (
              <SourceCard key={i} source={src} index={i} />
            ))}
          </div>
        </div>
      )}

      {/* Related searches */}
      {result.relatedSearches && result.relatedSearches.length > 0 && isLatest && (
        <div style={{ marginBottom: 16 }}>
          <div style={{
            fontSize: '10px',
            fontWeight: 600,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            color: 'var(--mu)',
            marginBottom: 8,
          }}>
            Related searches
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {result.relatedSearches.map((q, i) => (
              <button
                key={i}
                className="search-related-pill"
                style={{
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-pill)',
                  padding: '5px 12px',
                  fontSize: '11px',
                  color: 'var(--tx)',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  transition: 'all 0.2s ease',
                }}
                onMouseEnter={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'rgba(124, 58, 237, 0.3)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--pg)';
                  (e.currentTarget as HTMLElement).style.color = 'var(--pl)';
                }}
                onMouseLeave={e => {
                  (e.currentTarget as HTMLElement).style.borderColor = 'var(--b1)';
                  (e.currentTarget as HTMLElement).style.background = 'var(--s2)';
                  (e.currentTarget as HTMLElement).style.color = 'var(--tx)';
                }}
              >
                <Search size={10} style={{ marginRight: 4, opacity: 0.6 }} />
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Panel                                                         */
/* ------------------------------------------------------------------ */

export default function SearchPanel() {
  const { getModel, ws } = useStore();
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');
  const [recentSearches, setRecentSearches] = useState<string[]>(loadRecentSearches);
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [liveSources, setLiveSources] = useState<Source[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cleanup abort controller on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  // Auto-scroll to bottom when new content arrives
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [results, streamingAnswer, pipelineSteps]);

  /* ------ Pipeline simulation ------ */

  const initPipeline = (): PipelineStep[] => [
    { id: 'understand', label: 'Understanding your query...', icon: 'brain', status: 'active', detail: '' },
    { id: 'search', label: 'Searching the web...', icon: 'globe', status: 'pending', detail: '' },
    { id: 'analyze', label: 'Analyzing results...', icon: 'layers', status: 'pending', detail: '' },
    { id: 'generate', label: 'Generating answer...', icon: 'sparkles', status: 'pending', detail: '' },
  ];

  const advanceStep = (steps: PipelineStep[], stepId: string, detail?: string): PipelineStep[] => {
    return steps.map(s => {
      if (s.id === stepId) return { ...s, status: 'active' as StepStatus, detail: detail || s.detail };
      // Everything before becomes done
      const idx = steps.findIndex(x => x.id === stepId);
      const myIdx = steps.findIndex(x => x.id === s.id);
      if (myIdx < idx) return { ...s, status: 'done' as StepStatus };
      return s;
    });
  };

  const completeAllSteps = (steps: PipelineStep[]): PipelineStep[] => {
    return steps.map(s => ({ ...s, status: 'done' as StepStatus }));
  };

  /* ------ Simulated pipeline timing (fallback when backend doesn't send step events) ------ */

  const simulatePipeline = useCallback((onComplete: () => void) => {
    const steps = initPipeline();
    setPipelineSteps(steps);

    // Step 1: Understanding (already active)
    const t1 = setTimeout(() => {
      setPipelineSteps(prev => advanceStep(prev, 'search', 'Searching 5 sources...'));
    }, 1200);

    const t2 = setTimeout(() => {
      setPipelineSteps(prev => advanceStep(prev, 'analyze', 'Reading articles...'));
    }, 3000);

    const t3 = setTimeout(() => {
      setPipelineSteps(prev => advanceStep(prev, 'generate'));
      onComplete();
    }, 4500);

    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
  }, []);

  /* ------ HTTP-based search (primary) ------ */

  const doSearchHTTP = useCallback(async (query: string) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    try {
      // Try dedicated search endpoint first (GET with query params)
      const params = new URLSearchParams({ q: query, limit: '8' });
      const url = `${HTTP}/api/search?${params}`;
      const r = await fetch(url, {
        method: 'GET',
        headers: getAuthHeaders(),
        signal: ctrl.signal,
      });

      if (!r.ok) {
        // If search endpoint doesn't exist, fall back to chat
        if (r.status === 404 || r.status === 405) {
          return false; // Signal to use fallback
        }
        const d = await r.json().catch(() => ({}));
        throw new Error((d as any).detail || `Error ${r.status}`);
      }

      const data = await r.json();
      setPipelineSteps(prev => completeAllSteps(prev));

      const result: SearchResult = {
        query,
        answer: data.answer || '',
        sources: (data.sources || []).map((s: any) => ({
          url: s.url,
          title: s.title || '',
          snippet: s.snippet || '',
          domain: s.domain || getDomain(s.url),
          favicon: s.favicon || getFavicon(s.url),
        })),
        relatedSearches: data.related_searches || data.relatedSearches || [],
        timestamp: Date.now(),
      };
      setResults(prev => [...prev, result]);
      return true;
    } catch (err: any) {
      if (err.name === 'AbortError') return true; // Cancelled, not an error
      throw err;
    }
  }, []);

  /* ------ Fallback: send as chat message ------ */

  const doSearchFallback = useCallback(async (query: string) => {
    const model = getModel('search');

    // Use WS if available for streaming
    if (ws && ws.readyState === WebSocket.OPEN) {
      return new Promise<void>((resolve) => {
        let answer = '';

        const handleMsg = (ev: MessageEvent) => {
          let d: any;
          try { d = JSON.parse(ev.data); } catch { return; }

          if (d.type === 'chunk') {
            answer += d.content || '';
            setStreamingAnswer(answer);
            setPipelineSteps(prev => {
              const gen = prev.find(s => s.id === 'generate');
              if (gen?.status !== 'active' && gen?.status !== 'done') {
                return advanceStep(prev, 'generate');
              }
              return prev;
            });
          } else if (d.type === 'done') {
            ws.removeEventListener('message', handleMsg);
            setPipelineSteps(prev => completeAllSteps(prev));
            const finalAnswer = answer || d.content || '';
            const result: SearchResult = {
              query,
              answer: finalAnswer,
              sources: [],
              relatedSearches: [],
              timestamp: Date.now(),
            };
            setResults(prev => [...prev, result]);
            setStreamingAnswer('');
            useStore.getState().setActiveStream(null);
            resolve();
          } else if (d.type === 'error') {
            ws.removeEventListener('message', handleMsg);
            setError(d.content || d.error || 'Search error');
            useStore.getState().setActiveStream(null);
            resolve();
          }
        };

        ws.addEventListener('message', handleMsg);

        // Store the active stream so the global ws handler doesn't eat our messages
        useStore.getState().setActiveStream({
          type: 'search',
          rawText: '',
          onFirstChunk: null,
          onDone: null,
        });

        ws.send(JSON.stringify({
          message: `Search the web for: ${query}`,
          model: model || undefined,
        }));
      });
    } else {
      // HTTP fallback to chat endpoint
      try {
        const ctrl = new AbortController();
        abortRef.current = ctrl;
        const r = await fetch(`${HTTP}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
          body: JSON.stringify({
            message: `Search the web for: ${query}`,
            model: model || undefined,
          }),
          signal: ctrl.signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        setPipelineSteps(prev => completeAllSteps(prev));
        const result: SearchResult = {
          query,
          answer: data.reply || data.response || data.answer || '',
          sources: [],
          relatedSearches: [],
          timestamp: Date.now(),
        };
        setResults(prev => [...prev, result]);
      } catch (err: any) {
        setError(err.message || 'Search failed');
      }
    }
  }, [ws, getModel]);

  /* ------ Main search handler ------ */

  const doSearch = useCallback(async (query: string) => {
    query = query.trim();
    if (!query || searching) return;

    // Update recent searches
    const updated = addRecentSearch(query, recentSearches);
    setRecentSearches(updated);
    saveRecentSearches(updated);

    setSearching(true);
    setError('');
    setStreamingAnswer('');
    setLiveSources([]);

    // Start pipeline animation
    let cleanupSim: (() => void) | null = null;

    const steps = initPipeline();
    setPipelineSteps(steps);

    try {
      // Try dedicated search endpoint (POST)
      let handled = false;
      try {
        cleanupSim = simulatePipeline(() => {});
        handled = await doSearchHTTP(query);
      } catch {
        handled = false;
      }

      if (!handled) {
        // Fallback — try WS search or chat
        if (cleanupSim) cleanupSim();
        cleanupSim = simulatePipeline(() => {});
        await doSearchFallback(query);
      }
    } catch (err: any) {
      setError(err.message || 'Search failed');
    } finally {
      if (cleanupSim) cleanupSim();
      setSearching(false);
      setPipelineSteps(prev => prev.length > 0 ? completeAllSteps(prev) : prev);
      // Clear input
      if (inputRef.current) inputRef.current.value = '';
    }
  }, [searching, recentSearches, simulatePipeline, doSearchHTTP, doSearchFallback]);

  /* ------ Clear ------ */

  const clearSearch = () => {
    if (abortRef.current) abortRef.current.abort();
    setResults([]);
    setSearching(false);
    setError('');
    setStreamingAnswer('');
    setLiveSources([]);
    setPipelineSteps([]);
    if (inputRef.current) {
      inputRef.current.value = '';
      inputRef.current.focus();
    }
  };

  // Attach delegated handler for related pills
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const handler = (e: MouseEvent) => {
      const btn = (e.target as HTMLElement).closest('.search-related-pill') as HTMLButtonElement | null;
      if (!btn) return;
      const text = btn.textContent?.trim();
      if (text) doSearch(text);
    };
    el.addEventListener('click', handler);
    return () => el.removeEventListener('click', handler);
  }, [doSearch]);

  const showEmpty = !searching && results.length === 0 && !error;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Search bar */}
      <div style={{
        flexShrink: 0,
        padding: '10px 12px',
        borderBottom: '1px solid var(--b1)',
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-lg)',
          padding: '0 12px',
          backdropFilter: 'blur(12px)',
          transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
        }}
        onFocus={e => {
          e.currentTarget.style.borderColor = 'rgba(124, 58, 237, 0.35)';
          e.currentTarget.style.boxShadow = '0 0 0 3px rgba(124, 58, 237, 0.08)';
        }}
        onBlur={e => {
          e.currentTarget.style.borderColor = 'var(--b1)';
          e.currentTarget.style.boxShadow = 'none';
        }}
        >
          <Search size={14} style={{ color: 'var(--mu)', flexShrink: 0 }} />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search the web with AI..."
            autoFocus
            disabled={searching}
            onKeyDown={e => {
              if (e.key === 'Enter') doSearch(inputRef.current?.value || '');
            }}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              color: 'var(--tx)',
              fontSize: '13px',
              padding: '9px 0',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
          {searching ? (
            <div className="aura-thinking" style={{ transform: 'scale(0.7)' }}>
              <span /><span /><span />
            </div>
          ) : (
            <button
              onClick={() => doSearch(inputRef.current?.value || '')}
              style={{
                background: 'var(--p)',
                border: 'none',
                borderRadius: '50%',
                width: 26,
                height: 26,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                cursor: 'pointer',
                flexShrink: 0,
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'var(--p2)'; }}
              onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = 'var(--p)'; }}
            >
              <Search size={12} />
            </button>
          )}
        </div>
      </div>

      {/* Model pill + clear */}
      <div className="flex items-center justify-between px-3 py-1.5 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: '11px', color: 'var(--mu)' }}>Model:</span>
          <ModelPill featureKey="search" />
        </div>
        {results.length > 0 && (
          <button
            onClick={clearSearch}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 4,
              background: 'none',
              border: 'none',
              color: 'var(--mu)',
              fontSize: '10.5px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              padding: '2px 6px',
              borderRadius: 'var(--r-sm)',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = 'var(--rd)'; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = 'var(--mu)'; }}
          >
            <Trash2 size={11} />
            Clear
          </button>
        )}
      </div>

      {/* Results area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 pb-4" data-scroll-panel>

        {/* Empty state */}
        {showEmpty && (
          <div style={{ paddingTop: 32, textAlign: 'center' }}>
            <div style={{
              width: 48,
              height: 48,
              borderRadius: '50%',
              background: 'var(--pg)',
              border: '1px solid rgba(124, 58, 237, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 12px',
            }}>
              <Search size={20} style={{ color: 'var(--pl)' }} />
            </div>
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)', marginBottom: 4 }}>
              AI-Powered Search
            </div>
            <div style={{ fontSize: '11.5px', color: 'var(--mu)', marginBottom: 20 }}>
              Get answers with sources from across the web
            </div>

            {/* Recent searches */}
            {recentSearches.length > 0 && (
              <div style={{ textAlign: 'left' }}>
                <div style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--mu)',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                }}>
                  <Clock size={10} />
                  Recent searches
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {recentSearches.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        if (inputRef.current) inputRef.current.value = q;
                        doSearch(q);
                      }}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        padding: '7px 10px',
                        background: 'transparent',
                        border: '1px solid transparent',
                        borderRadius: 'var(--r-md)',
                        color: 'var(--tx)',
                        fontSize: '12px',
                        cursor: 'pointer',
                        fontFamily: 'inherit',
                        textAlign: 'left',
                        width: '100%',
                        transition: 'all 0.15s ease',
                      }}
                      onMouseEnter={e => {
                        (e.currentTarget as HTMLElement).style.background = 'var(--s2)';
                        (e.currentTarget as HTMLElement).style.borderColor = 'var(--b1)';
                      }}
                      onMouseLeave={e => {
                        (e.currentTarget as HTMLElement).style.background = 'transparent';
                        (e.currentTarget as HTMLElement).style.borderColor = 'transparent';
                      }}
                    >
                      <Search size={12} style={{ color: 'var(--mu)', flexShrink: 0 }} />
                      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '10px 12px',
            background: 'rgba(239, 68, 68, 0.08)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: 'var(--r-md)',
            marginTop: 12,
            fontSize: '12px',
            color: 'var(--rd)',
          }}>
            <X size={14} />
            {error}
          </div>
        )}

        {/* Previous results thread */}
        {results.map((r, i) => (
          <ResultEntry key={r.timestamp + '-' + i} result={r} isLatest={i === results.length - 1 && !searching} />
        ))}

        {/* Active search: pipeline + streaming answer */}
        {searching && (
          <div style={{ paddingTop: 4 }}>
            {/* Show the current query */}
            {inputRef.current?.value && (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '10px 0 0',
              }}>
                <Search size={13} style={{ color: 'var(--pl)', flexShrink: 0 }} />
                <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>
                  {inputRef.current.value}
                </span>
              </div>
            )}

            {/* Pipeline steps */}
            {pipelineSteps.length > 0 && <PipelineView steps={pipelineSteps} />}

            {/* Live sources appearing */}
            {liveSources.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                  color: 'var(--mu)',
                  marginBottom: 8,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}>
                  <Globe size={11} />
                  Sources ({liveSources.length})
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {liveSources.map((src, i) => (
                    <SourceCard key={i} source={src} index={i} />
                  ))}
                </div>
              </div>
            )}

            {/* Streaming answer */}
            {streamingAnswer && (
              <div style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-lg)',
                padding: '14px 16px',
                marginBottom: 12,
              }}>
                <div
                  className="md-body"
                  style={{ fontSize: '12.5px', lineHeight: 1.7 }}
                  dangerouslySetInnerHTML={{ __html: md(streamingAnswer) }}
                />
                <span className="streaming-cursor" />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Inline CSS for search-specific animations */}
      <style>{`
        @keyframes searchStepPulse {
          0%, 100% {
            box-shadow: 0 0 0 0 rgba(124, 58, 237, 0.2);
          }
          50% {
            box-shadow: 0 0 0 6px rgba(124, 58, 237, 0.05), 0 0 12px rgba(124, 58, 237, 0.1);
          }
        }
        @keyframes searchStepSpin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        :root.light .search-source-card:hover {
          box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
        }
      `}</style>
    </div>
  );
}
