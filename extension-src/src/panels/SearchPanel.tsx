import React, { useState, useRef } from 'react';
import { Search } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import { md } from '../markdown';

export default function SearchPanel() {
  const { getModel } = useStore();
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const doSearch = async (q: string) => {
    q = q.trim();
    if (!q) return;
    setLoading(true);
    setError('');
    setResults(null);
    try {
      const model = getModel('search');
      const url = `${HTTP}/api/search?q=${encodeURIComponent(q)}&limit=5` +
        (model ? `&model=${encodeURIComponent(model)}` : '');
      const r = await fetch(url);
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error((d as any).detail || `Error ${r.status}`);
      }
      setResults(await r.json());
    } catch (err: any) {
      setError(err.message || 'Search error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Search bar */}
      <div className="flex-shrink-0 p-3 flex gap-2" style={{ borderBottom: '1px solid var(--b1)' }}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Search the web…"
          autoFocus
          onKeyDown={e => { if (e.key === 'Enter') doSearch(inputRef.current?.value || ''); }}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12.5px',
            padding: '7px 10px',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={() => doSearch(inputRef.current?.value || '')}
          disabled={loading}
          style={{
            background: 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '7px 14px',
            cursor: 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {loading ? '…' : 'Go'}
        </button>
      </div>

      <div className="flex items-center gap-2 px-3 py-1.5 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <span style={{ fontSize: '11px', color: 'var(--mu)' }}>Model:</span>
        <ModelPill featureKey="search" />
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-3">
        {loading && (
          <div className="flex justify-center mt-8">
            <div className="dots"><span /><span /><span /></div>
          </div>
        )}
        {error && (
          <div style={{ color: 'var(--rd)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            ⚠ {error}
          </div>
        )}
        {results && (
          <>
            {results.sources?.length > 0 && (
              <div className="mb-4">
                <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 8 }}>
                  Sources
                </div>
                <div className="flex flex-col gap-2">
                  {results.sources.map((src: any, i: number) => (
                    <a
                      key={i}
                      href={src.url}
                      target="_blank"
                      rel="noopener"
                      style={{
                        display: 'block',
                        background: 'var(--s2)',
                        border: '1px solid var(--b1)',
                        borderRadius: 'var(--r-md)',
                        padding: '8px 10px',
                        textDecoration: 'none',
                      }}
                    >
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                        <span style={{ fontSize: '10px', color: 'var(--mu)', flexShrink: 0 }}>{i + 1}</span>
                        <span style={{ fontSize: '12px', color: 'var(--pl)', fontWeight: 500 }}>{src.title}</span>
                      </div>
                      {src.snippet && (
                        <div style={{ fontSize: '11px', color: 'var(--mu)', marginTop: 3 }}>{src.snippet}</div>
                      )}
                      <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 2 }}>{src.url}</div>
                    </a>
                  ))}
                </div>
              </div>
            )}
            {results.answer && (
              <div>
                <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 8 }}>
                  Answer
                </div>
                <div
                  className="md-body"
                  style={{ fontSize: '12.5px', lineHeight: 1.65 }}
                  dangerouslySetInnerHTML={{ __html: md(results.answer) }}
                />
              </div>
            )}
            {!results.sources?.length && !results.answer && (
              <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
                No results found.
              </div>
            )}
          </>
        )}
        {!loading && !error && !results && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            Search the web with AI-powered answers
          </div>
        )}
      </div>
    </div>
  );
}
