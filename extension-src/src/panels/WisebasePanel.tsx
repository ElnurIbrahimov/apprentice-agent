import React, { useState, useEffect, useRef } from 'react';
import { Search, Trash2 } from 'lucide-react';
import { HTTP, apiFetch } from '../api';
import { useStore } from '../store';

interface Entry {
  id: string;
  title?: string;
  text?: string;
  source_type?: string;
  created_at?: string;
}

export default function WisebasePanel() {
  const { activePanel } = useStore();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const load = async (q?: string) => {
    setLoading(true);
    setError('');
    try {
      const url = q
        ? `${HTTP}/api/knowledge/search?q=${encodeURIComponent(q)}&limit=20`
        : `${HTTP}/api/knowledge?limit=50`;
      const data = await apiFetch(url);
      setEntries(data.entries || data.results || []);
    } catch (err: any) {
      setError(err.message || 'Failed to load');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activePanel === 'wisebase') load();
  }, [activePanel]);

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
    <div className="flex flex-col h-full overflow-hidden">
      {/* Search */}
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div className="flex-1 relative">
          <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: 'var(--mu)' }} />
          <input
            ref={searchRef}
            value={query}
            onChange={e => handleSearch(e.target.value)}
            placeholder="Search knowledge base…"
            autoFocus
            style={{
              width: '100%',
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--tx)',
              fontSize: '12px',
              padding: '6px 8px 6px 28px',
              outline: 'none',
              fontFamily: 'inherit',
            }}
          />
        </div>
        <button
          onClick={() => { setQuery(''); load(); }}
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--mu)',
            fontSize: '11px',
            padding: '6px 10px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          All
        </button>
      </div>

      {/* Entries */}
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
        {loading && (
          <div className="flex justify-center mt-8">
            <div className="dots"><span /><span /><span /></div>
          </div>
        )}
        {error && <div style={{ color: 'var(--rd)', fontSize: '12px' }}>⚠ {error}</div>}
        {!loading && entries.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            {query ? 'No results found.' : 'Your knowledge base is empty.'}
          </div>
        )}
        {entries.map(entry => (
          <div
            key={entry.id}
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
            }}
          >
            <div className="flex items-start justify-between gap-2">
              <div style={{ flex: 1, minWidth: 0 }}>
                {entry.title && (
                  <div style={{ fontWeight: 500, fontSize: '12.5px', marginBottom: 4, color: 'var(--pl)' }}>
                    {entry.title}
                  </div>
                )}
                <div style={{ fontSize: '12px', color: 'var(--mu)', overflow: 'hidden', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical' }}>
                  {entry.text}
                </div>
                <div style={{ fontSize: '10px', color: 'var(--di)', marginTop: 6 }}>
                  {entry.source_type} · {entry.created_at ? new Date(entry.created_at).toLocaleDateString() : ''}
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
    </div>
  );
}
