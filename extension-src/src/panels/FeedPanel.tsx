/**
 * FeedPanel — browse captures saved to /api/feed/list.
 *
 * Click an item to see detail, delete removes it. Counterpart to CapturePanel
 * (which writes to /api/feed/save).
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Rss, RefreshCw, Trash2, ExternalLink, Loader2 } from 'lucide-react';
import { feed } from '../api/client';
import type { FeedItemSummary } from '../api/types';

export default function FeedPanel() {
  const [items, setItems] = useState<FeedItemSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<FeedItemSummary | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await feed.list(50, 0);
      setItems(r.items ?? []);
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const openDetail = useCallback(async (item: FeedItemSummary) => {
    setSelected(item);
    setDetailLoading(true);
    try {
      setDetail(await feed.get(item.id));
    } catch {
      setDetail(null);
    }
    setDetailLoading(false);
  }, []);

  const remove = useCallback(async (id: string) => {
    try {
      await feed.remove(id);
      setItems((i) => i.filter((it) => it.id !== id));
      if (selected?.id === id) {
        setSelected(null);
        setDetail(null);
      }
    } catch { /* silent */ }
  }, [selected]);

  return (
    <div className="panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--b1)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Rss size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          Feed
          <span style={{ color: 'var(--mu)', fontWeight: 400, marginLeft: 6 }}>{items.length}</span>
        </span>
        <button onClick={load} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      <div style={{ flex: 1, padding: 14, overflowY: 'auto' }}>
        {loading && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Loading…</div>}

        {!loading && items.length === 0 && (
          <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 20 }}>
            No saved captures yet. Use the Capture panel to save pages or components.
          </div>
        )}

        {items.map((item) => (
          <div
            key={item.id}
            onClick={() => openDetail(item)}
            style={{
              display: 'flex',
              gap: 10,
              padding: 10,
              marginBottom: 6,
              background: selected?.id === item.id ? 'var(--p)' : 'var(--s2)',
              color: selected?.id === item.id ? '#fff' : 'var(--tx)',
              border: '1px solid var(--b1)',
              borderRadius: 8,
              cursor: 'pointer',
              alignItems: 'center',
            }}
          >
            {item.thumbnail && (
              <img
                src={item.thumbnail}
                alt=""
                style={{ width: 48, height: 48, borderRadius: 6, objectFit: 'cover', flexShrink: 0 }}
              />
            )}
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 11, fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {item.title || `${item.type} capture`}
              </div>
              <div style={{ fontSize: 9, color: selected?.id === item.id ? 'rgba(255,255,255,0.7)' : 'var(--mu)', marginTop: 2 }}>
                {item.type} · {new Date(item.timestamp * 1000).toLocaleDateString()}
              </div>
              {item.source_url && (
                <div style={{ fontSize: 9, color: selected?.id === item.id ? 'rgba(255,255,255,0.6)' : 'var(--mu)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginTop: 2 }}>
                  {item.source_url}
                </div>
              )}
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); remove(item.id); }}
              style={{ background: 'none', border: 'none', color: selected?.id === item.id ? '#fff' : 'var(--mu)', cursor: 'pointer', padding: 4, flexShrink: 0 }}
            >
              <Trash2 size={12} />
            </button>
          </div>
        ))}
      </div>

      {selected && (
        <div style={{ padding: 14, borderTop: '1px solid var(--b1)', background: 'var(--s2)', maxHeight: '40%', overflowY: 'auto' }}>
          <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 6 }}>
            Detail
          </div>
          {detailLoading && <Loader2 size={12} className="spin" />}
          {detail && (
            <pre style={{ fontSize: 9, color: 'var(--tx)', overflow: 'auto', maxHeight: 200, margin: 0 }}>
              {JSON.stringify(detail, null, 2)}
            </pre>
          )}
          {selected.source_url && (
            <button
              onClick={() => window.open(selected.source_url!, '_blank', 'noopener')}
              style={{ marginTop: 8, padding: '6px 12px', background: 'var(--p)', border: 'none', borderRadius: 6, color: '#fff', fontSize: 11, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 4 }}
            >
              <ExternalLink size={11} /> Open source
            </button>
          )}
        </div>
      )}
    </div>
  );
}
