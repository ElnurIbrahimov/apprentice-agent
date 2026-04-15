/**
 * SharePanel — browse and manage active shared projects via /api/shares.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Share2, ExternalLink, Trash2, Copy, RefreshCw, CheckCircle2 } from 'lucide-react';
import { share } from '../api/client';
import type { ShareItem } from '../api/types';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatExpiresIn(ts: number): string {
  const ms = ts * 1000 - Date.now();
  if (ms <= 0) return 'expired';
  const days = Math.floor(ms / 86400_000);
  if (days > 0) return `${days}d left`;
  const hours = Math.floor(ms / 3600_000);
  return `${hours}h left`;
}

export default function SharePanel() {
  const [items, setItems] = useState<ShareItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setItems(await share.list());
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const remove = useCallback(async (id: string) => {
    try {
      await share.remove(id);
      setItems((i) => i.filter((s) => s.id !== id));
    } catch { /* silent */ }
  }, []);

  const copyUrl = useCallback(async (url: string, id: string) => {
    try {
      await navigator.clipboard.writeText(url);
      setCopied(id);
      setTimeout(() => setCopied(null), 1500);
    } catch { /* silent */ }
  }, []);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Share2 size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          Shares
          <span style={{ color: 'var(--mu)', fontWeight: 400, marginLeft: 6 }}>{items.length}</span>
        </span>
        <button onClick={load} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      {loading && <div style={{ fontSize: 11, color: 'var(--mu)' }}>Loading…</div>}

      {!loading && items.length === 0 && (
        <div style={{ fontSize: 11, color: 'var(--mu)', textAlign: 'center', padding: 20 }}>
          No active shares.
        </div>
      )}

      {items.map((item) => (
        <div key={item.id} style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {item.project_name}
              </div>
              <div style={{ fontSize: 9, color: 'var(--mu)' }}>
                {item.file_count} files · {formatBytes(item.total_bytes)} · {formatExpiresIn(item.expires_at)}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 10, color: 'var(--p)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', marginBottom: 8, fontFamily: 'monospace' }}>
            {item.url}
          </div>

          <div style={{ display: 'flex', gap: 6 }}>
            <button
              onClick={() => window.open(item.url, '_blank', 'noopener')}
              style={{ padding: '5px 10px', background: 'var(--p)', border: 'none', borderRadius: 6, color: '#fff', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
            >
              <ExternalLink size={10} /> Open
            </button>
            <button
              onClick={() => copyUrl(item.url, item.id)}
              style={{ padding: '5px 10px', background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 6, color: 'var(--tx)', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
            >
              {copied === item.id ? <CheckCircle2 size={10} style={{ color: '#34d399' }} /> : <Copy size={10} />}
              {copied === item.id ? 'Copied' : 'Copy'}
            </button>
            <button
              onClick={() => remove(item.id)}
              style={{ padding: '5px 10px', background: 'transparent', border: '1px solid rgba(248, 113, 113, 0.3)', borderRadius: 6, color: '#f87171', fontSize: 10, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, marginLeft: 'auto' }}
            >
              <Trash2 size={10} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
