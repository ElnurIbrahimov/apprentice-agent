/**
 * AmbientSurfaceCard — shows a soft recall when the user browses to a page
 * that matches past memories or lifelog entries.
 *
 * Mounted once in the sidebar header row. Listens for AMBIENT_SURFACE_HINT
 * from the background service worker. Dismissable; auto-hides after 30s.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Sparkles, X, HardDrive } from 'lucide-react';
import ext from '../ext';
import { useStore } from '../store';

interface SurfaceHint {
  url: string;
  title: string;
  memories: Array<{ id: string; content: string }>;
  lifelog: Array<{ url: string; title?: string; snippet?: string; ts_ms: number }>;
}

export default function AmbientSurfaceCard() {
  const [hint, setHint] = useState<SurfaceHint | null>(null);
  const setPanel = useStore((s) => s.setPanel);

  useEffect(() => {
    const handler = (msg: any) => {
      if (msg?.type !== 'AMBIENT_SURFACE_HINT') return;
      setHint({
        url: msg.url || '',
        title: msg.title || '',
        memories: Array.isArray(msg.memories) ? msg.memories : [],
        lifelog: Array.isArray(msg.lifelog) ? msg.lifelog : [],
      });
    };
    ext?.runtime?.onMessage?.addListener(handler);
    return () => ext?.runtime?.onMessage?.removeListener(handler);
  }, []);

  useEffect(() => {
    if (!hint) return;
    const t = setTimeout(() => setHint(null), 30_000);
    return () => clearTimeout(t);
  }, [hint]);

  const dismiss = useCallback(() => setHint(null), []);

  if (!hint) return null;

  const memCount = hint.memories.length;
  const logCount = hint.lifelog.length;
  const firstMem = hint.memories[0];
  const firstLog = hint.lifelog[0];

  return (
    <div
      style={{
        margin: '6px 10px',
        padding: '8px 10px',
        background: 'linear-gradient(135deg, rgba(124, 58, 237, 0.12), rgba(167, 139, 250, 0.06))',
        border: '1px solid rgba(124, 58, 237, 0.3)',
        borderRadius: 10,
        display: 'flex',
        alignItems: 'start',
        gap: 8,
        fontSize: 11,
        color: 'var(--tx)',
        animation: 'fadeIn 0.3s ease-out',
      }}
      role="status"
      aria-live="polite"
    >
      <Sparkles size={13} style={{ color: 'var(--p)', flexShrink: 0, marginTop: 1 }} />

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontWeight: 600, marginBottom: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          You've seen this area before
        </div>
        <div style={{ color: 'var(--mu)', fontSize: 10 }}>
          {memCount > 0 && <span>{memCount} memor{memCount === 1 ? 'y' : 'ies'}</span>}
          {memCount > 0 && logCount > 0 && <span> · </span>}
          {logCount > 0 && <span>{logCount} prior visit{logCount === 1 ? '' : 's'}</span>}
        </div>
        {firstMem && (
          <div style={{ marginTop: 4, fontSize: 10, color: 'var(--mu)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            💭 {firstMem.content.slice(0, 80)}
          </div>
        )}
        {!firstMem && firstLog && (
          <div style={{ marginTop: 4, fontSize: 10, color: 'var(--mu)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            🔗 {firstLog.title || firstLog.url}
          </div>
        )}
        {memCount > 0 && (
          <button
            onClick={() => {
              setPanel('memory-browser');
              dismiss();
            }}
            style={{
              marginTop: 6,
              padding: '3px 8px',
              background: 'rgba(124, 58, 237, 0.2)',
              border: '1px solid rgba(124, 58, 237, 0.4)',
              borderRadius: 6,
              color: 'var(--pl)',
              fontSize: 10,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: 3,
            }}
          >
            <HardDrive size={9} /> Open memory
          </button>
        )}
      </div>

      <button
        onClick={dismiss}
        aria-label="Dismiss"
        style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 2, flexShrink: 0 }}
      >
        <X size={12} />
      </button>
    </div>
  );
}
