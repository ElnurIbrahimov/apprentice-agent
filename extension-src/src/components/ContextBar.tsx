import React from 'react';
import { X } from 'lucide-react';
import { useStore } from '../store';

export default function ContextBar() {
  const { pendingCtx, setPendingCtx } = useStore();
  if (!pendingCtx) return null;

  const label = pendingCtx.title || pendingCtx.url || 'Context';
  const preview = pendingCtx.text.slice(0, 80) + (pendingCtx.text.length > 80 ? '…' : '');

  return (
    <div
      className="flex items-start gap-2 px-3 py-2 flex-shrink-0"
      style={{
        background: 'var(--pg)',
        borderBottom: '1px solid var(--b1)',
        fontSize: '11px',
      }}
    >
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate" style={{ color: 'var(--pl)' }}>
          {label}
        </div>
        <div className="truncate" style={{ color: 'var(--mu)' }}>
          {preview}
        </div>
      </div>
      <button
        onClick={() => setPendingCtx(null)}
        className="flex-shrink-0 mt-0.5"
        style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 0 }}
      >
        <X size={13} />
      </button>
    </div>
  );
}
