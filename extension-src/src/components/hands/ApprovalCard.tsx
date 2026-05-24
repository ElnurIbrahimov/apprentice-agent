import React, { useState } from 'react';
import { Check, X, Bell, Loader2 } from 'lucide-react';
import type { HandApprovalRequest } from '../../types';

interface Props {
  approval: HandApprovalRequest;
  onApprove: (name: string, requestId: string) => void;
  onDeny: (name: string, requestId: string) => void;
}

function formatAge(ageSeconds: number): string {
  if (ageSeconds < 60) return 'just now';
  if (ageSeconds < 3600) return `${Math.floor(ageSeconds / 60)}m ago`;
  if (ageSeconds < 86400) return `${Math.floor(ageSeconds / 3600)}h ago`;
  return `${Math.floor(ageSeconds / 86400)}d ago`;
}

function ageBorderStyle(ageSeconds: number): string {
  if (ageSeconds < 120) return '1px solid rgba(245,158,11,0.35)';
  if (ageSeconds < 600) return '1px solid rgba(234,179,8,0.4)';
  if (ageSeconds < 3600) return '1px solid rgba(249,115,22,0.4)';
  return '1px solid rgba(239,68,68,0.4)';
}

export default function ApprovalCard({ approval, onApprove, onDeny }: Props) {
  const [pending, setPending] = useState<'approve' | 'deny' | null>(null);
  const argsPreview = (() => {
    try { return JSON.stringify(approval.args).slice(0, 120); }
    catch { return ''; }
  })();
  const age = approval.age_seconds ?? 0;
  const borderStyle = ageBorderStyle(age);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        padding: '9px 10px',
        borderRadius: 10,
        background: 'rgba(245,158,11,0.08)',
        border: borderStyle,
        boxShadow: '0 0 0 1px rgba(245,158,11,0.1), 0 2px 8px rgba(0,0,0,0.2)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Bell size={12} style={{ color: 'var(--yl, #f59e0b)', flexShrink: 0 }} />
        <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          {approval.hand_name} → {approval.tool_name}
        </span>
        <span style={{ fontSize: 10, color: 'var(--mu)' }}>{formatAge(age)}</span>
      </div>
      {argsPreview && (
        <div style={{
          fontSize: 10.5,
          color: 'var(--mu)',
          fontFamily: 'monospace',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          {argsPreview}
        </div>
      )}
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          onClick={() => { setPending('approve'); onApprove(approval.hand_name, approval.request_id); }}
          disabled={pending !== null}
          style={{
            flex: 1,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
            background: 'rgba(34,197,94,0.18)',
            border: '1px solid rgba(34,197,94,0.4)',
            borderRadius: 6,
            color: 'var(--gr, #22c55e)',
            padding: '5px 8px',
            cursor: pending ? 'not-allowed' : 'pointer',
            fontSize: 11,
            fontFamily: 'inherit',
            opacity: pending ? 0.6 : 1,
          }}
        >
          {pending === 'approve' ? <><Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> pending...</> : <><Check size={11} /> Approve</>}
        </button>
        <button
          onClick={() => { setPending('deny'); onDeny(approval.hand_name, approval.request_id); }}
          disabled={pending !== null}
          style={{
            flex: 1,
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4,
            background: 'transparent',
            border: '1px solid var(--b1)',
            borderRadius: 6,
            color: 'var(--mu)',
            padding: '5px 8px',
            cursor: pending ? 'not-allowed' : 'pointer',
            fontSize: 11,
            fontFamily: 'inherit',
            opacity: pending ? 0.6 : 1,
          }}
        >
          {pending === 'deny' ? <><Loader2 size={11} style={{ animation: 'spin 1s linear infinite' }} /> pending...</> : <><X size={11} /> Deny</>}
        </button>
      </div>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
