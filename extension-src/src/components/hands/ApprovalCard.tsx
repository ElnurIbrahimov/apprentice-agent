import React from 'react';
import { Check, X, Bell } from 'lucide-react';
import type { HandApprovalRequest } from '../../types';

interface Props {
  approval: HandApprovalRequest;
  onApprove: (name: string, requestId: string) => void;
  onDeny: (name: string, requestId: string) => void;
}

export default function ApprovalCard({ approval, onApprove, onDeny }: Props) {
  const argsPreview = (() => {
    try { return JSON.stringify(approval.args).slice(0, 120); }
    catch { return ''; }
  })();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        padding: '9px 10px',
        borderRadius: 10,
        background: 'rgba(245,158,11,0.08)',
        border: '1px solid rgba(245,158,11,0.35)',
        boxShadow: '0 0 0 1px rgba(245,158,11,0.1), 0 2px 8px rgba(0,0,0,0.2)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <Bell size={12} style={{ color: 'var(--yl, #f59e0b)', flexShrink: 0 }} />
        <span style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          {approval.hand_name} → {approval.tool_name}
        </span>
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
          onClick={() => onApprove(approval.hand_name, approval.request_id)}
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
            cursor: 'pointer',
            fontSize: 11,
            fontFamily: 'inherit',
          }}
        >
          <Check size={11} /> Approve
        </button>
        <button
          onClick={() => onDeny(approval.hand_name, approval.request_id)}
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
            cursor: 'pointer',
            fontSize: 11,
            fontFamily: 'inherit',
          }}
        >
          <X size={11} /> Deny
        </button>
      </div>
    </div>
  );
}
