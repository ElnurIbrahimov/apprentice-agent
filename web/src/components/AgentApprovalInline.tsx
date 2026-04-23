/**
 * Inline approval prompt rendered into the active agent run when a
 * server-side Hand pauses for a sensitive-action approval. Non-blocking:
 * the server is waiting on us, not the other way around.
 */

import { ShieldCheckIcon, XMarkIcon, CheckIcon } from '@heroicons/react/24/outline';
import type { PendingApproval } from '../hooks/useHandApproval';

interface Props {
  pending: PendingApproval;
  onAllow: () => void;
  onDeny: () => void;
  onDismiss: () => void;
  resolving?: boolean;
}

export function AgentApprovalInline({ pending, onAllow, onDeny, onDismiss, resolving }: Props) {
  const argsPreview = formatArgs(pending.args);

  return (
    <div
      className="rounded-xl p-3.5 animate-slide-up-fade"
      style={{
        background: 'linear-gradient(180deg, rgba(245,158,11,0.12) 0%, rgba(245,158,11,0.04) 100%)',
        border: '1px solid rgba(245,158,11,0.35)',
        boxShadow: '0 8px 28px rgba(0,0,0,0.2)',
      }}
    >
      <div className="flex items-center gap-2 mb-2">
        <ShieldCheckIcon className="w-4 h-4" style={{ color: '#fbbf24' }} />
        <span className="text-xs font-semibold" style={{ color: '#fcd34d' }}>
          Approval required
        </span>
        <span className="text-[10px] text-chat-text-secondary">
          Hand <span className="font-mono">{pending.hand_name}</span> wants to use
          {' '}<span className="font-mono">{pending.tool_name}</span>
        </span>
        <div className="flex-1" />
        <button
          type="button"
          onClick={onDismiss}
          disabled={resolving}
          className="text-chat-text-secondary hover:text-chat-text transition-colors"
          title="Dismiss (the server will time out)"
          aria-label="Dismiss"
        >
          <XMarkIcon className="w-4 h-4" />
        </button>
      </div>

      {argsPreview && (
        <pre
          className="text-[11px] font-mono leading-relaxed mb-3 overflow-auto rounded p-2 whitespace-pre-wrap break-words"
          style={{
            background: 'rgba(0,0,0,0.25)',
            color: 'var(--text-secondary)',
            maxHeight: 180,
            border: '1px solid var(--border-subtle)',
          }}
        >
          {argsPreview}
        </pre>
      )}

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onAllow}
          disabled={resolving}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium text-white transition-all disabled:opacity-50"
          style={{
            background: 'rgba(34,197,94,0.9)',
            boxShadow: '0 2px 10px rgba(34,197,94,0.3)',
          }}
        >
          <CheckIcon className="w-3.5 h-3.5" />
          Allow once
        </button>
        <button
          type="button"
          onClick={onDeny}
          disabled={resolving}
          className="flex-1 inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-all disabled:opacity-50"
          style={{
            background: 'rgba(239,68,68,0.18)',
            color: '#fca5a5',
            border: '1px solid rgba(239,68,68,0.35)',
          }}
        >
          <XMarkIcon className="w-3.5 h-3.5" />
          Deny
        </button>
      </div>
    </div>
  );
}

function formatArgs(args: Record<string, unknown>): string {
  if (!args || Object.keys(args).length === 0) return '';
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}
