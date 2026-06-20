import React, { useState } from 'react';
import { Play, Pause, Power, PowerOff, Trash2, ChevronDown, Loader2, AlertCircle } from 'lucide-react';
import type { HandStats } from '../../types';

const STATE_COLORS: Record<string, string> = {
  inactive: 'var(--di, #6b7280)',
  active: 'var(--gr, #22c55e)',
  running: 'var(--bl, #3b82f6)',
  paused: 'var(--yl, #f59e0b)',
  cooldown: 'var(--pl, #a78bfa)',
  error: 'var(--rd, #ef4444)',
};

function relTime(ts: number | null): string {
  if (!ts) return 'never';
  const diff = (Date.now() - ts * 1000) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

interface Props {
  hand: HandStats;
  handsError?: string | null;
  onRun: (name: string) => void;
  onPause: (name: string) => void;
  onActivate: (name: string) => void;
  onDeactivate: (name: string) => void;
  onDelete: (name: string) => void;
}

export default function HandStatusCard({ hand, handsError, onRun, onPause, onActivate, onDeactivate, onDelete }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [optimisticState, setOptimisticState] = useState<string | null>(null);
  const color = STATE_COLORS[hand.state] || STATE_COLORS.inactive;
  const isRunning = hand.state === 'running';
  const isActive = hand.state === 'active' || hand.state === 'running' || hand.state === 'cooldown';
  const handError = handsError && handsError.includes(hand.name) ? handsError : null;
  const displayState = optimisticState || hand.state;
  const isPending = pendingAction !== null;

  return (
    <div
      style={{
        borderRadius: 10,
        background: 'var(--s2)',
        border: `1px solid ${handError ? 'var(--rd)' : 'var(--b1)'}`,
        overflow: 'hidden',
        position: 'relative',
      }}
    >
      {isPending && (
        <div style={{
          position: 'absolute',
          inset: 0,
          background: 'rgba(0,0,0,0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10,
          borderRadius: 10,
        }}>
          <Loader2 size={16} style={{ color: 'var(--tx)', animation: 'spin 1s linear infinite' }} />
        </div>
      )}
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          width: '100%',
          padding: '9px 10px',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          fontFamily: 'inherit',
        }}
      >
        <span style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: STATE_COLORS[displayState] || STATE_COLORS.inactive,
          flexShrink: 0,
          boxShadow: displayState === 'running' || optimisticState === 'running' ? `0 0 6px ${STATE_COLORS[displayState] || STATE_COLORS.inactive}` : 'none',
          animation: optimisticState === 'running' ? 'pulse 1.5s ease-in-out infinite' : 'none',
        }} />
        <span style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 12,
            fontWeight: 500,
            color: 'var(--tx)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {hand.name}
          </div>
          <div style={{ fontSize: 10.5, color: 'var(--mu)' }}>
            {relTime(hand.last_run_ts)} · {hand.total_runs} runs
            {handError && <span style={{ color: 'var(--rd)', marginLeft: 6 }}><AlertCircle size={10} style={{ verticalAlign: 'middle' }} /> error</span>}
          </div>
        </span>
        <ChevronDown
          size={12}
          style={{ color: 'var(--mu)', transition: 'transform 0.2s', transform: expanded ? 'rotate(180deg)' : 'rotate(0)' }}
        />
      </button>

      {expanded && (
        <div style={{ padding: '0 10px 10px', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {hand.description && (
            <div style={{ fontSize: 11, color: 'var(--mu)', lineHeight: 1.45 }}>{hand.description}</div>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 10.5, color: 'var(--mu)' }}>
            <span>state: <span style={{ color: 'var(--tx)' }}>{hand.state}</span></span>
            <span>model: <span style={{ color: 'var(--tx)' }}>{hand.model_preference}</span></span>
            <span>failures: <span style={{ color: hand.consecutive_failures > 0 ? 'var(--rd)' : 'var(--tx)' }}>{hand.consecutive_failures}</span></span>
            {hand.trigger_on_drive && <span>drive: <span style={{ color: 'var(--tx)' }}>{hand.trigger_on_drive}</span></span>}
          </div>
          {hand.last_error && (
            <div style={{ fontSize: 10.5, color: 'var(--rd)', fontFamily: 'monospace' }}>
              {String(hand.last_error).slice(0, 140)}
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            <button
              onClick={() => { setPendingAction('run'); setOptimisticState('running'); onRun(hand.name); setTimeout(() => { setPendingAction(null); setOptimisticState(null); }, 3000); }}
              disabled={isRunning || isPending}
              style={{ ...btnStyle('var(--pl)'), opacity: (isRunning || isPending) ? 0.6 : 1, cursor: (isRunning || isPending) ? 'not-allowed' : 'pointer' }}
            >
              {pendingAction === 'run' ? <Loader2 size={10} style={{ animation: 'spin 1s linear infinite' }} /> : <Play size={10} />} Run
            </button>
            {isActive ? (
              <>
                <button
                  onClick={() => { setPendingAction('pause'); onPause(hand.name); setTimeout(() => setPendingAction(null), 3000); }}
                  disabled={isPending}
                  style={{ ...btnStyle('var(--yl, #f59e0b)'), opacity: isPending ? 0.6 : 1, cursor: isPending ? 'not-allowed' : 'pointer' }}
                >
                  <Pause size={10} /> Pause
                </button>
                <button
                  onClick={() => { setPendingAction('deactivate'); onDeactivate(hand.name); setTimeout(() => setPendingAction(null), 3000); }}
                  disabled={isPending}
                  style={{ ...btnStyle('var(--mu)'), opacity: isPending ? 0.6 : 1, cursor: isPending ? 'not-allowed' : 'pointer' }}
                >
                  <PowerOff size={10} /> Stop
                </button>
              </>
            ) : (
              <button
                onClick={() => { setPendingAction('activate'); onActivate(hand.name); setTimeout(() => setPendingAction(null), 3000); }}
                disabled={isPending}
                style={{ ...btnStyle('var(--gr, #22c55e)'), opacity: isPending ? 0.6 : 1, cursor: isPending ? 'not-allowed' : 'pointer' }}
              >
                <Power size={10} /> Activate
              </button>
            )}
            {hand.is_custom && (
              <button
                onClick={() => { setPendingAction('delete'); onDelete(hand.name); setTimeout(() => setPendingAction(null), 3000); }}
                disabled={isPending}
                style={{ ...btnStyle('var(--rd)'), opacity: isPending ? 0.6 : 1, cursor: isPending ? 'not-allowed' : 'pointer' }}
              >
                <Trash2 size={10} /> Delete
              </button>
            )}
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
      `}</style>
    </div>
  );
}

function btnStyle(color: string): React.CSSProperties {
  return {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: 'transparent',
    border: '1px solid var(--b1)',
    borderRadius: 5,
    color,
    padding: '4px 7px',
    cursor: 'pointer',
    fontSize: 10.5,
    fontFamily: 'inherit',
  };
}
