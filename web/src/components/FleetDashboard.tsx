import { useState, useMemo } from 'react';
import type { FleetTask } from '../types';
import { XMarkIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface FleetDashboardProps {
  goal: string;
  tasks: FleetTask[];
  totalElapsed: number;
  onClose?: () => void;
}

const STATUS_CONFIG = {
  pending: { icon: '\u23f3', label: 'Pending', color: 'rgba(255,255,255,0.3)', glow: false },
  running: { icon: '\ud83d\udd04', label: 'Running', color: '#60a5fa', glow: true },
  done: { icon: '\u2705', label: 'Done', color: '#34d399', glow: false },
  failed: { icon: '\u274c', label: 'Failed', color: '#f87171', glow: false },
} as const;

function formatElapsed(seconds: number): string {
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function TaskRow({ task }: { task: FleetTask }) {
  const [expanded, setExpanded] = useState(false);
  const config = STATUS_CONFIG[task.status];
  const hasDetail = Boolean(task.result || task.error);

  return (
    <div
      style={{
        background: task.status === 'running'
          ? 'rgba(96, 165, 250, 0.06)'
          : 'rgba(255,255,255,0.02)',
        border: '1px solid',
        borderColor: task.status === 'running'
          ? 'rgba(96, 165, 250, 0.2)'
          : 'rgba(255,255,255,0.06)',
        borderRadius: 8,
        marginBottom: 6,
        transition: 'all 0.3s ease',
        ...(task.status === 'running' ? {
          boxShadow: '0 0 12px rgba(96, 165, 250, 0.1)',
          animation: 'fleetPulse 2s ease-in-out infinite',
        } : {}),
      }}
    >
      <button
        onClick={() => hasDetail && setExpanded(prev => !prev)}
        disabled={!hasDetail}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          width: '100%',
          padding: '10px 14px',
          background: 'none',
          border: 'none',
          color: 'inherit',
          cursor: hasDetail ? 'pointer' : 'default',
          textAlign: 'left',
          fontSize: 13,
        }}
      >
        {/* Status icon */}
        <span
          style={{
            fontSize: 16,
            flexShrink: 0,
            width: 24,
            textAlign: 'center',
            ...(task.status === 'running' ? { animation: 'fleetSpin 1s linear infinite' } : {}),
          }}
        >
          {config.icon}
        </span>

        {/* Description */}
        <span
          style={{
            flex: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            color: task.status === 'pending'
              ? 'rgba(255,255,255,0.4)'
              : 'rgba(255,255,255,0.85)',
          }}
          title={task.description}
        >
          {task.description}
        </span>

        {/* Elapsed */}
        <span
          style={{
            flexShrink: 0,
            fontSize: 11,
            color: config.color,
            fontFamily: 'monospace',
            minWidth: 40,
            textAlign: 'right',
          }}
        >
          {task.elapsed > 0 ? formatElapsed(task.elapsed) : '\u2014'}
        </span>

        {/* Expand arrow */}
        {hasDetail && (
          expanded
            ? <ChevronUpIcon style={{ width: 14, height: 14, flexShrink: 0, color: 'rgba(255,255,255,0.4)' }} />
            : <ChevronDownIcon style={{ width: 14, height: 14, flexShrink: 0, color: 'rgba(255,255,255,0.4)' }} />
        )}
      </button>

      {/* Expanded detail */}
      {expanded && hasDetail && (
        <div
          style={{
            padding: '0 14px 10px 48px',
            fontSize: 12,
            lineHeight: 1.5,
            color: task.error ? '#fca5a5' : 'rgba(255,255,255,0.6)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            animation: 'fleetFadeIn 0.2s ease',
          }}
        >
          {task.error || task.result}
        </div>
      )}
    </div>
  );
}

export function FleetDashboard({ goal, tasks, totalElapsed, onClose }: FleetDashboardProps) {
  const { doneCount, failedCount, completedCount, progress } = useMemo(() => {
    const done = tasks.filter(t => t.status === 'done').length;
    const failed = tasks.filter(t => t.status === 'failed').length;
    const completed = done + failed;
    const pct = tasks.length > 0 ? (completed / tasks.length) * 100 : 0;
    return { doneCount: done, failedCount: failed, completedCount: completed, progress: pct };
  }, [tasks]);

  const isComplete = completedCount === tasks.length && tasks.length > 0;

  return (
    <div
      style={{
        background: 'rgba(10, 10, 18, 0.75)',
        backdropFilter: 'blur(16px)',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: 14,
        margin: '12px 16px',
        overflow: 'hidden',
        animation: 'fleetSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '14px 16px 10px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
          <span style={{ fontSize: 18 }}>{'\ud83d\ude80'}</span>
          <span
            style={{
              fontSize: 14,
              fontWeight: 600,
              color: 'rgba(255,255,255,0.9)',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            Fleet: {goal}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
          {/* Progress counter */}
          <span
            style={{
              fontSize: 12,
              fontFamily: 'monospace',
              color: isComplete ? '#34d399' : 'rgba(255,255,255,0.5)',
              background: isComplete ? 'rgba(52, 211, 153, 0.1)' : 'rgba(255,255,255,0.05)',
              padding: '3px 8px',
              borderRadius: 6,
              border: `1px solid ${isComplete ? 'rgba(52, 211, 153, 0.2)' : 'rgba(255,255,255,0.08)'}`,
            }}
          >
            {completedCount}/{tasks.length}
          </span>

          {/* Total elapsed */}
          <span
            style={{
              fontSize: 12,
              fontFamily: 'monospace',
              color: 'rgba(255,255,255,0.4)',
            }}
          >
            {formatElapsed(totalElapsed)}
          </span>

          {/* Close button */}
          {onClose && (
            <button
              onClick={onClose}
              aria-label="Close fleet dashboard"
              style={{
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 6,
                padding: 4,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'rgba(255,255,255,0.5)',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.12)';
                (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.8)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.06)';
                (e.currentTarget as HTMLElement).style.color = 'rgba(255,255,255,0.5)';
              }}
            >
              <XMarkIcon style={{ width: 14, height: 14 }} />
            </button>
          )}
        </div>
      </div>

      {/* Task list */}
      <div style={{ padding: '10px 12px 6px' }}>
        {tasks.map(task => (
          <TaskRow key={task.id} task={task} />
        ))}
      </div>

      {/* Progress bar */}
      <div style={{ padding: '0 16px 14px' }}>
        <div
          style={{
            height: 4,
            borderRadius: 2,
            background: 'rgba(255,255,255,0.06)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              borderRadius: 2,
              width: `${progress}%`,
              background: failedCount > 0
                ? 'linear-gradient(90deg, #34d399, #f87171)'
                : 'linear-gradient(90deg, #60a5fa, #34d399)',
              transition: 'width 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          />
        </div>
        {/* Status summary */}
        {(doneCount > 0 || failedCount > 0) && (
          <div
            style={{
              display: 'flex',
              gap: 12,
              marginTop: 6,
              fontSize: 11,
              color: 'rgba(255,255,255,0.35)',
            }}
          >
            {doneCount > 0 && <span style={{ color: '#34d399' }}>{doneCount} done</span>}
            {failedCount > 0 && <span style={{ color: '#f87171' }}>{failedCount} failed</span>}
            {tasks.length - completedCount > 0 && (
              <span>{tasks.length - completedCount} remaining</span>
            )}
          </div>
        )}
      </div>

    </div>
  );
}
