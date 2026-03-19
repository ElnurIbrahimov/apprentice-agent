import { useState, useEffect, useRef, useCallback } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import ReactMarkdown from 'react-markdown';

const ACTION_COLORS: Record<string, string> = {
  notify: '#3b82f6',    // blue
  suggest: '#8b5cf6',   // purple
  remind: '#f59e0b',    // amber
  ask: '#14b8a6',       // teal
  intervene: '#ef4444', // red
  prepare: '#22c55e',   // green
};

const ACTION_ICONS: Record<string, string> = {
  notify: '💡',
  suggest: '✨',
  remind: '⏰',
  ask: '🤔',
  intervene: '⚡',
  prepare: '📋',
};

const AUTO_DISMISS_MS = 30000;

interface ProactiveCardProps {
  action: string;
  content: string;
  trigger?: string;
  onDismiss: () => void;
  onAction?: () => void;
}

export function ProactiveCard({ action, content, trigger, onDismiss, onAction }: ProactiveCardProps) {
  const [progress, setProgress] = useState(100);
  const [exiting, setExiting] = useState(false);
  const startTimeRef = useRef(Date.now());
  const rafRef = useRef<number>(0);

  const accentColor = ACTION_COLORS[action] || '#8b5cf6';
  const icon = ACTION_ICONS[action] || '💭';

  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  const handleDismiss = useCallback(() => {
    setExiting(true);
    dismissTimerRef.current = setTimeout(() => onDismissRef.current(), 300);
  }, []);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      if (dismissTimerRef.current) clearTimeout(dismissTimerRef.current);
    };
  }, []);

  // Auto-dismiss countdown (stable — no deps that churn)
  useEffect(() => {
    const tick = () => {
      const elapsed = Date.now() - startTimeRef.current;
      const remaining = Math.max(0, 100 - (elapsed / AUTO_DISMISS_MS) * 100);
      setProgress(remaining);
      if (remaining <= 0) {
        handleDismiss();
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [handleDismiss]);

  // Check if content has markdown formatting
  const hasMarkdown = /[*_`#\[\]|>-]/.test(content);

  return (
    <div
      className={`proactive-card ${exiting ? 'proactive-card-exit' : ''}`}
      style={{ '--accent': accentColor } as React.CSSProperties}
    >
      {/* Left accent stripe */}
      <div
        className="absolute left-0 top-0 bottom-0 w-[3px] rounded-l-xl"
        style={{ background: accentColor }}
      />

      {/* Header */}
      <div className="flex items-center gap-2 mb-2 pl-3">
        <span className="text-base">{icon}</span>
        <span className="text-xs text-white/50">
          {trigger || action}
        </span>
        <button
          onClick={handleDismiss}
          className="ml-auto p-1 rounded text-white/30 hover:text-white/70 hover:bg-white/10 transition-all"
          aria-label="Dismiss notification"
        >
          <XMarkIcon className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Body */}
      <div className="pl-3 pr-2 text-sm text-white/80 leading-relaxed">
        {hasMarkdown ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        ) : (
          <p>{content}</p>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center gap-2 mt-3 pl-3">
        <button
          onClick={handleDismiss}
          className="text-xs px-3 py-1.5 rounded-lg text-white/50 hover:text-white/80 hover:bg-white/8 transition-all"
        >
          Dismiss
        </button>
        {onAction && (
          <button
            onClick={() => {
              onAction();
              handleDismiss();
            }}
            className="text-xs px-3 py-1.5 rounded-lg font-medium transition-all"
            style={{
              background: `${accentColor}22`,
              color: accentColor,
              border: `1px solid ${accentColor}33`,
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLElement).style.background = `${accentColor}33`;
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLElement).style.background = `${accentColor}22`;
            }}
          >
            Take action
          </button>
        )}
      </div>

      {/* Auto-dismiss progress bar */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-white/5 rounded-b-xl overflow-hidden">
        <div
          className="h-full transition-none"
          style={{
            width: `${progress}%`,
            background: `linear-gradient(90deg, ${accentColor}66, ${accentColor}33)`,
          }}
        />
      </div>
    </div>
  );
}
