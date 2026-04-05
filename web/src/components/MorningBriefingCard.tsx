/**
 * MorningBriefingCard — Aura's daily digest.
 *
 * Shows the latest morning briefing (project status, drive states, research
 * findings, dream insights) and lets you trigger a new one.
 */

import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { haptics } from '../utils/haptics';
import ReactMarkdown from 'react-markdown';
import {
  SunIcon,
  ArrowPathIcon,
  PlayIcon,
  ClockIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

interface HandStatus {
  name: string;
  description: string;
  state: string;
  total_runs: number;
  total_cost: number;
  last_run: string | null;
  last_summary?: string;
}

interface HandHistoryEntry {
  hand: string;
  summary: string;
  timestamp: string;
  tokens_used?: number;
  cost_usd?: number;
  success?: boolean;
}

function formatRelTime(iso: string | null): string {
  if (!iso) return 'Never';
  try {
    const d = new Date(iso);
    const diff = Date.now() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
  } catch { return 'Unknown'; }
}

export function MorningBriefingCard() {
  const [handStatus, setHandStatus] = useState<HandStatus | null>(null);
  const [latestBriefing, setLatestBriefing] = useState<HandHistoryEntry | null>(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      const [statusRes, historyRes] = await Promise.all([
        fetch('/api/hands/morning_briefing'),
        fetch('/api/hands/history?limit=5'),
      ]);
      if (statusRes.ok) {
        const data = await statusRes.json();
        setHandStatus(data);
        setRunning(data.state === 'running');
      }
      if (historyRes.ok) {
        const data = await historyRes.json();
        const briefings = (data.history || []).filter((h: HandHistoryEntry) => h.hand === 'morning_briefing');
        if (briefings.length > 0) setLatestBriefing(briefings[0]);
      }
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  usePolling(fetchData, 30000);

  const triggerRun = useCallback(async () => {
    haptics.medium();
    setRunning(true);
    try {
      await fetch('/api/hands/morning_briefing/run', { method: 'POST' });
      // Poll for completion
      setTimeout(() => fetchData(), 3000);
      setTimeout(() => fetchData(), 8000);
      setTimeout(() => fetchData(), 15000);
    } catch { /* silent */ }
  }, [fetchData]);

  if (loading && !handStatus && !latestBriefing) {
    return (
      <div className="rounded-xl p-4 animate-pulse" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <div className="h-5 rounded w-32 mb-3" style={{ background: 'var(--surface-3)' }} />
        <div className="space-y-2">
          <div className="h-3 rounded w-full" style={{ background: 'var(--surface-2)' }} />
          <div className="h-3 rounded w-3/4" style={{ background: 'var(--surface-2)' }} />
        </div>
      </div>
    );
  }

  const briefingContent = latestBriefing?.summary || handStatus?.last_summary;
  const isStale = handStatus?.last_run ? (Date.now() - new Date(handStatus.last_run).getTime()) > 24 * 60 * 60 * 1000 : true;

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderLeft: '3px solid #f59e0b',
      }}
    >
      {/* Header */}
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left transition-colors"
        style={{ background: running ? 'rgba(245, 158, 11, 0.08)' : 'transparent' }}
      >
        <SunIcon className="w-5 h-5 flex-shrink-0" style={{ color: '#f59e0b' }} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-chat-text">Morning Briefing</span>
            {running && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full animate-pulse" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#f59e0b' }}>
                Generating...
              </span>
            )}
            {isStale && !running && (
              <span className="text-[10px] px-1.5 py-0.5 rounded-full" style={{ background: 'rgba(161, 161, 170, 0.15)', color: 'var(--text-tertiary)' }}>
                Stale
              </span>
            )}
          </div>
          {handStatus?.last_run && (
            <div className="flex items-center gap-1 mt-0.5">
              <ClockIcon className="w-3 h-3" style={{ color: 'var(--text-tertiary)' }} />
              <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                {formatRelTime(handStatus.last_run)} | {handStatus.total_runs} runs
              </span>
            </div>
          )}
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); triggerRun(); }}
          disabled={running}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95 disabled:opacity-50"
          style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}
        >
          {running ? <ArrowPathIcon className="w-3.5 h-3.5 animate-spin" /> : <PlayIcon className="w-3.5 h-3.5" />}
          {running ? 'Running' : 'Run Now'}
        </button>
      </button>

      {/* Content */}
      {expanded && briefingContent && (
        <div className="px-4 pb-4 pt-1">
          <div className="text-sm leading-relaxed prose-compact" style={{ color: 'var(--text-primary)' }}>
            <ReactMarkdown>{briefingContent}</ReactMarkdown>
          </div>
        </div>
      )}

      {expanded && !briefingContent && (
        <div className="px-4 pb-4 pt-1 flex items-center gap-2">
          <SparklesIcon className="w-4 h-4" style={{ color: 'var(--text-tertiary)' }} />
          <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
            No briefing yet. Tap "Run Now" to generate your first daily digest.
          </span>
        </div>
      )}
    </div>
  );
}
