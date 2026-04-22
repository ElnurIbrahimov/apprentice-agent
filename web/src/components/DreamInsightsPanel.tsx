/**
 * DreamInsightsPanel — What Aura discovers while sleeping.
 *
 * Premium visualization of dream journal, insights with confidence bars,
 * sleep phase tracking, and learned context. Makes Aura's inner life visible.
 */

import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { haptics } from '../utils/haptics';
import {
  MoonIcon,
  SunIcon,
  SparklesIcon,
  LightBulbIcon,
  LinkIcon,
  BeakerIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface DreamEntry {
  phase: string;
  timestamp: string;
  content: string;
}

interface DreamInsight {
  type: string;
  confidence: number;
  content: string;
}

interface LearnedContext {
  version?: string;
  generated_at?: string;
  user_summary?: string;
  key_facts_count?: number;
  preferences_count?: number;
  ongoing_topics?: string[];
  conversations_processed?: number;
}

interface NeuroDreamStatus {
  enabled: boolean;
  is_sleeping: boolean;
  current_phase: string | null;
  total_sessions: number;
  total_insights: number;
  dream_journal: DreamEntry[];
  insights: DreamInsight[];
  learned_context: LearnedContext | null;
}

const PHASE_CONFIG: Record<string, { icon: string; color: string; label: string }> = {
  light: { icon: '\uD83C\uDF19', color: '#818cf8', label: 'Light Sleep' },
  deep: { icon: '\uD83D\uDCA4', color: '#6366f1', label: 'Deep Sleep' },
  rem: { icon: '\uD83C\uDF08', color: '#a78bfa', label: 'REM' },
  waking: { icon: '\u2600\uFE0F', color: '#fbbf24', label: 'Waking' },
};

const INSIGHT_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  connection: LinkIcon,
  pattern: ChartBarIcon,
  hypothesis: BeakerIcon,
  prediction: SparklesIcon,
};

const INSIGHT_COLORS: Record<string, string> = {
  connection: '#60a5fa',
  pattern: '#34d399',
  hypothesis: '#f59e0b',
  prediction: '#a78bfa',
};

function formatTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    if (d.toDateString() === now.toDateString()) return 'Today';
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (d.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch { return ''; }
}

export function DreamInsightsPanel() {
  const [status, setStatus] = useState<NeuroDreamStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await apiFetch('/api/neurodream');
      if (res.ok) setStatus(await res.json());
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  usePolling(fetchStatus, 15000);

  const triggerSleep = useCallback(async () => {
    haptics.medium();
    setActionLoading(true);
    try {
      await apiFetch('/api/neurodream/sleep', { method: 'POST' });
      setTimeout(fetchStatus, 2000);
    } catch { /* silent */ }
    setActionLoading(false);
  }, [fetchStatus]);

  const triggerWake = useCallback(async () => {
    haptics.light();
    setActionLoading(true);
    try {
      await apiFetch('/api/neurodream/wake', { method: 'POST' });
      setTimeout(fetchStatus, 2000);
    } catch { /* silent */ }
    setActionLoading(false);
  }, [fetchStatus]);

  if (loading && !status) {
    return (
      <div className="p-4 space-y-3 animate-pulse">
        {[1, 2, 3].map(i => (
          <div key={i} className="h-20 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        ))}
      </div>
    );
  }

  if (!status?.enabled) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-12">
        <MoonIcon className="w-8 h-8 mb-2" style={{ color: 'var(--text-tertiary)' }} />
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>Dream system initializing...</p>
      </div>
    );
  }

  const phase = status.current_phase ? PHASE_CONFIG[status.current_phase] || PHASE_CONFIG.light : null;

  return (
    <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
      {/* Sleep Status + Controls */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {status.is_sleeping ? (
              <>
                <span className="text-lg">{phase?.icon || '\uD83D\uDCA4'}</span>
                <div>
                  <span className="text-sm font-semibold" style={{ color: phase?.color || '#818cf8' }}>
                    {phase?.label || 'Sleeping'}
                  </span>
                  <span className="text-[10px] ml-2 animate-pulse" style={{ color: 'var(--text-tertiary)' }}>dreaming...</span>
                </div>
              </>
            ) : (
              <>
                <span className="text-lg">{'\u2600\uFE0F'}</span>
                <span className="text-sm font-semibold text-chat-text">Awake</span>
              </>
            )}
          </div>
          <div className="flex gap-2">
            {status.is_sleeping ? (
              <button
                onClick={triggerWake}
                disabled={actionLoading}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95 disabled:opacity-50"
                style={{ background: 'rgba(251, 191, 36, 0.15)', color: '#fbbf24' }}
              >
                <SunIcon className="w-3.5 h-3.5" /> Wake
              </button>
            ) : (
              <button
                onClick={triggerSleep}
                disabled={actionLoading}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95 disabled:opacity-50"
                style={{ background: 'rgba(129, 140, 248, 0.15)', color: '#818cf8' }}
              >
                <MoonIcon className="w-3.5 h-3.5" /> Sleep
              </button>
            )}
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-lg p-2.5 text-center" style={{ background: 'var(--surface-2)' }}>
            <div className="text-lg font-bold text-chat-text">{status.total_sessions}</div>
            <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>Sleep Sessions</div>
          </div>
          <div className="rounded-lg p-2.5 text-center" style={{ background: 'var(--surface-2)' }}>
            <div className="text-lg font-bold text-chat-text">{status.total_insights}</div>
            <div className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>Insights Found</div>
          </div>
        </div>
      </section>

      {/* Dream Insights — the premium section */}
      {status.insights && status.insights.length > 0 && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
            <LightBulbIcon className="w-3.5 h-3.5" />
            Dream Insights
          </h3>
          <div className="space-y-2.5">
            {status.insights.map((insight, i) => {
              const Icon = INSIGHT_ICONS[insight.type] || SparklesIcon;
              const color = INSIGHT_COLORS[insight.type] || '#a1a1aa';
              const conf = typeof insight.confidence === 'number'
                ? (insight.confidence > 1 ? insight.confidence : insight.confidence * 100)
                : 0;
              return (
                <div
                  key={i}
                  className="rounded-xl p-3"
                  style={{
                    background: `${color}08`,
                    border: `1px solid ${color}20`,
                    animation: `spring-up 0.3s ease ${i * 60}ms both`,
                  }}
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <span style={{ color }}><Icon className="w-3.5 h-3.5" /></span>
                    <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color }}>
                      {insight.type}
                    </span>
                    <div className="flex-1" />
                    <div className="flex items-center gap-1">
                      <div className="w-12 h-1.5 rounded-full" style={{ background: `${color}20` }}>
                        <div className="h-full rounded-full" style={{ width: `${conf}%`, background: color, transition: 'width 0.5s ease' }} />
                      </div>
                      <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>{conf.toFixed(0)}%</span>
                    </div>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                    {insight.content}
                  </p>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Dream Journal */}
      {status.dream_journal && status.dream_journal.length > 0 && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
            <MoonIcon className="w-3.5 h-3.5" />
            Dream Journal
          </h3>
          <div className="space-y-2">
            {status.dream_journal.map((entry, i) => {
              const p = PHASE_CONFIG[entry.phase] || PHASE_CONFIG.light;
              return (
                <div key={i} className="flex gap-2.5 items-start">
                  <div className="flex flex-col items-center flex-shrink-0 mt-0.5">
                    <span className="text-sm">{p.icon}</span>
                    {i < status.dream_journal.length - 1 && (
                      <div className="w-px flex-1 mt-1" style={{ background: 'var(--border-default)', minHeight: 16 }} />
                    )}
                  </div>
                  <div className="flex-1 min-w-0 pb-2">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-semibold" style={{ color: p.color }}>{p.label}</span>
                      <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                        {formatDate(entry.timestamp)} {formatTime(entry.timestamp)}
                      </span>
                    </div>
                    <p className="text-xs mt-0.5 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      {entry.content}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Learned Context */}
      {status.learned_context && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)' }}>
            <SparklesIcon className="w-3.5 h-3.5" />
            Learned Context
          </h3>
          {status.learned_context.user_summary && (
            <p className="text-xs leading-relaxed mb-2" style={{ color: 'var(--text-primary)' }}>
              {status.learned_context.user_summary}
            </p>
          )}
          <div className="flex flex-wrap gap-2 text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
            {status.learned_context.key_facts_count != null && (
              <span>{status.learned_context.key_facts_count} facts</span>
            )}
            {status.learned_context.preferences_count != null && (
              <span>{status.learned_context.preferences_count} preferences</span>
            )}
            {status.learned_context.conversations_processed != null && (
              <span>{status.learned_context.conversations_processed} convos processed</span>
            )}
          </div>
          {status.learned_context.ongoing_topics && status.learned_context.ongoing_topics.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mt-2">
              {status.learned_context.ongoing_topics.map(topic => (
                <span key={topic} className="px-2 py-0.5 rounded-full text-[10px]" style={{ background: 'var(--surface-3)', color: 'var(--text-secondary)' }}>
                  {topic}
                </span>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
