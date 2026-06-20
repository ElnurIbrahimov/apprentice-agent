/**
 * ConsciousnessPanel — Aura's mind at a glance.
 *
 * 4 sections: Drive Gauges, Emotional State, Theory of Mind, Cognitive Load.
 * All data from existing API endpoints, polled every 10s.
 */

import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { DriveState, CognitiveLoad, ToMSummary } from '../types';
import { EMOTION_COLORS } from '../utils/emotionConstants';
import { apiFetch } from '../utils/apiFetch';

// ─── Drive config ───
const DRIVES: { key: keyof DriveState; label: string; color: string; icon: string }[] = [
  { key: 'curiosity', label: 'Curiosity', color: '#a78bfa', icon: '\uD83D\uDD2D' },
  { key: 'competence', label: 'Competence', color: '#34d399', icon: '\uD83C\uDFAF' },
  { key: 'social', label: 'Social', color: '#f472b6', icon: '\uD83E\uDD1D' },
  { key: 'coherence', label: 'Coherence', color: '#60a5fa', icon: '\uD83E\uDDE9' },
];

// ─── SVG Radial Gauge ───
function DriveGauge({ value, label, color, icon }: { value: number; label: string; color: string; icon: string }) {
  const pct = Math.max(0, Math.min(1, value));
  const radius = 36;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - pct);

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative w-20 h-20 sm:w-24 sm:h-24">
        <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
          {/* Track */}
          <circle cx="40" cy="40" r={radius} fill="none" stroke="var(--border-default)" strokeWidth="5" />
          {/* Fill */}
          <circle
            cx="40" cy="40" r={radius} fill="none"
            stroke={color} strokeWidth="5" strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke-dashoffset 1s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
          />
        </svg>
        {/* Center label */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-lg">{icon}</span>
          <span className="text-xs font-bold" style={{ color }}>{Math.round(pct * 100)}%</span>
        </div>
      </div>
      <span className="text-[10px] sm:text-xs font-semibold text-chat-text-secondary">{label}</span>
    </div>
  );
}

// ─── PAD Bar (pleasure/arousal/dominance) ───
function PADBar({ label, value, color }: { label: string; value: number; color: string }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const pct = ((clamped + 1) / 2) * 100; // 0-100

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] sm:text-xs font-medium w-8 text-right" style={{ color: 'var(--text-secondary)' }}>
        {label[0]}
      </span>
      <div className="flex-1 h-2 rounded-full relative" style={{ background: 'var(--border-default)' }}>
        {/* Center marker */}
        <div className="absolute left-1/2 top-0 w-0.5 h-2 -translate-x-1/2" style={{ background: 'var(--text-tertiary)' }} />
        {/* Value dot */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 rounded-full shadow-sm"
          style={{
            left: `${pct}%`,
            transform: `translate(-50%, -50%)`,
            background: color,
            boxShadow: `0 0 8px ${color}60`,
            transition: 'left 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        />
      </div>
      <span className="text-[10px] font-mono w-10 text-right" style={{ color: 'var(--text-secondary)' }}>
        {clamped >= 0 ? '+' : ''}{(clamped * 100).toFixed(0)}
      </span>
    </div>
  );
}

// ─── Topic Pill ───
function TopicPill({ topic, level }: { topic: string; level: string }) {
  const levelColors: Record<string, string> = {
    expert: '#34d399',
    advanced: '#60a5fa',
    intermediate: '#fbbf24',
    beginner: '#f87171',
    novice: '#a1a1aa',
  };
  const c = levelColors[level?.toLowerCase()] || '#a1a1aa';
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium"
      style={{ background: `${c}20`, color: c, border: `1px solid ${c}30` }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
      {topic}
    </span>
  );
}

// ─── Cognitive Breathing Circle ───
function BreathingCircle({ breathRate, glowIntensity }: { breathRate: number; glowIntensity: number }) {
  const duration = breathRate > 0 ? Math.max(1.5, 6 / breathRate) : 3.5;
  const glow = Math.max(0.1, Math.min(1, glowIntensity));

  return (
    <div className="flex items-center justify-center">
      <div
        className="w-16 h-16 rounded-full relative"
        style={{
          background: `radial-gradient(circle, var(--chat-accent) 0%, transparent 70%)`,
          opacity: 0.3 + glow * 0.5,
          animation: `breathe-glow ${duration}s ease-in-out infinite`,
        }}
      >
        <div
          className="absolute inset-2 rounded-full"
          style={{
            background: 'linear-gradient(135deg, var(--chat-accent), #6366f1)',
            animation: `breathe-core ${duration}s ease-in-out infinite`,
          }}
        />
      </div>
    </div>
  );
}

// ─── Main Component ───
export function ConsciousnessPanel() {
  const [drives, setDrives] = useState<DriveState | null>(null);
  const [alma, setAlma] = useState<any>(null);
  const [tom, setTom] = useState<ToMSummary | null>(null);
  const [cogLoad, setCogLoad] = useState<CognitiveLoad | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    const results = await Promise.allSettled([
      apiFetch('/api/motivation/drives').then(r => r.ok ? r.json() : null),
      apiFetch('/api/alma/state').then(r => r.ok ? r.json() : null),
      apiFetch('/api/theory-of-mind/model').then(r => r.ok ? r.json() : null),
      apiFetch('/api/idle-presence/cognitive-load').then(r => r.ok ? r.json() : null),
    ]);
    if (results[0].status === 'fulfilled' && results[0].value) setDrives(results[0].value.drives || results[0].value);
    if (results[1].status === 'fulfilled' && results[1].value) setAlma(results[1].value);
    if (results[2].status === 'fulfilled' && results[2].value) setTom(results[2].value);
    if (results[3].status === 'fulfilled' && results[3].value) setCogLoad(results[3].value);
    setLoading(false);
  }, []);

  usePolling(fetchAll, 10000);

  if (loading && !drives && !alma) {
    return (
      <div className="p-4 space-y-4 animate-pulse">
        <div className="h-24 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        <div className="h-20 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        <div className="h-16 rounded-xl" style={{ background: 'var(--surface-2)' }} />
      </div>
    );
  }

  const emotionColor = alma?.dominant_emotion ? (EMOTION_COLORS[alma.dominant_emotion] || '#a1a1aa') : '#a1a1aa';
  const topics = tom?.top_topics || (tom as any)?.topics
    ? Object.entries((tom as any)?.topics || {}).slice(0, 8).map(([topic, level]) => ({ topic, level: String(level) }))
    : [];

  return (
    <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
      {/* Section 1: Drive Gauges */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
          Intrinsic Drives
        </h3>
        <div className="grid grid-cols-4 gap-2">
          {DRIVES.map(d => (
            <DriveGauge
              key={d.key}
              value={drives?.[d.key] ?? 0}
              label={d.label}
              color={d.color}
              icon={d.icon}
            />
          ))}
        </div>
      </section>

      {/* Section 2: Emotional State */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider" style={{ color: 'var(--text-secondary)' }}>
            Emotional State
          </h3>
          {alma?.dominant_emotion && (
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: emotionColor, boxShadow: `0 0 6px ${emotionColor}` }} />
              <span className="text-xs font-medium capitalize" style={{ color: emotionColor }}>
                {alma.dominant_emotion}
              </span>
              {alma?.intensity != null && (
                <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                  {Math.round(alma.intensity * 100)}%
                </span>
              )}
            </div>
          )}
        </div>
        <div className="space-y-2">
          <PADBar label="Pleasure" value={alma?.pad?.pleasure ?? 0} color="#10b981" />
          <PADBar label="Arousal" value={alma?.pad?.arousal ?? 0} color="#f59e0b" />
          <PADBar label="Dominance" value={alma?.pad?.dominance ?? 0} color="#3b82f6" />
        </div>
      </section>

      {/* Section 3: Theory of Mind */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
          Theory of Mind
        </h3>
        {tom ? (
          <div className="space-y-3">
            {/* Style summary */}
            {tom.style_guidance && (
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {tom.style_guidance.slice(0, 200)}{tom.style_guidance.length > 200 ? '...' : ''}
              </p>
            )}
            {/* Topic pills */}
            {topics.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {topics.map(t => <TopicPill key={t.topic} topic={t.topic} level={t.level} />)}
              </div>
            )}
            {topics.length === 0 && !tom.style_guidance && (
              <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                Keep chatting — I'm learning about you.
              </p>
            )}
          </div>
        ) : (
          <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>Theory of Mind initializing...</p>
        )}
      </section>

      {/* Section 4: Cognitive Load */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <h3 className="text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
          Cognitive State
        </h3>
        <div className="flex items-center gap-4">
          <BreathingCircle
            breathRate={cogLoad?.breath_rate ?? 1}
            glowIntensity={cogLoad?.glow_intensity ?? 0.3}
          />
          <div className="flex-1 space-y-1.5">
            {cogLoad?.breakdown ? (
              Object.entries(cogLoad.breakdown).slice(0, 4).map(([key, val]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-[10px] w-16 capitalize" style={{ color: 'var(--text-secondary)' }}>{key}</span>
                  <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--border-default)' }}>
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, (val as number) * 100)}%`,
                        background: 'var(--chat-accent)',
                        transition: 'width 0.8s ease',
                      }}
                    />
                  </div>
                </div>
              ))
            ) : (
              <p className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
                {cogLoad ? `Breath: ${cogLoad.breath_rate?.toFixed(1)}/s` : 'Monitoring...'}
              </p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
