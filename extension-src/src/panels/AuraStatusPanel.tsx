/**
 * AuraStatusPanel — Compact consciousness dashboard for the extension sidebar.
 *
 * Shows drives, emotions, proactive insights, and dream status in a
 * sidebar-friendly compact layout. Polls /api endpoints for live data.
 */

import React, { useState, useCallback, useEffect } from 'react';
import { Activity, Brain, Eye, Heart, Lightbulb, Moon, Sun, Sparkles, Zap, RefreshCw } from 'lucide-react';
import { API_KEY, HTTP } from '../api';

interface DriveState { [key: string]: number }
interface CuriosityTarget { label: string; gap_type: string; question?: string }
interface DreamInsight { type: string; confidence: number; content: string }

const DRIVE_CONFIG: { key: string; label: string; color: string; icon: React.ReactNode }[] = [
  { key: 'curiosity', label: 'Curiosity', color: '#a78bfa', icon: <Eye size={12} /> },
  { key: 'competence', label: 'Competence', color: '#34d399', icon: <Zap size={12} /> },
  { key: 'social', label: 'Social', color: '#f472b6', icon: <Heart size={12} /> },
  { key: 'coherence', label: 'Coherence', color: '#60a5fa', icon: <Brain size={12} /> },
];

export default function AuraStatusPanel() {
  const [drives, setDrives] = useState<DriveState | null>(null);
  const [emotion, setEmotion] = useState<{ dominant: string; intensity: number; pad: { pleasure: number; arousal: number; dominance: number } } | null>(null);
  const [curiosityTargets, setCuriosityTargets] = useState<CuriosityTarget[]>([]);
  const [dreamStatus, setDreamStatus] = useState<{ is_sleeping: boolean; total_insights: number; total_sessions: number; insights: DreamInsight[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState(0);

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (API_KEY) headers['X-API-Key'] = API_KEY;

  const fetchAll = useCallback(async () => {
    try {
      const results = await Promise.allSettled([
        fetch(`${HTTP}/api/motivation/drives`, { headers }).then(r => r.ok ? r.json() : null),
        fetch(`${HTTP}/api/alma/state`, { headers }).then(r => r.ok ? r.json() : null),
        fetch(`${HTTP}/api/proactive/curiosity`, { headers }).then(r => r.ok ? r.json() : null),
        fetch(`${HTTP}/api/neurodream`, { headers }).then(r => r.ok ? r.json() : null),
      ]);

      if (results[0].status === 'fulfilled' && results[0].value) {
        setDrives(results[0].value.drives || results[0].value);
      }
      if (results[1].status === 'fulfilled' && results[1].value) {
        const a = results[1].value;
        setEmotion({
          dominant: a.dominant_emotion || 'neutral',
          intensity: a.intensity || 0,
          pad: a.pad || { pleasure: 0, arousal: 0, dominance: 0 },
        });
      }
      if (results[2].status === 'fulfilled' && results[2].value) {
        setCuriosityTargets((results[2].value.targets || []).slice(0, 3));
      }
      if (results[3].status === 'fulfilled' && results[3].value) {
        setDreamStatus(results[3].value);
      }
    } catch { /* silent */ }
    setLoading(false);
    setLastRefresh(Date.now());
  }, []);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  if (loading && !drives && !emotion) {
    return (
      <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
        {[1, 2, 3].map(i => (
          <div key={i} style={{ height: 60, borderRadius: 10, background: 'var(--s2)', animation: 'pulse 1.5s infinite' }} />
        ))}
      </div>
    );
  }

  return (
    <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Brain size={14} style={{ color: 'var(--p)' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)' }}>Aura's Mind</span>
        </div>
        <button
          onClick={fetchAll}
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <RefreshCw size={12} />
        </button>
      </div>

      {/* Drives */}
      <div style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
        <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 8 }}>
          Intrinsic Drives
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          {DRIVE_CONFIG.map(d => {
            const val = drives?.[d.key] ?? 0;
            const pct = Math.round(val * 100);
            return (
              <div key={d.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ color: d.color, flexShrink: 0 }}>{d.icon}</span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                    <span style={{ fontSize: 9, color: 'var(--mu)' }}>{d.label}</span>
                    <span style={{ fontSize: 9, fontWeight: 600, color: d.color }}>{pct}%</span>
                  </div>
                  <div style={{ height: 3, borderRadius: 2, background: 'var(--b1)' }}>
                    <div style={{ height: '100%', borderRadius: 2, width: `${pct}%`, background: d.color, transition: 'width 0.8s ease' }} />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Emotional State */}
      {emotion && (
        <div style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <span style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)' }}>
              Emotion
            </span>
            <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--p)', textTransform: 'capitalize' }}>
              {emotion.dominant} {Math.round(emotion.intensity * 100)}%
            </span>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {[
              { label: 'P', value: emotion.pad.pleasure, color: '#10b981' },
              { label: 'A', value: emotion.pad.arousal, color: '#f59e0b' },
              { label: 'D', value: emotion.pad.dominance, color: '#3b82f6' },
            ].map(p => (
              <div key={p.label} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ fontSize: 8, color: 'var(--di)', width: 8 }}>{p.label}</span>
                <div style={{ flex: 1, height: 3, borderRadius: 2, background: 'var(--b1)', position: 'relative' }}>
                  <div style={{ position: 'absolute', left: '50%', top: 0, width: 1, height: 3, background: 'var(--di)' }} />
                  <div
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: `${((p.value + 1) / 2) * 100}%`,
                      transform: 'translate(-50%, -50%)',
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background: p.color,
                      boxShadow: `0 0 4px ${p.color}60`,
                      transition: 'left 0.5s ease',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Curiosity Targets */}
      {curiosityTargets.length > 0 && (
        <div style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 6 }}>
            <Lightbulb size={10} style={{ color: '#a78bfa' }} />
            <span style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)' }}>
              Curiosity ({curiosityTargets.length})
            </span>
          </div>
          {curiosityTargets.map((t, i) => (
            <div key={i} style={{ fontSize: 10, color: 'var(--tx)', marginBottom: 4, lineHeight: 1.4, borderLeft: '2px solid #a78bfa30', paddingLeft: 8 }}>
              {t.question || `"${t.label}" — ${t.gap_type?.replace(/_/g, ' ')}`}
            </div>
          ))}
        </div>
      )}

      {/* Dream Status */}
      {dreamStatus && (
        <div style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {dreamStatus.is_sleeping
                ? <Moon size={10} style={{ color: '#818cf8' }} />
                : <Sun size={10} style={{ color: '#fbbf24' }} />
              }
              <span style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)' }}>
                {dreamStatus.is_sleeping ? 'Dreaming' : 'Awake'}
              </span>
            </div>
            <span style={{ fontSize: 9, color: 'var(--di)' }}>
              {dreamStatus.total_sessions} sessions · {dreamStatus.total_insights} insights
            </span>
          </div>
          {dreamStatus.insights?.slice(0, 2).map((ins, i) => (
            <div key={i} style={{ fontSize: 10, color: 'var(--mu)', marginBottom: 3, lineHeight: 1.4, display: 'flex', gap: 4, alignItems: 'flex-start' }}>
              <Sparkles size={9} style={{ color: '#a78bfa', flexShrink: 0, marginTop: 2 }} />
              <span>{ins.content?.slice(0, 100)}{(ins.content?.length || 0) > 100 ? '...' : ''}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
