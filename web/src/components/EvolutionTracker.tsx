/**
 * EvolutionTracker — Watch Aura improve itself.
 *
 * Visualizes GEPA evolution runs: current status, Pareto frontier,
 * candidate genealogy, improvement metrics. Makes self-improvement visible.
 */

import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { haptics } from '../utils/haptics';
import {
  ArrowPathIcon,
  PlayIcon,
  SparklesIcon,
  ChartBarIcon,
  CheckCircleIcon,
  XCircleIcon,
  BeakerIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface Candidate {
  id: number;
  parent_id: number;
  avg_score: number;
  scores: Record<string, number>;
  components: Record<string, string>;
  created_at?: string;
}

interface EvolutionResult {
  best_candidate?: Candidate;
  all_candidates?: Candidate[];
  iterations_run: number;
  total_evals: number;
  pareto_front?: Record<string, number>;
  improvement: number;
  duration_seconds: number;
  stop_reason: string;
}

interface EvolutionStatus {
  status: 'idle' | 'starting' | 'running' | 'complete' | 'error';
  result: EvolutionResult | null;
  run_id: string | null;
}

const STATUS_CONFIG: Record<string, { color: string; label: string; icon: React.ComponentType<{ className?: string }> }> = {
  idle: { color: '#a1a1aa', label: 'Idle', icon: BeakerIcon },
  starting: { color: '#fbbf24', label: 'Starting', icon: ArrowPathIcon },
  running: { color: '#818cf8', label: 'Evolving', icon: ArrowPathIcon },
  complete: { color: '#34d399', label: 'Complete', icon: CheckCircleIcon },
  error: { color: '#f87171', label: 'Error', icon: XCircleIcon },
};

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs.toFixed(1)}s`;
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return `${m}m ${s}s`;
}

function ScoreBar({ label, score, best }: { label: string; score: number; best?: boolean }) {
  const pct = Math.min(100, score * 100);
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] w-16 truncate" style={{ color: 'var(--text-secondary)' }}>{label}</span>
      <div className="flex-1 h-2 rounded-full" style={{ background: 'var(--border-default)' }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: best ? '#34d399' : 'var(--chat-accent)',
          }}
        />
      </div>
      <span className="text-[10px] font-mono w-10 text-right" style={{ color: best ? '#34d399' : 'var(--text-secondary)' }}>
        {pct.toFixed(0)}%
      </span>
    </div>
  );
}

export function EvolutionTracker() {
  const [evoStatus, setEvoStatus] = useState<EvolutionStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [showCandidates, setShowCandidates] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await apiFetch('/api/evolution/status');
      if (res.ok) setEvoStatus(await res.json());
    } catch { /* silent */ }
    setLoading(false);
  }, []);

  usePolling(fetchStatus, evoStatus?.status === 'running' ? 5000 : 30000);

  const startRun = useCallback(async () => {
    haptics.medium();
    setStarting(true);
    try {
      await apiFetch('/api/evolution/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_iterations: 5, timeout_seconds: 300 }),
      });
      setTimeout(fetchStatus, 2000);
    } catch { /* silent */ }
    setStarting(false);
  }, [fetchStatus]);

  if (loading && !evoStatus) {
    return (
      <div className="p-4 space-y-3 animate-pulse">
        <div className="h-24 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        <div className="h-32 rounded-xl" style={{ background: 'var(--surface-2)' }} />
      </div>
    );
  }

  const st = evoStatus?.status || 'idle';
  const cfg = STATUS_CONFIG[st] || STATUS_CONFIG.idle;
  const StatusIcon = cfg.icon;
  const result = evoStatus?.result;
  const isRunning = st === 'running' || st === 'starting';

  return (
    <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
      {/* Status Card */}
      <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span style={{ color: cfg.color }}>
              <StatusIcon className={`w-5 h-5 ${isRunning ? 'animate-spin' : ''}`} />
            </span>
            <div>
              <span className="text-sm font-semibold" style={{ color: cfg.color }}>{cfg.label}</span>
              {evoStatus?.run_id && (
                <span className="text-[10px] ml-2" style={{ color: 'var(--text-tertiary)' }}>
                  #{evoStatus.run_id.slice(0, 8)}
                </span>
              )}
            </div>
          </div>
          <button
            onClick={startRun}
            disabled={isRunning || starting}
            className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95 disabled:opacity-50"
            style={{ background: 'rgba(129, 140, 248, 0.15)', color: '#818cf8' }}
          >
            {isRunning ? <ArrowPathIcon className="w-3.5 h-3.5 animate-spin" /> : <PlayIcon className="w-3.5 h-3.5" />}
            {isRunning ? 'Running' : 'Evolve'}
          </button>
        </div>

        {/* Quick stats */}
        {result && (
          <div className="grid grid-cols-4 gap-2">
            {[
              { label: 'Iterations', value: result.iterations_run },
              { label: 'Evaluations', value: result.total_evals },
              { label: 'Improvement', value: `${result.improvement >= 0 ? '+' : ''}${(result.improvement * 100).toFixed(1)}%` },
              { label: 'Duration', value: formatDuration(result.duration_seconds) },
            ].map(stat => (
              <div key={stat.label} className="rounded-lg p-2 text-center" style={{ background: 'var(--surface-2)' }}>
                <div className="text-sm font-bold text-chat-text">{stat.value}</div>
                <div className="text-[9px]" style={{ color: 'var(--text-tertiary)' }}>{stat.label}</div>
              </div>
            ))}
          </div>
        )}

        {!result && st === 'idle' && (
          <div className="flex items-center gap-2 py-2">
            <SparklesIcon className="w-4 h-4" style={{ color: 'var(--text-tertiary)' }} />
            <span className="text-xs" style={{ color: 'var(--text-tertiary)' }}>
              No evolution runs yet. Tap "Evolve" to start improving Aura's skills.
            </span>
          </div>
        )}
      </section>

      {/* Best Candidate */}
      {result?.best_candidate && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
            <SparklesIcon className="w-3.5 h-3.5" />
            Best Candidate #{result.best_candidate.id}
          </h3>

          <div className="flex items-center gap-3 mb-3">
            <div className="text-2xl font-bold" style={{ color: '#34d399' }}>
              {(result.best_candidate.avg_score * 100).toFixed(1)}%
            </div>
            <div className="text-xs" style={{ color: 'var(--text-secondary)' }}>
              Average Score
              {result.best_candidate.parent_id >= 0 && (
                <span className="ml-1" style={{ color: 'var(--text-tertiary)' }}>
                  (evolved from #{result.best_candidate.parent_id})
                </span>
              )}
            </div>
          </div>

          {/* Per-example scores */}
          <div className="space-y-1.5">
            {Object.entries(result.best_candidate.scores).slice(0, 8).map(([example, score]) => (
              <ScoreBar key={example} label={example} score={score} best={score > 0.8} />
            ))}
          </div>

          {/* Skills evolved */}
          {Object.keys(result.best_candidate.components).length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {Object.keys(result.best_candidate.components).map(skill => (
                <span key={skill} className="px-2 py-0.5 rounded-full text-[10px] font-medium"
                  style={{ background: 'rgba(129, 140, 248, 0.15)', color: '#818cf8' }}>
                  {skill}
                </span>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Pareto Frontier */}
      {result?.pareto_front && Object.keys(result.pareto_front).length > 0 && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-3" style={{ color: 'var(--text-secondary)' }}>
            <ChartBarIcon className="w-3.5 h-3.5" />
            Pareto Frontier ({Object.keys(result.pareto_front).length} examples)
          </h3>
          <div className="space-y-1">
            {Object.entries(result.pareto_front).map(([example, candidateId]) => (
              <div key={example} className="flex items-center justify-between px-2 py-1.5 rounded-lg" style={{ background: 'var(--surface-2)' }}>
                <span className="text-xs truncate flex-1" style={{ color: 'var(--text-secondary)' }}>{example}</span>
                <span className="text-[10px] font-mono ml-2" style={{ color: '#818cf8' }}>#{candidateId}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Candidate Genealogy */}
      {result?.all_candidates && result.all_candidates.length > 1 && (
        <section className="rounded-xl p-4" style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)' }}>
          <button
            onClick={() => setShowCandidates(p => !p)}
            className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider w-full text-left"
            style={{ color: 'var(--text-secondary)' }}
          >
            <BeakerIcon className="w-3.5 h-3.5" />
            All Candidates ({result.all_candidates.length})
            <span className="ml-auto text-[10px]">{showCandidates ? '\u25B2' : '\u25BC'}</span>
          </button>

          {showCandidates && (
            <div className="mt-3 space-y-2">
              {result.all_candidates
                .sort((a, b) => b.avg_score - a.avg_score)
                .map(candidate => {
                  const isBest = candidate.id === result.best_candidate?.id;
                  return (
                    <div
                      key={candidate.id}
                      className="flex items-center gap-3 px-3 py-2 rounded-lg"
                      style={{
                        background: isBest ? 'rgba(52, 211, 153, 0.08)' : 'var(--surface-2)',
                        border: isBest ? '1px solid rgba(52, 211, 153, 0.3)' : '1px solid transparent',
                      }}
                    >
                      <span className="text-xs font-mono w-8" style={{ color: isBest ? '#34d399' : 'var(--text-tertiary)' }}>
                        #{candidate.id}
                      </span>
                      <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--border-default)' }}>
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${candidate.avg_score * 100}%`,
                            background: isBest ? '#34d399' : 'var(--chat-accent)',
                            transition: 'width 0.5s ease',
                          }}
                        />
                      </div>
                      <span className="text-[10px] font-mono w-12 text-right" style={{ color: isBest ? '#34d399' : 'var(--text-secondary)' }}>
                        {(candidate.avg_score * 100).toFixed(1)}%
                      </span>
                      {candidate.parent_id >= 0 && (
                        <span className="text-[9px]" style={{ color: 'var(--text-tertiary)' }}>
                          from #{candidate.parent_id}
                        </span>
                      )}
                    </div>
                  );
                })}
            </div>
          )}
        </section>
      )}

      {/* Stop reason */}
      {result?.stop_reason && (
        <div className="text-center text-[10px] py-2" style={{ color: 'var(--text-tertiary)' }}>
          Stopped: {result.stop_reason.replace(/_/g, ' ')}
        </div>
      )}
    </div>
  );
}
