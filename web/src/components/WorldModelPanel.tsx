/**
 * WorldModelPanel — Aura's understanding of your life context.
 *
 * Visualizes projects, goals, and key entities from the Knowledge Graph,
 * showing what Aura knows about your world and what needs attention.
 */

import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import {
  FolderIcon,
  FlagIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface KGNode {
  id: string;
  label: string;
  type: string;
  confidence?: number;
  access_count?: number;
  properties?: Record<string, any>;
}

// How many days since access before "stale"
const STALE_DAYS = 7;

function daysSince(dateStr?: string): number {
  if (!dateStr) return 999;
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return 999;
    return Math.floor((Date.now() - d.getTime()) / 86400000);
  } catch { return 999; }
}

function healthColor(days: number): string {
  if (days <= 2) return '#34d399'; // active
  if (days <= STALE_DAYS) return '#fbbf24'; // slowing
  return '#f87171'; // stale
}

function healthLabel(days: number): string {
  if (days <= 2) return 'Active';
  if (days <= STALE_DAYS) return 'Slowing';
  return 'Stale';
}

function formatDaysAgo(days: number): string {
  if (days === 0) return 'Today';
  if (days === 1) return 'Yesterday';
  if (days < 30) return `${days}d ago`;
  if (days < 365) return `${Math.floor(days / 30)}mo ago`;
  return `${Math.floor(days / 365)}y ago`;
}

export function WorldModelPanel() {
  const [projects, setProjects] = useState<KGNode[]>([]);
  const [goals, setGoals] = useState<KGNode[]>([]);
  const [entities, setEntities] = useState<KGNode[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await apiFetch('/api/knowledge-graph');
      if (!res.ok) { setError('Could not load knowledge graph'); setLoading(false); return; }
      const data = await res.json();
      const nodes: KGNode[] = data.nodes || [];

      setProjects(nodes.filter(n => n.type === 'project').sort((a, b) => (b.access_count || 0) - (a.access_count || 0)));
      setGoals(nodes.filter(n => n.type === 'goal').sort((a, b) => (b.access_count || 0) - (a.access_count || 0)));
      setEntities(nodes.filter(n => !['project', 'goal'].includes(n.type)).sort((a, b) => (b.access_count || 0) - (a.access_count || 0)).slice(0, 12));
      setError(null);
    } catch {
      setError('Failed to fetch world model');
    }
    setLoading(false);
  }, []);

  usePolling(fetchData, 30000);

  if (loading && projects.length === 0) {
    return (
      <div className="p-4 space-y-3 animate-pulse">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="h-16 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        ))}
      </div>
    );
  }

  if (error && projects.length === 0 && goals.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full py-12">
        <ExclamationTriangleIcon className="w-8 h-8 mb-2" style={{ color: 'var(--text-tertiary)' }} />
        <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{error}</p>
      </div>
    );
  }

  const isEmpty = projects.length === 0 && goals.length === 0 && entities.length === 0;

  return (
    <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
      {isEmpty ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <FolderIcon className="w-10 h-10 mb-3" style={{ color: 'var(--text-tertiary)' }} />
          <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            Your world model is empty
          </p>
          <p className="text-xs mt-1 max-w-xs" style={{ color: 'var(--text-tertiary)' }}>
            Start chatting about your projects, goals, and interests. Aura will learn and build a model of your world.
          </p>
        </div>
      ) : (
        <>
          {/* Projects */}
          {projects.length > 0 && (
            <section>
              <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)' }}>
                <FolderIcon className="w-3.5 h-3.5" />
                Projects ({projects.length})
              </h3>
              <div className="space-y-2">
                {projects.map((p, i) => {
                  const days = daysSince(p.properties?.last_mentioned || p.properties?.updated_at);
                  const color = healthColor(days);
                  return (
                    <div
                      key={p.id}
                      className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
                      style={{
                        background: 'var(--surface-1)',
                        border: `1px solid var(--border-default)`,
                        animation: `spring-up 0.3s ease ${i * 40}ms both`,
                      }}
                    >
                      <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: color, boxShadow: `0 0 6px ${color}60` }} />
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-chat-text truncate">{p.label}</div>
                        <div className="flex items-center gap-2 mt-0.5">
                          <span className="text-[10px]" style={{ color }}>{healthLabel(days)}</span>
                          <span className="text-[10px]" style={{ color: 'var(--text-tertiary)' }}>
                            <ClockIcon className="w-3 h-3 inline -mt-px" /> {formatDaysAgo(days)}
                          </span>
                        </div>
                      </div>
                      {p.access_count != null && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded-md" style={{ background: 'var(--surface-2)', color: 'var(--text-tertiary)' }}>
                          {p.access_count}x
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* Goals */}
          {goals.length > 0 && (
            <section>
              <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)' }}>
                <FlagIcon className="w-3.5 h-3.5" />
                Goals ({goals.length})
              </h3>
              <div className="space-y-2">
                {goals.map((g, i) => (
                  <div
                    key={g.id}
                    className="px-3 py-2.5 rounded-xl"
                    style={{
                      background: 'var(--surface-1)',
                      border: '1px solid var(--border-default)',
                      animation: `spring-up 0.3s ease ${i * 40}ms both`,
                    }}
                  >
                    <div className="text-sm font-medium text-chat-text">{g.label}</div>
                    {g.confidence != null && (
                      <div className="mt-1.5 h-1.5 rounded-full" style={{ background: 'var(--border-default)' }}>
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${Math.min(100, g.confidence * 100)}%`,
                            background: '#34d399',
                            transition: 'width 0.5s ease',
                          }}
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Key Entities */}
          {entities.length > 0 && (
            <section>
              <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-secondary)' }}>
                <ArrowPathIcon className="w-3.5 h-3.5" />
                Key Entities ({entities.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {entities.map(e => {
                  const typeColors: Record<string, string> = {
                    concept: '#a78bfa', entity: '#60a5fa', person: '#f472b6',
                    tool: '#fbbf24', event: '#f97316', fact: '#34d399',
                  };
                  const c = typeColors[e.type] || '#a1a1aa';
                  return (
                    <span
                      key={e.id}
                      className="inline-flex items-center gap-1 px-2 py-1 rounded-lg text-xs"
                      style={{ background: `${c}15`, color: c, border: `1px solid ${c}25` }}
                    >
                      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
                      {e.label}
                    </span>
                  );
                })}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
