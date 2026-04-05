import { useState, useEffect, useCallback, useMemo } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { HandStats, HandHistoryEntry, ApprovalRequest } from '../types';

// ── State config ────────────────────────────────────────────────────────────

const STATE_CONFIG: Record<string, { icon: string; color: string; bg: string }> = {
  inactive:  { icon: '\u26ab', color: 'text-gray-400',   bg: 'bg-gray-500/10' },
  active:    { icon: '\ud83d\udfe2', color: 'text-green-400',  bg: 'bg-green-500/10' },
  running:   { icon: '\ud83d\udd04', color: 'text-blue-400',   bg: 'bg-blue-500/10' },
  paused:    { icon: '\u23f8\ufe0f', color: 'text-yellow-400', bg: 'bg-yellow-500/10' },
  cooldown:  { icon: '\u23f3', color: 'text-purple-400', bg: 'bg-purple-500/10' },
  error:     { icon: '\ud83d\udd34', color: 'text-red-400',    bg: 'bg-red-500/10' },
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatRelTime(isoOrTs: string | number | null): string {
  if (!isoOrTs) return 'Never';
  const ts = typeof isoOrTs === 'number' ? isoOrTs * 1000 : new Date(isoOrTs).getTime();
  const diff = (Date.now() - ts) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

// ── HandCard ────────────────────────────────────────────────────────────────

function HandCard({ hand, onAction }: {
  hand: HandStats;
  onAction: (name: string, action: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [handHistory, setHandHistory] = useState<HandHistoryEntry[]>([]);
  const cfg = STATE_CONFIG[hand.state] ?? STATE_CONFIG.inactive;

  useEffect(() => {
    if (!expanded) return;
    fetch(`/api/hands/history?limit=10&hand=${encodeURIComponent(hand.name)}`)
      .then(r => r.json())
      .then(data => setHandHistory(data.history || []))
      .catch(() => {});
  }, [expanded, hand.name]);

  return (
    <div className={`rounded-lg border border-chat-border/20 ${cfg.bg} overflow-hidden`}>
      <div className="px-4 py-3 flex items-center justify-between">
        <button
          onClick={() => setExpanded(prev => !prev)}
          className="flex items-center gap-3 flex-1 text-left"
        >
          <span className="text-lg">{cfg.icon}</span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-chat-text capitalize">{hand.name}</p>
            <p className="text-[11px] text-chat-text-secondary/60 truncate">{hand.description}</p>
          </div>
          <div className="text-right text-[11px] text-chat-text-secondary/60 shrink-0">
            <p>{hand.total_runs} runs · ${hand.total_cost.toFixed(4)}</p>
            <p>Last: {formatRelTime(hand.last_run)}</p>
          </div>
        </button>
      </div>

      {expanded && (
        <div className="px-4 pb-3 border-t border-chat-border/10 pt-2">
          <div className="grid grid-cols-2 gap-2 text-[11px] text-chat-text-secondary/80 mb-3">
            <p>State: <span className={cfg.color}>{hand.state}</span></p>
            <p>Model: {hand.model_preference}</p>
            <p>Failures: {hand.consecutive_failures}</p>
            <p>Drive: {hand.trigger_on_drive ?? 'None'}</p>
            <p>Idle only: {hand.idle_only ? 'Yes' : 'No'}</p>
          </div>

          {/* Per-hand run history dots */}
          {handHistory.length > 0 && (
            <div className="mb-3">
              <p className="text-[10px] text-chat-text-secondary/50 font-medium mb-1.5 uppercase tracking-wide">Recent Runs</p>
              <div className="flex gap-1.5 flex-wrap">
                {handHistory.map((entry, i) => {
                  const data = entry.action_data;
                  const success = data.success as boolean;
                  return (
                    <div key={i}
                      title={`${entry.timestamp} — ${String(data.summary || '').slice(0, 80)}`}
                      className={`w-2.5 h-2.5 rounded-full ${success ? 'bg-green-500/70' : 'bg-red-500/70'}`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {hand.last_error && (
            <p className="text-[11px] text-red-400/80 mb-2 truncate">
              Error: {hand.last_error}
            </p>
          )}

          <div className="flex gap-2">
            <button
              onClick={() => onAction(hand.name, 'run')}
              disabled={hand.state === 'running'}
              className="px-3 py-1 text-[11px] rounded bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 disabled:opacity-30 transition-colors"
            >
              Run Now
            </button>
            {hand.state === 'inactive' || hand.state === 'cooldown' ? (
              <button
                onClick={() => onAction(hand.name, 'activate')}
                className="px-3 py-1 text-[11px] rounded bg-green-500/20 text-green-300 hover:bg-green-500/30 transition-colors"
              >
                Activate
              </button>
            ) : hand.state === 'paused' ? (
              <button
                onClick={() => onAction(hand.name, 'activate')}
                className="px-3 py-1 text-[11px] rounded bg-green-500/20 text-green-300 hover:bg-green-500/30 transition-colors"
              >
                Resume
              </button>
            ) : null}
            {['active', 'running'].includes(hand.state) && (
              <button
                onClick={() => onAction(hand.name, 'pause')}
                className="px-3 py-1 text-[11px] rounded bg-yellow-500/20 text-yellow-300 hover:bg-yellow-500/30 transition-colors"
              >
                Pause
              </button>
            )}
            {['active', 'running', 'paused'].includes(hand.state) && (
              <button
                onClick={() => onAction(hand.name, 'deactivate')}
                className="px-3 py-1 text-[11px] rounded bg-gray-500/20 text-gray-300 hover:bg-gray-500/30 transition-colors"
              >
                Deactivate
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ── ApprovalCard ────────────────────────────────────────────────────────────

function ApprovalCard({ req, onResolve }: {
  req: ApprovalRequest;
  onResolve: (name: string, approved: boolean) => void;
}) {
  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm">\ud83d\udd10</span>
        <p className="text-xs font-medium text-amber-300">Approval Required</p>
        <span className="text-[10px] text-chat-text-secondary/50 ml-auto">
          {Math.round(req.age_seconds)}s ago
        </span>
      </div>
      <p className="text-[11px] text-chat-text-secondary/80 mb-2">
        <span className="text-chat-text capitalize">{req.hand_name}</span> wants to use{' '}
        <span className="text-amber-300 font-mono">{req.tool_name}</span>
      </p>
      <div className="flex gap-2">
        <button
          onClick={() => onResolve(req.hand_name, true)}
          className="px-3 py-1 text-[11px] rounded bg-green-500/20 text-green-300 hover:bg-green-500/30 transition-colors"
        >
          Approve
        </button>
        <button
          onClick={() => onResolve(req.hand_name, false)}
          className="px-3 py-1 text-[11px] rounded bg-red-500/20 text-red-300 hover:bg-red-500/30 transition-colors"
        >
          Deny
        </button>
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────

export default function HandsDashboard() {
  const [hands, setHands] = useState<HandStats[]>([]);
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [history, setHistory] = useState<HandHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [handsRes, approvalsRes, historyRes] = await Promise.all([
        fetch('/api/hands').then(r => r.json()).catch(() => ({ hands: [] })),
        fetch('/api/hands/approvals').then(r => r.json()).catch(() => ({ approvals: [] })),
        fetch('/api/hands/history?limit=10').then(r => r.json()).catch(() => ({ history: [] })),
      ]);
      setHands(handsRes.hands || []);
      setApprovals(approvalsRes.approvals || []);
      setHistory(historyRes.history || []);
    } catch {
      // Swallow
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);
  usePolling(fetchAll, 15000);

  const handleAction = useCallback(async (name: string, action: string) => {
    try {
      await fetch(`/api/hands/${name}/${action}`, { method: 'POST' });
      // Refresh immediately
      await fetchAll();
    } catch {
      // Swallow
    }
  }, [fetchAll]);

  const handleApproval = useCallback(async (name: string, approved: boolean) => {
    try {
      await fetch(`/api/hands/${name}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      });
      await fetchAll();
    } catch {
      // Swallow
    }
  }, [fetchAll]);

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-chat-text-secondary/50 text-sm">Loading hands...</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-chat-text">Autonomous Hands</h2>
        <span className="text-[10px] text-chat-text-secondary/40">
          {hands.length} registered · {hands.filter(h => h.state === 'active' || h.state === 'running').length} active
        </span>
      </div>

      {/* Stats summary */}
      {hands.length > 0 && (() => {
        const totalRuns = hands.reduce((s, h) => s + h.total_runs, 0);
        const totalCost = hands.reduce((s, h) => s + h.total_cost, 0);
        const activeCount = hands.filter(h => ['active', 'running'].includes(h.state)).length;
        const successRate = history.length > 0
          ? Math.round(history.filter(e => (e.action_data as Record<string, unknown>).success === true).length / history.length * 100)
          : null;
        return (
          <div className="grid grid-cols-4 gap-2 p-2.5 rounded-lg border border-chat-border/10" style={{ background: 'var(--surface-1)' }}>
            {[
              { label: 'Runs', value: String(totalRuns) },
              { label: 'Success', value: successRate !== null ? `${successRate}%` : '—' },
              { label: 'Cost', value: `$${totalCost.toFixed(3)}` },
              { label: 'Active', value: `${activeCount}/${hands.length}` },
            ].map(s => (
              <div key={s.label} className="text-center">
                <p className="text-[10px] text-chat-text-secondary/50">{s.label}</p>
                <p className="text-sm font-medium text-chat-text">{s.value}</p>
              </div>
            ))}
          </div>
        );
      })()}

      {/* Approval Queue */}
      {approvals.length > 0 && (
        <div className="space-y-2">
          <p className="text-[11px] text-amber-300/80 font-medium">
            Pending Approvals ({approvals.length})
          </p>
          {approvals.map(req => (
            <ApprovalCard key={req.request_id} req={req} onResolve={handleApproval} />
          ))}
        </div>
      )}

      {/* Hand Cards */}
      <div className="space-y-2">
        {hands.length === 0 ? (
          <div className="text-center py-8 text-chat-text-secondary text-xs">
            No autonomous hands registered yet.
          </div>
        ) : (
          hands.map(hand => (
            <HandCard key={hand.name} hand={hand} onAction={handleAction} />
          ))
        )}
      </div>

      {/* Recent History */}
      {history.length > 0 && (
        <div>
          <p className="text-[11px] text-chat-text-secondary/60 font-medium mb-2">Recent Activity</p>
          <div className="space-y-1">
            {history.map((entry, i) => {
              const data = (entry.action_data || {}) as Record<string, unknown>;
              const success = data.success as boolean;
              const handName = data.hand as string || entry.agent_id?.replace('hand:', '') || '?';
              const summary = (data.summary as string || '').slice(0, 120);
              return (
                <div
                  key={i}
                  className="rounded border border-chat-border/10 bg-chat-surface/30 px-3 py-2 text-[11px]"
                >
                  <div className="flex items-center gap-2">
                    <span>{success ? '\u2705' : '\u274c'}</span>
                    <span className="text-chat-text capitalize">{handName}</span>
                    <span className="text-chat-text-secondary/40 ml-auto">{entry.timestamp}</span>
                  </div>
                  {summary && (
                    <p className="text-chat-text-secondary/60 mt-1 truncate">{summary}</p>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
