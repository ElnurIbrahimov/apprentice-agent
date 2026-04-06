import { useState, useEffect, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { HandStats, HandHistoryEntry, ApprovalRequest, HandTemplate } from '../types';

// ── State config ─────────────────────────────────────────────────────────────

const STATE_CONFIG: Record<string, { icon: string; color: string; bg: string; dot: string }> = {
  inactive:  { icon: '⚫', color: 'text-gray-400',   bg: 'bg-gray-500/10',    dot: 'bg-gray-500' },
  active:    { icon: '🟢', color: 'text-green-400',  bg: 'bg-green-500/10',   dot: 'bg-green-400' },
  running:   { icon: '🔄', color: 'text-blue-400',   bg: 'bg-blue-500/10',    dot: 'bg-blue-400' },
  paused:    { icon: '⏸️', color: 'text-yellow-400', bg: 'bg-yellow-500/10',  dot: 'bg-yellow-400' },
  cooldown:  { icon: '⏳', color: 'text-purple-400', bg: 'bg-purple-500/10',  dot: 'bg-purple-400' },
  error:     { icon: '🔴', color: 'text-red-400',    bg: 'bg-red-500/10',     dot: 'bg-red-400' },
};

const EXEC_COLORS: Record<string, string> = {
  success: 'bg-green-500',
  failed:  'bg-red-500',
  running: 'bg-yellow-400',
  skipped: 'bg-gray-500',
  pending: 'bg-yellow-400',
};

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatRelTime(isoOrTs: string | number | null): string {
  if (!isoOrTs) return 'Never';
  const ts = typeof isoOrTs === 'number' ? isoOrTs * 1000 : new Date(isoOrTs).getTime();
  const diff = (Date.now() - ts) / 1000;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

// ── PulsingDot ────────────────────────────────────────────────────────────────

function PulsingDot({ state }: { state: string }) {
  const cfg = STATE_CONFIG[state] ?? STATE_CONFIG.inactive;
  const isRunning = state === 'running';
  const isActive = state === 'active';
  return (
    <span className="relative flex h-2.5 w-2.5 shrink-0">
      {(isRunning || isActive) && (
        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${cfg.dot} opacity-60`} />
      )}
      <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${cfg.dot}`} />
    </span>
  );
}

// ── StatusCard (top grid) ────────────────────────────────────────────────────

function StatusCard({ hand, onAction, onClick }: {
  hand: HandStats;
  onAction: (name: string, action: string) => void;
  onClick: () => void;
}) {
  const cfg = STATE_CONFIG[hand.state] ?? STATE_CONFIG.inactive;
  const [running, setRunning] = useState(false);

  const handleRunNow = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setRunning(true);
    await onAction(hand.name, 'run');
    setTimeout(() => setRunning(false), 2000);
  };

  return (
    <div
      onClick={onClick}
      className={`relative rounded-xl border border-chat-border/20 ${cfg.bg} p-3 cursor-pointer hover:border-chat-border/40 transition-all group`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <PulsingDot state={hand.state} />
          <p className="text-xs font-semibold text-chat-text capitalize truncate">{hand.name}</p>
        </div>
        <span className={`text-[10px] font-medium px-1.5 py-0.5 rounded-full ${cfg.bg} ${cfg.color} shrink-0 border border-current/20`}>
          {hand.state}
        </span>
      </div>

      <p className="text-[10px] text-chat-text-secondary/60 truncate mb-2">{hand.description}</p>

      <div className="grid grid-cols-2 gap-x-2 text-[10px] text-chat-text-secondary/50 mb-2">
        <span>Last: {formatRelTime(hand.last_run)}</span>
        <span>{hand.total_runs} runs</span>
      </div>

      {hand.last_error && (
        <p className="text-[10px] text-red-400/70 truncate mb-2">⚠ {hand.last_error}</p>
      )}

      <button
        onClick={handleRunNow}
        disabled={hand.state === 'running' || running}
        className="w-full py-1 text-[10px] rounded-lg bg-blue-500/15 text-blue-300 hover:bg-blue-500/25 disabled:opacity-30 transition-colors opacity-0 group-hover:opacity-100"
      >
        {running ? 'Triggering…' : 'Run Now'}
      </button>
    </div>
  );
}

// ── ExecutionTimeline ─────────────────────────────────────────────────────────

interface TimelineEntry {
  id: string;
  handName: string;
  timestamp: string;
  success: boolean | null;
  summary: string;
  duration?: number;
  isRunning?: boolean;
}

function ExecutionTimeline({ entries, onSelect }: {
  entries: TimelineEntry[];
  onSelect: (entry: TimelineEntry) => void;
}) {
  if (entries.length === 0) {
    return (
      <div className="text-center py-6 text-chat-text-secondary/40 text-xs">
        No executions yet
      </div>
    );
  }

  return (
    <div className="space-y-0.5 relative">
      {/* Vertical line */}
      <div className="absolute left-[11px] top-0 bottom-0 w-px bg-chat-border/20" />

      {entries.map((entry) => {
        const status = entry.isRunning ? 'running' : entry.success === true ? 'success' : entry.success === false ? 'failed' : 'pending';
        const dotColor = EXEC_COLORS[status] ?? EXEC_COLORS.pending;

        return (
          <button
            key={entry.id}
            onClick={() => onSelect(entry)}
            className="w-full flex items-start gap-3 pl-1 pr-2 py-1.5 rounded-lg hover:bg-chat-surface/40 transition-colors text-left group"
          >
            {/* Dot on timeline */}
            <span className={`relative z-10 mt-1 flex-shrink-0 w-[10px] h-[10px] rounded-full ${dotColor} ${entry.isRunning ? 'animate-pulse' : ''} ring-2 ring-chat-bg/80`} />

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-medium text-chat-text capitalize">{entry.handName}</span>
                <span className={`text-[9px] px-1 py-px rounded ${
                  status === 'success' ? 'bg-green-500/15 text-green-400' :
                  status === 'failed'  ? 'bg-red-500/15 text-red-400' :
                  status === 'running' ? 'bg-yellow-500/15 text-yellow-400' :
                  'bg-gray-500/15 text-gray-400'
                }`}>
                  {status}
                </span>
                {entry.duration != null && (
                  <span className="text-[9px] text-chat-text-secondary/40">{formatDuration(entry.duration)}</span>
                )}
              </div>
              {entry.summary && (
                <p className="text-[10px] text-chat-text-secondary/50 truncate mt-0.5">{entry.summary}</p>
              )}
            </div>

            <span className="text-[9px] text-chat-text-secondary/30 shrink-0 mt-0.5 group-hover:text-chat-text-secondary/60">
              {formatRelTime(entry.timestamp)}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ── Detail Slide-out ──────────────────────────────────────────────────────────

function DetailPanel({ entry, onClose }: {
  entry: TimelineEntry | null;
  onClose: () => void;
}) {
  if (!entry) return null;
  const status = entry.isRunning ? 'running' : entry.success === true ? 'success' : 'failed';

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/40" onClick={onClose} />

      <div
        className="relative ml-auto w-80 h-full border-l border-chat-border/30 flex flex-col"
        style={{ background: 'var(--surface-1)' }}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-chat-border/20">
          <div>
            <p className="text-sm font-semibold text-chat-text capitalize">{entry.handName}</p>
            <p className="text-[10px] text-chat-text-secondary/50">{entry.timestamp}</p>
          </div>
          <button onClick={onClose} className="text-chat-text-secondary/40 hover:text-chat-text text-lg leading-none">×</button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs ${
            status === 'success' ? 'bg-green-500/15 text-green-400' :
            status === 'failed'  ? 'bg-red-500/15 text-red-400' :
            'bg-yellow-500/15 text-yellow-400'
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${EXEC_COLORS[status]}`} />
            {status}
            {entry.duration != null && ` · ${formatDuration(entry.duration)}`}
          </div>

          {entry.summary && (
            <div>
              <p className="text-[10px] text-chat-text-secondary/50 uppercase tracking-wide font-medium mb-1">Summary</p>
              <p className="text-xs text-chat-text-secondary/80 leading-relaxed">{entry.summary}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── ApprovalCard ──────────────────────────────────────────────────────────────

function ApprovalCard({ req, onResolve }: {
  req: ApprovalRequest;
  onResolve: (name: string, approved: boolean) => void;
}) {
  const [resolving, setResolving] = useState<'approve' | 'deny' | null>(null);

  const handle = async (approved: boolean) => {
    setResolving(approved ? 'approve' : 'deny');
    await onResolve(req.hand_name, approved);
  };

  return (
    <div className="rounded-xl border-2 border-amber-500/40 bg-amber-500/5 px-4 py-3 shadow-[0_0_12px_rgba(245,158,11,0.08)]">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-base">🔔</span>
        <p className="text-xs font-semibold text-amber-300 flex-1">Approval Required</p>
        <span className="text-[10px] text-chat-text-secondary/40">
          {Math.round(req.age_seconds)}s ago
        </span>
      </div>

      <div className="mb-1">
        <span className="text-xs text-chat-text font-medium capitalize">{req.hand_name}</span>
        <span className="text-xs text-chat-text-secondary/60"> wants to use </span>
        <span className="text-xs text-amber-300 font-mono bg-amber-500/10 px-1 py-px rounded">{req.tool_name}</span>
      </div>

      {req.args && Object.keys(req.args).length > 0 && (
        <div className="mt-2 mb-3 rounded-lg bg-black/20 p-2 text-[10px] font-mono text-chat-text-secondary/60 max-h-16 overflow-y-auto">
          {JSON.stringify(req.args, null, 2)}
        </div>
      )}

      <div className="flex gap-2 mt-2">
        <button
          onClick={() => handle(true)}
          disabled={resolving !== null}
          className="flex-1 py-1.5 text-xs rounded-lg bg-green-500/20 text-green-300 hover:bg-green-500/30 disabled:opacity-40 transition-colors font-medium"
        >
          {resolving === 'approve' ? '…' : '✓ Approve'}
        </button>
        <button
          onClick={() => handle(false)}
          disabled={resolving !== null}
          className="flex-1 py-1.5 text-xs rounded-lg bg-red-500/20 text-red-300 hover:bg-red-500/30 disabled:opacity-40 transition-colors font-medium"
        >
          {resolving === 'deny' ? '…' : '✗ Deny'}
        </button>
      </div>
    </div>
  );
}

// ── HandCard (expanded row, existing style preserved) ─────────────────────────

function HandCard({ hand, onAction, onDeleteHand }: {
  hand: HandStats;
  onAction: (name: string, action: string) => void;
  onDeleteHand: (name: string) => void;
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
          <PulsingDot state={hand.state} />
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

          <div className="flex gap-2 flex-wrap">
            <button
              onClick={() => onAction(hand.name, 'run')}
              disabled={hand.state === 'running'}
              className="px-3 py-1 text-[11px] rounded bg-blue-500/20 text-blue-300 hover:bg-blue-500/30 disabled:opacity-30 transition-colors"
            >
              Run Now
            </button>
            {hand.is_custom && (
              <button onClick={() => onDeleteHand(hand.name)}
                className="px-3 py-1 text-[11px] rounded bg-red-500/20 text-red-300 hover:bg-red-500/30">
                Delete
              </button>
            )}
            {(hand.state === 'inactive' || hand.state === 'cooldown') ? (
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

// ── Main Component ────────────────────────────────────────────────────────────

export default function HandsDashboard() {
  const [hands, setHands] = useState<HandStats[]>([]);
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [history, setHistory] = useState<HandHistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [createDesc, setCreateDesc] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [templates, setTemplates] = useState<HandTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [templateVars, setTemplateVars] = useState<Record<string, string>>({});
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [view, setView] = useState<'status' | 'timeline' | 'list'>('status');
  const [selectedExec, setSelectedExec] = useState<TimelineEntry | null>(null);
  const [globalPausing, setGlobalPausing] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Live action trace from WebSocket
  const [liveTrace, setLiveTrace] = useState<Array<{ hand: string; step: number; description: string; timestamp: number }>>([]);

  useEffect(() => {
    const handler = (e: Event) => {
      const d = (e as CustomEvent).detail;
      setLiveTrace(prev => [...prev.slice(-19), { hand: d.hand, step: d.step, description: d.description, timestamp: d.timestamp }]);
    };
    const clearHandler = () => setLiveTrace([]);
    document.addEventListener('aura:action_trace', handler);
    document.addEventListener('aura:hand_event', clearHandler);
    return () => {
      document.removeEventListener('aura:action_trace', handler);
      document.removeEventListener('aura:hand_event', clearHandler);
    };
  }, []);

  const fetchAll = useCallback(async () => {
    try {
      const [handsRes, approvalsRes, historyRes] = await Promise.all([
        fetch('/api/hands').then(r => r.json()).catch(() => ({ hands: [] })),
        fetch('/api/hands/approvals').then(r => r.json()).catch(() => ({ approvals: [] })),
        fetch('/api/hands/history?limit=30').then(r => r.json()).catch(() => ({ history: [] })),
      ]);
      setHands(handsRes.hands || []);
      setApprovals(approvalsRes.approvals || []);
      setHistory(historyRes.history || []);
      setError(null);
      setLastRefresh(new Date());
    } catch {
      setError('Failed to load hands data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);
  usePolling(fetchAll, 10000);

  const handleAction = useCallback(async (name: string, action: string) => {
    try {
      await fetch(`/api/hands/${name}/${action}`, { method: 'POST' });
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

  const handleCreate = async () => {
    setCreating(true); setCreateError(null);
    try {
      const res = await fetch('/api/hands/create', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: createDesc }),
      });
      if (res.ok) { setShowCreateModal(false); setCreateDesc(''); fetchAll(); }
      else { const d = await res.json(); setCreateError(d.detail || 'Failed'); }
    } catch { setCreateError('Network error'); }
    finally { setCreating(false); }
  };

  const handleFromTemplate = async (name: string) => {
    try {
      const res = await fetch('/api/hands/from-template', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template_name: name, variables: templateVars }),
      });
      if (res.ok) { setShowTemplates(false); setSelectedTemplate(null); setTemplateVars({}); fetchAll(); }
    } catch {}
  };

  const handleDeleteHand = async (name: string) => {
    if (!confirm(`Delete hand "${name}"?`)) return;
    await fetch(`/api/hands/${name}`, { method: 'DELETE' });
    fetchAll();
  };

  const fetchTemplates = async () => {
    try {
      const res = await fetch('/api/hands/templates');
      if (res.ok) { const d = await res.json(); setTemplates(d.templates || []); }
    } catch {}
  };

  const handlePauseAll = async () => {
    setGlobalPausing(true);
    const active = hands.filter(h => ['active', 'running'].includes(h.state));
    await Promise.all(active.map(h => fetch(`/api/hands/${h.name}/pause`, { method: 'POST' }).catch(() => {})));
    await fetchAll();
    setGlobalPausing(false);
  };

  const handleResumeAll = async () => {
    setGlobalPausing(true);
    const paused = hands.filter(h => h.state === 'paused');
    await Promise.all(paused.map(h => fetch(`/api/hands/${h.name}/activate`, { method: 'POST' }).catch(() => {})));
    await fetchAll();
    setGlobalPausing(false);
  };

  // ── Derived stats ──────────────────────────────────────────────────────────

  const activeCount   = hands.filter(h => ['active', 'running'].includes(h.state)).length;
  const pausedCount   = hands.filter(h => h.state === 'paused').length;
  const runningNow    = hands.filter(h => h.state === 'running').length;
  const successRate   = history.length > 0
    ? Math.round(history.filter(e => (e.action_data as Record<string, unknown>).success === true).length / history.length * 100)
    : null;

  // Today's executions (approximate: within 24h)
  const todayExecs = history.filter(e => {
    const ts = new Date(e.timestamp).getTime();
    return !isNaN(ts) && (Date.now() - ts) < 86400000;
  }).length;

  // ── Timeline entries built from history ───────────────────────────────────

  const timelineEntries: TimelineEntry[] = history.map((entry, i) => {
    const data = (entry.action_data || {}) as Record<string, unknown>;
    const handName = (data.hand as string) || entry.agent_id?.replace('hand:', '') || '?';
    return {
      id: `${i}-${entry.timestamp}`,
      handName,
      timestamp: entry.timestamp,
      success: data.success as boolean ?? null,
      summary: (data.summary as string || '').slice(0, 200),
      duration: typeof data.duration_ms === 'number' ? data.duration_ms : undefined,
      isRunning: hands.find(h => h.name === handName)?.state === 'running',
    };
  });

  // ── Loading / Error ────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3">
        <div className="w-6 h-6 border-2 border-chat-accent/40 border-t-chat-accent rounded-full animate-spin" />
        <p className="text-chat-text-secondary/50 text-xs">Loading mission control…</p>
      </div>
    );
  }

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="h-full overflow-y-auto">
      {/* Detail slide-out */}
      <DetailPanel entry={selectedExec} onClose={() => setSelectedExec(null)} />

      <div className="p-4 space-y-4">

        {/* ── Header ── */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-chat-text">Mission Control</h2>
            <p className="text-[10px] text-chat-text-secondary/40 mt-0.5">
              {hands.length} hands · refreshed {formatRelTime(lastRefresh.toISOString())}
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            {error && (
              <span className="text-[10px] text-red-400/80 bg-red-500/10 px-2 py-0.5 rounded-full">
                ⚠ {error}
              </span>
            )}
            <button
              onClick={fetchAll}
              className="p-1.5 rounded-lg text-chat-text-secondary/40 hover:text-chat-text hover:bg-chat-surface/50 transition-colors"
              title="Refresh now"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M1 4v6h6M23 20v-6h-6" /><path d="M20.49 9A9 9 0 0 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15" />
              </svg>
            </button>
          </div>
        </div>

        {/* ── Stats Row ── */}
        <div className="grid grid-cols-5 gap-1.5">
          {[
            { label: 'Total',   value: String(hands.length),              sub: 'hands' },
            { label: 'Active',  value: String(activeCount),               sub: runningNow > 0 ? `${runningNow} running` : 'hands',  highlight: activeCount > 0 ? 'text-green-400' : '' },
            { label: 'Paused',  value: String(pausedCount),               sub: 'hands',  highlight: pausedCount > 0 ? 'text-yellow-400' : '' },
            { label: 'Today',   value: String(todayExecs),                sub: 'execs' },
            { label: 'Success', value: successRate !== null ? `${successRate}%` : '—', sub: 'rate', highlight: successRate !== null && successRate >= 80 ? 'text-green-400' : successRate !== null && successRate < 60 ? 'text-red-400' : '' },
          ].map(s => (
            <div key={s.label} className="text-center rounded-lg border border-chat-border/10 py-2 px-1" style={{ background: 'var(--surface-1)' }}>
              <p className={`text-sm font-bold ${s.highlight || 'text-chat-text'}`}>{s.value}</p>
              <p className="text-[9px] text-chat-text-secondary/40 mt-0.5 leading-tight">{s.label}</p>
            </div>
          ))}
        </div>

        {/* ── Quick Actions bar ── */}
        <div className="flex gap-2 flex-wrap">
          <button onClick={() => { fetchTemplates(); setShowTemplates(true); }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-chat-accent text-white hover:opacity-90 transition-opacity">
            📋 Templates
          </button>
          <button onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs border border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-border/60 transition-colors">
            + New Hand
          </button>

          <div className="flex-1" />

          {pausedCount > 0 && (
            <button
              onClick={handleResumeAll}
              disabled={globalPausing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-green-500/15 text-green-300 hover:bg-green-500/25 disabled:opacity-40 transition-colors"
            >
              ▶ Resume All ({pausedCount})
            </button>
          )}
          {activeCount > 0 && (
            <button
              onClick={handlePauseAll}
              disabled={globalPausing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs bg-yellow-500/15 text-yellow-300 hover:bg-yellow-500/25 disabled:opacity-40 transition-colors"
            >
              ⏸ Pause All ({activeCount})
            </button>
          )}
        </div>

        {/* ── Approval Queue (prominent) ── */}
        {approvals.length > 0 && (
          <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm">🔔</span>
              <p className="text-xs font-semibold text-amber-300 flex-1">Pending Approvals</p>
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-amber-500 text-[10px] font-bold text-black">
                {approvals.length}
              </span>
            </div>
            {approvals.map(req => (
              <ApprovalCard key={req.request_id} req={req} onResolve={handleApproval} />
            ))}
          </div>
        )}

        {/* ── Live Trace ── */}
        {liveTrace.length > 0 && (
          <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
              <p className="text-[11px] text-blue-400/80 font-semibold">Live Trace</p>
            </div>
            <div className="space-y-1 max-h-28 overflow-y-auto">
              {liveTrace.map((t, i) => (
                <div key={i} className="flex items-center gap-2 text-[11px] text-chat-text-secondary/70">
                  <span className="text-blue-400/50 font-mono w-4 text-right shrink-0">{t.step}</span>
                  <span className="text-chat-text-secondary capitalize shrink-0">{t.hand}</span>
                  <span className="text-chat-text-secondary/30">—</span>
                  <span className="truncate">{t.description}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── View Tabs ── */}
        {hands.length > 0 && (
          <div className="flex gap-1 rounded-lg p-1 border border-chat-border/10" style={{ background: 'var(--surface-1)' }}>
            {([['status', 'Status Grid'], ['timeline', 'Timeline'], ['list', 'Detail List']] as const).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setView(id)}
                className={`flex-1 py-1.5 text-[11px] rounded-md transition-colors font-medium ${
                  view === id
                    ? 'bg-chat-accent/20 text-chat-accent'
                    : 'text-chat-text-secondary/50 hover:text-chat-text-secondary'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* ── Status Grid View ── */}
        {view === 'status' && hands.length > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {hands.map(hand => (
              <StatusCard
                key={hand.name}
                hand={hand}
                onAction={handleAction}
                onClick={() => setView('list')}
              />
            ))}
          </div>
        )}

        {/* ── Timeline View ── */}
        {view === 'timeline' && (
          <div className="rounded-xl border border-chat-border/15 p-3" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center justify-between mb-3">
              <p className="text-[11px] font-semibold text-chat-text-secondary/70 uppercase tracking-wide">Execution Timeline</p>
              <span className="text-[10px] text-chat-text-secondary/40">{timelineEntries.length} executions</span>
            </div>
            <ExecutionTimeline entries={timelineEntries} onSelect={setSelectedExec} />
          </div>
        )}

        {/* ── Detail List View ── */}
        {view === 'list' && (
          <div className="space-y-2">
            {hands.length === 0 ? (
              <div className="text-center py-8 text-chat-text-secondary text-xs">
                No autonomous hands registered yet.
              </div>
            ) : (
              hands.map(hand => (
                <HandCard key={hand.name} hand={hand} onAction={handleAction} onDeleteHand={handleDeleteHand} />
              ))
            )}
          </div>
        )}

        {/* ── Empty state ── */}
        {hands.length === 0 && (
          <div className="text-center py-12 space-y-3">
            <p className="text-3xl">🤖</p>
            <p className="text-sm font-medium text-chat-text">No hands registered</p>
            <p className="text-xs text-chat-text-secondary/50">Create your first autonomous hand to get started</p>
            <button
              onClick={() => setShowCreateModal(true)}
              className="mt-2 px-4 py-2 text-xs rounded-xl bg-chat-accent text-white hover:opacity-90 transition-opacity"
            >
              + Create First Hand
            </button>
          </div>
        )}

      </div>{/* end p-4 */}

      {/* ── Create Custom Hand Modal ── */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-md rounded-2xl p-6 border border-chat-border shadow-2xl" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-xl">🖐</span>
              <h3 className="text-sm font-semibold text-chat-text">Create Custom Hand</h3>
            </div>
            <p className="text-[11px] text-chat-text-secondary/60 mb-3">
              Describe what this hand should do autonomously. Be specific about triggers and actions.
            </p>
            <textarea
              value={createDesc}
              onChange={e => setCreateDesc(e.target.value)}
              placeholder="e.g. Monitor Hacker News for AI papers and summarize them daily at 9am"
              className="w-full p-3 rounded-xl border border-chat-border text-sm text-chat-text resize-none h-28 outline-none focus:border-chat-accent transition-colors"
              style={{ background: 'var(--surface-2)' }}
              autoFocus
            />
            {createError && <p className="text-xs text-red-400 mt-2">{createError}</p>}
            <div className="flex justify-end gap-2 mt-4">
              <button onClick={() => { setShowCreateModal(false); setCreateDesc(''); setCreateError(null); }}
                className="px-4 py-2 text-xs text-chat-text-secondary hover:text-chat-text transition-colors">
                Cancel
              </button>
              <button
                onClick={handleCreate}
                disabled={!createDesc.trim() || creating}
                className="px-5 py-2 text-xs rounded-xl bg-chat-accent text-white disabled:opacity-40 hover:opacity-90 transition-opacity font-medium"
              >
                {creating ? 'Creating…' : 'Create Hand'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Template Picker Modal ── */}
      {showTemplates && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl p-6 border border-chat-border shadow-2xl" style={{ background: 'var(--surface-1)' }}>
            <div className="flex items-center gap-3 mb-4">
              <span className="text-xl">📋</span>
              <h3 className="text-sm font-semibold text-chat-text">Hand Templates</h3>
            </div>
            {templates.length === 0 ? (
              <p className="text-xs text-chat-text-secondary/60 py-6 text-center">No templates available.</p>
            ) : (
              <div className="grid grid-cols-1 gap-2 max-h-72 overflow-y-auto mb-4 pr-1">
                {templates.map(t => (
                  <button
                    key={t.name}
                    onClick={() => setSelectedTemplate(selectedTemplate === t.name ? null : t.name)}
                    className={`text-left rounded-xl border px-4 py-3 transition-colors ${
                      selectedTemplate === t.name
                        ? 'border-chat-accent bg-chat-accent/10'
                        : 'border-chat-border/30 hover:border-chat-border'
                    }`}
                  >
                    <p className="text-xs font-medium text-chat-text capitalize">{t.name}</p>
                    <p className="text-[11px] text-chat-text-secondary/60 mt-0.5">{t.description}</p>
                    {t.trigger_on_drive && (
                      <p className="text-[10px] text-chat-text-secondary/40 mt-1">Drive: {t.trigger_on_drive}</p>
                    )}
                  </button>
                ))}
              </div>
            )}
            {selectedTemplate && (
              <div className="mb-4">
                <p className="text-[11px] text-chat-text-secondary/60 mb-2">Variables (optional)</p>
                <input
                  type="text"
                  placeholder="key=value, e.g. topic=AI safety"
                  value={templateVars['__raw'] || ''}
                  onChange={e => setTemplateVars({ __raw: e.target.value })}
                  className="w-full p-2.5 rounded-xl border border-chat-border text-xs text-chat-text outline-none focus:border-chat-accent transition-colors"
                  style={{ background: 'var(--surface-2)' }}
                />
              </div>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={() => { setShowTemplates(false); setSelectedTemplate(null); setTemplateVars({}); }}
                className="px-4 py-2 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => selectedTemplate && handleFromTemplate(selectedTemplate)}
                disabled={!selectedTemplate}
                className="px-5 py-2 text-xs rounded-xl bg-chat-accent text-white disabled:opacity-40 hover:opacity-90 transition-opacity font-medium"
              >
                Use Template
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
