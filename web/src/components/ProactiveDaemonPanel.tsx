import { useState, useCallback, useRef } from 'react';
import { usePolling } from '../hooks/usePolling';
import {
  PlayIcon,
  PauseIcon,
  StopIcon,
  BoltIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface DaemonStatus {
  running: boolean;
  state: string;
  stats: {
    events_received?: number;
    events_filtered?: number;
    decisions_made?: number;
    messages_sent?: number;
    uptime_seconds?: number;
  };
  beliefs?: {
    user_busy: number;
    user_receptive: number;
    task_urgent: number;
    context_stable: number;
    uncertainty: number;
  };
  pending_messages: number;
}

interface ProactiveMessage {
  id?: string;
  action: string;
  content: string;
  priority: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

const BELIEF_COLORS: Record<string, string> = {
  user_busy: '#ef4444',      // Red
  user_receptive: '#10b981', // Green
  task_urgent: '#f59e0b',    // Amber
  context_stable: '#3b82f6', // Blue
  uncertainty: '#8b5cf6',    // Purple
};

const BELIEF_LABELS: Record<string, string> = {
  user_busy: 'User Busy',
  user_receptive: 'Receptive',
  task_urgent: 'Urgent Task',
  context_stable: 'Context Stable',
  uncertainty: 'Uncertainty',
};

export function ProactiveDaemonPanel() {
  const [status, setStatus] = useState<DaemonStatus | null>(null);
  const [messages, setMessages] = useState<ProactiveMessage[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const seenIds = useRef<Set<string>>(new Set());

  // Fetch daemon status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await apiFetch('/api/proactive/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
        setError(null);
      }
    } catch (e: any) {
      setError('Failed to fetch daemon status');
      console.error('Daemon status error:', e);
    }
  }, []);

  // Fetch pending messages for the panel history view only.
  // Chat injection is handled by the WebSocket hook (proactive type messages)
  // to avoid duplicates.
  const fetchMessages = useCallback(async () => {
    if (!status?.running) return;
    try {
      const response = await apiFetch('/api/proactive/messages');
      if (response.ok) {
        const data = await response.json();
        if (data.messages?.length > 0) {
          const newMsgs = data.messages.filter((m: ProactiveMessage) => {
            const id = m.id || `${m.timestamp}-${(m.content || '').slice(0, 20)}`;
            if (seenIds.current.has(id)) return false;
            seenIds.current.add(id);
            return true;
          });
          if (newMsgs.length > 0) {
            setMessages(prev => [...prev, ...newMsgs].slice(-10));
          }
        }
      }
    } catch (e: any) {
      console.error('Messages fetch error:', e);
    }
  }, [status?.running]);

  // Poll for status and messages (15s - daemon status doesn't change rapidly)
  const fetchAll = useCallback(async () => {
    await fetchStatus();
    await fetchMessages();
  }, [fetchStatus, fetchMessages]);
  usePolling(fetchAll, 30000);

  // Control functions
  const daemonAction = async (action: string) => {
    setLoading(true);
    try {
      const res = await apiFetch(`/api/proactive/${action}`, { method: 'POST' });
      if (!res.ok) setError(`${action} failed: HTTP ${res.status}`);
      else setError(null);
      await fetchStatus();
    } catch {
      setError(`${action} failed: network error`);
    } finally {
      setLoading(false);
    }
  };

  const startDaemon = () => daemonAction('start');
  const stopDaemon = () => daemonAction('stop');
  const pauseDaemon = () => daemonAction('pause');
  const resumeDaemon = () => daemonAction('resume');

  const triggerDecision = async () => {
    setLoading(true);
    try {
      const response = await apiFetch('/api/proactive/decide', { method: 'POST' });
      if (response.ok) {
        const decision = await response.json();
        // Add to messages for visibility
        setMessages(prev => [...prev, {
          action: decision.action,
          content: decision.reasoning,
          priority: 'DECISION',
          timestamp: new Date().toISOString(),
          metadata: { confidence: decision.confidence },
        }].slice(-10));
      }
    } finally {
      setLoading(false);
    }
  };

  const sendTestMessage = async () => {
    setLoading(true);
    try {
      const actions = ['notify', 'suggest', 'remind', 'ask'];
      const randomAction = actions[Math.floor(Math.random() * actions.length)];

      const response = await apiFetch('/api/proactive/test-message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: randomAction }),
      });

      if (response.ok) {
        const data = await response.json();
        setMessages(prev => [...prev, {
          action: data.action,
          content: data.content,
          priority: 'TEST',
          timestamp: new Date().toISOString(),
          metadata: { test: true },
        }].slice(-10));
        // Test messages appear in the panel only; WS delivers them to chat
      }
    } finally {
      setLoading(false);
    }
  };

  if (error && !status) {
    return (
      <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
        <div className="text-xs text-red-400">{error}</div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="p-3 bg-chat-assistant/50 rounded-lg animate-pulse">
        <div className="h-4 bg-chat-border/30 rounded w-32 mb-2"></div>
        <div className="h-6 bg-chat-border/30 rounded w-full"></div>
      </div>
    );
  }

  const stateColor = status.running
    ? 'bg-green-500'
    : status.state === 'paused'
    ? 'bg-yellow-500'
    : 'bg-gray-500';

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={`w-2.5 h-2.5 rounded-full ${stateColor} ${status.running ? 'animate-pulse' : ''}`} />
          <span className="text-chat-text font-medium text-sm">
            Gateway Daemon
          </span>
          <span className="text-chat-text-secondary text-xs capitalize">
            {status.state}
          </span>
        </div>
        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4 text-chat-text-secondary" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 text-chat-text-secondary" />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3">
          {/* Control buttons */}
          <div className="flex gap-2">
            {!status.running ? (
              <button
                onClick={startDaemon}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded-lg border border-green-500/30 transition-colors disabled:opacity-50"
              >
                <PlayIcon className="w-3.5 h-3.5" />
                Start
              </button>
            ) : (
              <>
                {status.state === 'paused' ? (
                  <button
                    onClick={resumeDaemon}
                    disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded-lg border border-green-500/30 transition-colors disabled:opacity-50"
                  >
                    <PlayIcon className="w-3.5 h-3.5" />
                    Resume
                  </button>
                ) : (
                  <button
                    onClick={pauseDaemon}
                    disabled={loading}
                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-yellow-600/20 hover:bg-yellow-600/30 text-yellow-400 rounded-lg border border-yellow-500/30 transition-colors disabled:opacity-50"
                  >
                    <PauseIcon className="w-3.5 h-3.5" />
                    Pause
                  </button>
                )}
                <button
                  onClick={stopDaemon}
                  disabled={loading}
                  className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg border border-red-500/30 transition-colors disabled:opacity-50"
                >
                  <StopIcon className="w-3.5 h-3.5" />
                  Stop
                </button>
              </>
            )}
            <button
              onClick={triggerDecision}
              disabled={loading}
              className="flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 rounded-lg border border-purple-500/30 transition-colors disabled:opacity-50"
              title="Trigger proactive decision"
            >
              <BoltIcon className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Test message button — only works when daemon is running */}
          <button
            onClick={sendTestMessage}
            disabled={loading || !status?.running}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs bg-gradient-to-r from-pink-600/20 to-purple-600/20 hover:from-pink-600/30 hover:to-purple-600/30 text-pink-300 rounded-lg border border-pink-500/30 transition-all disabled:opacity-50"
          >
            <span>💬</span>
            <span>Send Test Proactive Message</span>
          </button>

          {/* Beliefs visualization */}
          {status.beliefs && (
            <div className="space-y-2">
              <div className="text-xs text-chat-text-secondary font-medium">Active Inference Beliefs</div>
              {Object.entries(status.beliefs).map(([key, value]) => (
                <div key={key} className="flex items-center gap-2">
                  <span className="text-xs text-chat-text-secondary w-24 truncate">
                    {BELIEF_LABELS[key] || key}
                  </span>
                  <div className="flex-1 h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${value * 100}%`,
                        backgroundColor: BELIEF_COLORS[key] || '#6b7280',
                      }}
                    />
                  </div>
                  <span className="text-xs text-chat-text-secondary w-8 text-right">
                    {Math.round(value * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Stats */}
          {status.stats && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              {status.stats.uptime_seconds !== undefined && (
                <div className="bg-chat-bg/30 rounded px-2 py-1.5">
                  <div className="text-chat-text-secondary">Uptime</div>
                  <div className="text-chat-text font-mono">
                    {Math.floor(status.stats.uptime_seconds / 60)}m {Math.floor(status.stats.uptime_seconds % 60)}s
                  </div>
                </div>
              )}
              <div className="bg-chat-bg/30 rounded px-2 py-1.5">
                <div className="text-chat-text-secondary">Decisions</div>
                <div className="text-chat-text font-mono">{status.stats.decisions_made || 0}</div>
              </div>
              <div className="bg-chat-bg/30 rounded px-2 py-1.5">
                <div className="text-chat-text-secondary">Events</div>
                <div className="text-chat-text font-mono">{status.stats.events_received || 0}</div>
              </div>
              <div className="bg-chat-bg/30 rounded px-2 py-1.5">
                <div className="text-chat-text-secondary">Messages</div>
                <div className="text-chat-text font-mono">{status.stats.messages_sent || 0}</div>
              </div>
            </div>
          )}

          {/* Recent messages/decisions */}
          {messages.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-xs text-chat-text-secondary font-medium">Recent Activity</div>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {messages.slice(-5).reverse().map((msg, idx) => (
                  <div
                    key={idx}
                    className="text-xs bg-chat-bg/30 rounded px-2 py-1.5 flex items-start gap-2"
                  >
                    <span className="text-purple-400 font-medium uppercase shrink-0">
                      {msg.action}
                    </span>
                    <span className="text-chat-text-secondary truncate">
                      {msg.content}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
