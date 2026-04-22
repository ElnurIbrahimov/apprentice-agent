import { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { NeuroDreamStatus } from '../types';
import { ArrowPathIcon, MoonIcon, SunIcon } from '@heroicons/react/24/outline';
import { formatPercent } from '../utils/format';
import { apiFetch } from '../utils/apiFetch';

const PHASE_ICONS: Record<string, string> = {
  light: '🌙',
  deep: '💤',
  rem: '🌈',
};

export function NeuroDreamPanel() {
  const [status, setStatus] = useState<NeuroDreamStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const res = await apiFetch('/api/neurodream');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        setFetchError(null);
      }
    } catch (e) {
      setFetchError('Failed to connect to NeuroDream');
      console.error('Failed to fetch NeuroDream status:', e);
    }
    setLoading(false);
  };

  usePolling(fetchStatus, 15000);

  const triggerSleep = async () => {
    setActionLoading(true);
    try {
      await apiFetch('/api/neurodream/sleep', { method: 'POST' });
      await fetchStatus();
    } catch (e) {
      console.error('Failed to trigger sleep:', e);
    }
    setActionLoading(false);
  };

  const triggerWake = async () => {
    setActionLoading(true);
    try {
      await apiFetch('/api/neurodream/wake', { method: 'POST' });
      await fetchStatus();
    } catch (e) {
      console.error('Failed to wake up:', e);
    }
    setActionLoading(false);
  };

  if (fetchError) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <div className="text-xs text-amber-400">{fetchError}</div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4 animate-pulse">
        <div className="h-4 bg-chat-border/30 rounded w-28 mb-2"></div>
        <div className="h-6 bg-chat-border/30 rounded w-full"></div>
      </div>
    );
  }

  if (status.loading) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <MoonIcon className="w-4 h-4" /> NeuroDream
        </h3>
        <p className="text-chat-text-secondary text-sm">
          Agent initializing... check back in a moment.
        </p>
      </div>
    );
  }

  if (!status.enabled) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <MoonIcon className="w-4 h-4" /> NeuroDream
        </h3>
        <p className="text-chat-text-secondary text-sm">NeuroDream engine unavailable.</p>
      </div>
    );
  }

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <MoonIcon className="w-5 h-5 text-indigo-400" />
          NeuroDream
          {status.is_sleeping && (
            <span className="text-xs bg-indigo-600 px-2 py-0.5 rounded-full animate-pulse">
              Sleeping
            </span>
          )}
        </h3>
        <button
          onClick={fetchStatus}
          className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
          disabled={loading}
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Current Phase */}
      {status.is_sleeping && status.current_phase && (
        <div className="mb-4 p-3 bg-indigo-900/30 rounded-lg border border-indigo-500/30">
          <div className="text-xs text-indigo-300 mb-1">Current Phase</div>
          <div className="text-lg text-indigo-200 capitalize flex items-center gap-2">
            {PHASE_ICONS[status.current_phase] || '💭'}
            {status.current_phase}
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="bg-chat-assistant rounded p-2 text-center">
          <div className="text-xl font-bold text-chat-text">{status.total_sessions}</div>
          <div className="text-xs text-chat-text-secondary">Sleep Sessions</div>
        </div>
        <div className="bg-chat-assistant rounded p-2 text-center">
          <div className="text-xl font-bold text-chat-text">{status.total_insights}</div>
          <div className="text-xs text-chat-text-secondary">Insights</div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={triggerSleep}
          disabled={actionLoading || status.is_sleeping}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded text-sm"
        >
          <MoonIcon className="w-4 h-4" />
          Sleep Now
        </button>
        <button
          onClick={triggerWake}
          disabled={actionLoading || !status.is_sleeping}
          className="flex-1 flex items-center justify-center gap-2 px-3 py-2 bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded text-sm"
        >
          <SunIcon className="w-4 h-4" />
          Wake Up
        </button>
      </div>

      {/* Dream Journal */}
      {status.dream_journal.length > 0 && (
        <div className="mb-4">
          <div className="text-xs text-chat-text-secondary mb-2">Dream Journal</div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {status.dream_journal.map((entry, i) => (
              <div key={i} className="text-xs bg-chat-assistant rounded p-2">
                <div className="flex items-center gap-1 mb-1">
                  <span>{PHASE_ICONS[entry.phase] || '💭'}</span>
                  <span className="text-chat-text uppercase">{entry.phase}</span>
                  <span className="text-chat-text-secondary">
                    {entry.timestamp?.slice(0, 16)}
                  </span>
                </div>
                <div className="text-chat-text-secondary">
                  {entry.content?.slice(0, 80)}...
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Insights */}
      {status.insights.length > 0 && (
        <div>
          <div className="text-xs text-chat-text-secondary mb-2">Insights</div>
          <div className="space-y-1 max-h-32 overflow-y-auto">
            {status.insights.map((insight, i) => (
              <div key={i} className="text-xs bg-chat-assistant rounded p-2">
                <div className="flex justify-between mb-1">
                  <span className="text-chat-text capitalize">{insight.type}</span>
                  <span className="text-chat-text-secondary">
                    {typeof insight.confidence === 'number' && insight.confidence <= 1
                      ? formatPercent(insight.confidence)
                      : `${Math.round(insight.confidence || 0)}%`}
                  </span>
                </div>
                <div className="text-chat-text-secondary">
                  {insight.content?.slice(0, 60)}...
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
