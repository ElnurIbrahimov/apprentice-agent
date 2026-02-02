import { useState, useEffect } from 'react';
import type { GuardianStatus } from '../types';
import { ArrowPathIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';

export function GuardianPanel() {
  const [status, setStatus] = useState<GuardianStatus | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/guardian');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch Guardian status:', e);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!status?.enabled) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <ShieldCheckIcon className="w-5 h-5" />
          Metacognitive Guardian
        </h3>
        <div className="text-chat-text-secondary text-sm">Guardian not loaded</div>
      </div>
    );
  }

  const levelColors: Record<string, string> = {
    low: 'text-green-400',
    medium: 'text-yellow-400',
    high: 'text-orange-400',
    critical: 'text-red-400',
  };

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <ShieldCheckIcon className="w-5 h-5 text-blue-400" />
          Metacognitive Guardian
        </h3>
        <button
          onClick={fetchStatus}
          className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
          disabled={loading}
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Monitoring Level */}
      <div className="mb-4">
        <div className="text-xs text-chat-text-secondary mb-1">Monitoring Level</div>
        <div className={`text-lg font-medium capitalize ${levelColors[status.monitoring_level] || 'text-chat-text'}`}>
          {status.monitoring_level}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <div className="bg-chat-assistant rounded p-2 text-center">
          <div className="text-xl font-bold text-chat-text">{status.interventions}</div>
          <div className="text-xs text-chat-text-secondary">Interventions</div>
        </div>
        <div className="bg-chat-assistant rounded p-2 text-center">
          <div className="text-xl font-bold text-chat-text">{status.patterns_learned}</div>
          <div className="text-xs text-chat-text-secondary">Patterns</div>
        </div>
        <div className="bg-chat-assistant rounded p-2 text-center">
          <div className="text-xl font-bold text-chat-text">{status.session_predictions}</div>
          <div className="text-xs text-chat-text-secondary">Predictions</div>
        </div>
      </div>

      {/* Recent Predictions */}
      {status.recent_predictions.length > 0 && (
        <div>
          <div className="text-xs text-chat-text-secondary mb-2">Recent Predictions</div>
          <div className="space-y-1">
            {status.recent_predictions.slice(0, 3).map((pred, i) => (
              <div key={i} className="text-xs bg-chat-assistant rounded p-2">
                <div className="flex justify-between">
                  <span className="text-chat-text">{pred.type}</span>
                  <span className={pred.probability > 0.6 ? 'text-red-400' : 'text-yellow-400'}>
                    {Math.round(pred.probability * 100)}%
                  </span>
                </div>
                <div className="text-chat-text-secondary">{pred.action}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
