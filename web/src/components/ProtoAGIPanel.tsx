import { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ArrowPathIcon, CpuChipIcon, CheckCircleIcon, QuestionMarkCircleIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';

interface ProtoAGIStatus {
  enabled: boolean;
  mode: string;
  cycle_count: number;
  facts: number;
  beliefs: number;
  speculations: number;
  verifier_pass_rate: number;
  pending_confirmations: number;
  last_action: string | null;
}

export function ProtoAGIPanel() {
  const [status, setStatus] = useState<ProtoAGIStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      // Use absolute URL to bypass Vite proxy issues
      const res = await fetch('http://127.0.0.1:8000/api/proto-agi');
      console.log('[ProtoAGI] Fetch response:', res.status, res.ok);
      if (res.ok) {
        const data = await res.json();
        console.log('[ProtoAGI] Data received:', data);
        setStatus(data);
      } else {
        const errMsg = `Response not OK: ${res.status}`;
        console.error('[ProtoAGI]', errMsg);
        setError(errMsg);
      }
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : 'Unknown error';
      console.error('[ProtoAGI] Failed to fetch:', e);
      setError(errMsg);
    }
    setLoading(false);
  };

  usePolling(fetchStatus, 15000);

  if (!status?.enabled) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <CpuChipIcon className="w-5 h-5" />
          Proto-AGI Truth Spine
        </h3>
        <div className="text-chat-text-secondary text-sm">
          {loading ? 'Loading...' : error ? `Error: ${error}` : status ? `Disabled (enabled=${status.enabled})` : 'Proto-AGI not loaded'}
        </div>
        <div className="text-chat-text-secondary text-xs mt-1 opacity-50">
          Debug: status={status ? JSON.stringify(status).substring(0, 50) : 'null'}
        </div>
      </div>
    );
  }

  const modeColors: Record<string, string> = {
    idle: 'text-gray-400',
    assist: 'text-blue-400',
    operate: 'text-green-400',
    error: 'text-red-400',
  };

  const totalMemories = status.facts + status.beliefs + status.speculations;
  const factPercentage = totalMemories > 0 ? (status.facts / totalMemories) * 100 : 0;
  const passRate = status.verifier_pass_rate * 100;

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <CpuChipIcon className="w-5 h-5 text-purple-400" />
          Proto-AGI Truth Spine
        </h3>
        <button
          onClick={fetchStatus}
          className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
          disabled={loading}
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Mode & Cycle */}
      <div className="flex justify-between items-center mb-4">
        <div>
          <div className="text-xs text-chat-text-secondary mb-1">Operation Mode</div>
          <div className={`text-lg font-medium capitalize ${modeColors[status.mode] || 'text-chat-text'}`}>
            {status.mode}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-chat-text-secondary mb-1">Cycle</div>
          <div className="text-lg font-medium text-chat-text">{status.cycle_count}</div>
        </div>
      </div>

      {/* Memory Tiers - The Truth Spine */}
      <div className="mb-4">
        <div className="text-xs text-chat-text-secondary mb-2">Memory Tiers (Truth Spine)</div>
        <div className="grid grid-cols-3 gap-2">
          <div className="bg-green-900/30 border border-green-600/30 rounded p-2 text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <CheckCircleIcon className="w-4 h-4 text-green-400" />
            </div>
            <div className="text-xl font-bold text-green-400">{status.facts}</div>
            <div className="text-xs text-green-300">FACTS</div>
          </div>
          <div className="bg-yellow-900/30 border border-yellow-600/30 rounded p-2 text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <QuestionMarkCircleIcon className="w-4 h-4 text-yellow-400" />
            </div>
            <div className="text-xl font-bold text-yellow-400">{status.beliefs}</div>
            <div className="text-xs text-yellow-300">BELIEFS</div>
          </div>
          <div className="bg-red-900/30 border border-red-600/30 rounded p-2 text-center">
            <div className="flex items-center justify-center gap-1 mb-1">
              <ExclamationCircleIcon className="w-4 h-4 text-red-400" />
            </div>
            <div className="text-xl font-bold text-red-400">{status.speculations}</div>
            <div className="text-xs text-red-300">SPECULATION</div>
          </div>
        </div>
      </div>

      {/* Verifier Stats */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-chat-text-secondary mb-1">
          <span>Verifier Pass Rate</span>
          <span className={passRate >= 80 ? 'text-green-400' : passRate >= 50 ? 'text-yellow-400' : 'text-red-400'}>
            {passRate.toFixed(0)}%
          </span>
        </div>
        <div className="w-full bg-chat-assistant rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              passRate >= 80 ? 'bg-green-500' : passRate >= 50 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${passRate}%` }}
          />
        </div>
      </div>

      {/* Fact Ratio */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-chat-text-secondary mb-1">
          <span>Verified Facts Ratio</span>
          <span className="text-green-400">{factPercentage.toFixed(0)}%</span>
        </div>
        <div className="w-full bg-chat-assistant rounded-full h-2">
          <div
            className="h-2 rounded-full bg-green-500 transition-all duration-300"
            style={{ width: `${factPercentage}%` }}
          />
        </div>
      </div>

      {/* Pending Confirmations */}
      {status.pending_confirmations > 0 && (
        <div className="bg-orange-900/30 border border-orange-600/30 rounded p-2 mb-4">
          <div className="text-xs text-orange-300">
            {status.pending_confirmations} action(s) pending confirmation
          </div>
        </div>
      )}

      {/* Last Action */}
      {status.last_action && (
        <div className="text-xs">
          <div className="text-chat-text-secondary mb-1">Last Action</div>
          <div className="text-chat-text bg-chat-assistant rounded p-2 truncate">
            {status.last_action}
          </div>
        </div>
      )}

      {/* Truth Spine Principle */}
      <div className="mt-4 pt-3 border-t border-chat-border">
        <div className="text-xs text-chat-text-secondary italic text-center">
          "If you can't verify it with an artifact, it's SPECULATION"
        </div>
      </div>
    </div>
  );
}
