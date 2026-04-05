import { useState } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import type { StatusResponse } from '../types';

interface SystemStatsPanelProps {
  status: StatusResponse | null;
}

export function SystemStatsPanel({ status }: SystemStatsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!status) {
    return (
      <div className="p-3 bg-chat-assistant/50 rounded-xl animate-pulse">
        <div className="h-4 bg-chat-border/30 rounded w-24 mb-2"></div>
        <div className="h-6 bg-chat-border/30 rounded w-full"></div>
      </div>
    );
  }

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-lg">📊</span>
            <span className="text-chat-text font-medium text-sm">Stats</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-chat-text-secondary">
            <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
              {status.memory_count} memories
            </span>
            <span className="px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300">
              {status.query_count} queries
            </span>
          </div>
        </div>
        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4 text-chat-text-secondary" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 text-chat-text-secondary" />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-2">
          {/* Compact stats grid */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-chat-bg/30 rounded-lg p-2">
              <div className="text-chat-text-secondary mb-0.5">Default Model</div>
              <div className="text-chat-text font-medium truncate">{status.model}</div>
            </div>
            <div className="bg-chat-bg/30 rounded-lg p-2">
              <div className="text-chat-text-secondary mb-0.5">Last Used</div>
              <div className="text-chat-text font-medium truncate">
                {status.last_model_used || '-'}
              </div>
            </div>
          </div>

          {/* AURA status bar */}
          <div className="flex items-center justify-between bg-chat-bg/30 rounded-lg p-2">
            <span className="text-xs text-chat-text-secondary">AURA System</span>
            <span className={`text-xs font-medium px-2 py-0.5 rounded ${
              status.aura_enabled
                ? 'bg-green-500/20 text-green-400'
                : 'bg-gray-500/20 text-gray-400'
            }`}>
              {status.aura_enabled ? 'Active' : 'Inactive'}
            </span>
          </div>

          {/* Memory & Query bars */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="text-xs text-chat-text-secondary w-16">Memories</span>
              <div className="flex-1 h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, (status.memory_count / Math.max(status.memory_count, 50)) * 100)}%` }}
                />
              </div>
              <span className="text-xs text-chat-text font-mono w-10 text-right">{status.memory_count}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-chat-text-secondary w-16">Queries</span>
              <div className="flex-1 h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(100, (status.query_count / Math.max(status.query_count, 20)) * 100)}%` }}
                />
              </div>
              <span className="text-xs text-chat-text font-mono w-10 text-right">{status.query_count}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
