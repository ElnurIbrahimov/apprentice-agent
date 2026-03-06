import { useState, useCallback, useRef } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface RecallEvent {
  id: string;
  source: string;
  count: number;
  query: string;
  memories: string[];
  timestamp: string;
}

interface RecallStatus {
  is_active: boolean;
  last_recall: string | null;
  recent_count: number;
  recent_events: RecallEvent[];
}

const SOURCE_ICONS: Record<string, string> = {
  amem: '🧠',
  rag: '📚',
  kg: '🕸️',
  retriever: '🔍',
};

const SOURCE_COLORS: Record<string, string> = {
  amem: 'text-purple-300 bg-purple-500/20',
  rag: 'text-blue-300 bg-blue-500/20',
  kg: 'text-cyan-300 bg-cyan-500/20',
  retriever: 'text-amber-300 bg-amber-500/20',
};

const SOURCE_LABELS: Record<string, string> = {
  amem: 'A-MEM',
  rag: 'RAG',
  kg: 'Knowledge Graph',
  retriever: 'Retriever',
};

export function MemoryRecallIndicator() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [status, setStatus] = useState<RecallStatus | null>(null);
  const [isGlowing, setIsGlowing] = useState(false);
  const [pulseKey, setPulseKey] = useState(0);
  const prevStatusRef = useRef<{ is_active?: boolean; last_recall?: string | null }>({});

  // Fetch recall status
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/memory/recalls/status');
      if (response.ok) {
        const data = await response.json();

        // Check if we have new activity
        const prev = prevStatusRef.current;
        if (data.is_active && (!prev.is_active || data.last_recall !== prev.last_recall)) {
          // Trigger glow animation
          setIsGlowing(true);
          setPulseKey(prev => prev + 1);
          setTimeout(() => setIsGlowing(false), 3000);
        }

        prevStatusRef.current = { is_active: data.is_active, last_recall: data.last_recall };
        setStatus(data);
      }
    } catch (e) {
      // Silently ignore - not critical
    }
  }, []);

  // Poll for status (30s - memory recalls don't happen that often)
  usePolling(fetchStatus, 30000);

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();

    if (diff < 60000) return 'just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    return date.toLocaleTimeString();
  };

  const recentEvent = status?.recent_events?.[0];

  return (
    <div className={`bg-chat-assistant/60 rounded-xl border overflow-hidden transition-all duration-500 ${
      isGlowing
        ? 'border-purple-500/70 shadow-lg shadow-purple-500/20'
        : 'border-chat-border/30'
    }`}>
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Memory icon with glow effect */}
          <div className="relative">
            <span className="text-lg">🧠</span>
            {isGlowing && (
              <span
                key={pulseKey}
                className="absolute inset-0 rounded-full bg-purple-500/50 animate-ping"
              />
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Memory Recall</span>
              {status?.is_active && (
                <span className="flex h-2 w-2">
                  <span className="absolute inline-flex h-2 w-2 rounded-full bg-purple-400 opacity-75 animate-ping"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
                </span>
              )}
            </div>

            {/* Current status line */}
            {recentEvent && !isExpanded && (
              <div className="text-xs text-chat-text-secondary truncate mt-0.5">
                {SOURCE_ICONS[recentEvent.source]} {recentEvent.count} memories {formatTime(recentEvent.timestamp)}
              </div>
            )}
          </div>
        </div>

        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4 text-chat-text-secondary shrink-0" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 text-chat-text-secondary shrink-0" />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-2">
          {/* Activity indicator */}
          {status?.is_active ? (
            <div className="flex items-center gap-2 bg-gradient-to-r from-purple-900/40 to-transparent rounded-lg p-2 border-l-2 border-purple-500">
              <div className="flex gap-0.5">
                <span className="w-1.5 h-4 rounded-full bg-purple-500 animate-pulse"></span>
                <span className="w-1.5 h-3 rounded-full bg-purple-400 animate-pulse" style={{ animationDelay: '0.1s' }}></span>
                <span className="w-1.5 h-5 rounded-full bg-purple-300 animate-pulse" style={{ animationDelay: '0.2s' }}></span>
              </div>
              <span className="text-xs text-purple-300 font-medium">Retrieving memories...</span>
            </div>
          ) : (
            <div className="text-xs text-chat-text-secondary/70 px-2">
              {status?.recent_count ? 'Memory system ready' : 'Ready — recall events appear during conversation.'}
            </div>
          )}

          {/* Recent recalls */}
          {status?.recent_events && status.recent_events.length > 0 && (
            <div className="space-y-1.5">
              <div className="text-xs text-chat-text-secondary font-medium">Recent Recalls</div>
              {status.recent_events.slice(0, 4).map((event, idx) => (
                <div
                  key={event.id || idx}
                  className={`bg-chat-bg/30 rounded-lg p-2 ${idx === 0 && isGlowing ? 'ring-1 ring-purple-500/50' : ''}`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <span className="text-sm">{SOURCE_ICONS[event.source] || '📝'}</span>
                      <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${SOURCE_COLORS[event.source] || 'text-chat-text bg-chat-bg/50'}`}>
                        {SOURCE_LABELS[event.source] || event.source}
                      </span>
                      <span className="text-xs text-chat-text-secondary">
                        {event.count} {event.count === 1 ? 'memory' : 'memories'}
                      </span>
                    </div>
                    <span className="text-xs text-chat-text-secondary/50">
                      {formatTime(event.timestamp)}
                    </span>
                  </div>

                  {/* Show memory previews */}
                  {event.memories.length > 0 && (
                    <div className="space-y-1 mt-1.5">
                      {event.memories.slice(0, 2).map((memory, mIdx) => (
                        <div key={mIdx} className="text-xs text-chat-text-secondary/80 truncate pl-6 italic">
                          "{memory.substring(0, 60)}..."
                        </div>
                      ))}
                      {event.memories.length > 2 && (
                        <div className="text-xs text-chat-text-secondary/50 pl-6">
                          +{event.memories.length - 2} more
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Memory sources legend */}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {Object.entries(SOURCE_LABELS).map(([key, label]) => (
              <div key={key} className="flex items-center gap-1 text-xs text-chat-text-secondary/60">
                <span>{SOURCE_ICONS[key]}</span>
                <span>{label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
