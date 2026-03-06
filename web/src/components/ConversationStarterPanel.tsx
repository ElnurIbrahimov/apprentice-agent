import { useState, useCallback } from 'react';
import { ChevronDownIcon, ChevronUpIcon, ChatBubbleLeftEllipsisIcon } from '@heroicons/react/24/outline';
import { usePolling } from '../hooks/usePolling';

interface StarterStats {
  total_generated: number;
  total_delivered: number;
  total_dismissed: number;
  pending: boolean;
  starters_this_hour: number;
  max_per_hour: number;
}

export function ConversationStarterPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [stats, setStats] = useState<StarterStats | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGenerated, setLastGenerated] = useState<string | null>(null);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const response = await fetch('/api/conversation/starter/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (e) {
      // Silently ignore
    }
  }, []);

  // Poll for stats
  usePolling(fetchStats, 10000);

  // Trigger a conversation starter
  const triggerStarter = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('/api/conversation/starter/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: true }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.generated && data.starter) {
          setLastGenerated(data.starter.content);
          fetchStats();
        }
      }
    } catch (e) {
      console.error('Failed to generate starter:', e);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-lg">💬</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Spontaneous</span>
              {stats?.pending && (
                <span className="flex h-2 w-2">
                  <span className="absolute inline-flex h-2 w-2 rounded-full bg-green-400 opacity-75 animate-ping"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
              )}
            </div>
            {!isExpanded && stats && (
              <div className="text-xs text-chat-text-secondary truncate mt-0.5">
                {stats.starters_this_hour}/{stats.max_per_hour} this hour
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
        <div className="px-3 pb-3 space-y-3">
          {/* Description */}
          <div className="text-xs text-chat-text-secondary/70">
            {stats && stats.total_generated === 0
              ? 'Auto-generates topics after ~60s of idle time.'
              : 'AURA can spontaneously start conversations based on context, time, and your focus.'}
          </div>

          {/* Stats */}
          {stats && (
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="bg-chat-bg/30 rounded-lg p-2 text-center">
                <div className="text-chat-text font-mono text-lg">{stats.total_generated}</div>
                <div className="text-chat-text-secondary/60">Generated</div>
              </div>
              <div className="bg-chat-bg/30 rounded-lg p-2 text-center">
                <div className="text-chat-text font-mono text-lg">{stats.total_delivered}</div>
                <div className="text-chat-text-secondary/60">Delivered</div>
              </div>
              <div className="bg-chat-bg/30 rounded-lg p-2 text-center">
                <div className="text-chat-text font-mono text-lg">{stats.starters_this_hour}</div>
                <div className="text-chat-text-secondary/60">This Hour</div>
              </div>
            </div>
          )}

          {/* Rate limit indicator */}
          {stats && (
            <div className="space-y-1">
              <div className="flex justify-between text-[10px] text-chat-text-secondary/60">
                <span>Hourly Limit</span>
                <span>{stats.starters_this_hour}/{stats.max_per_hour}</span>
              </div>
              <div className="h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-cyan-500 to-blue-500"
                  style={{
                    width: `${(stats.starters_this_hour / stats.max_per_hour) * 100}%`,
                  }}
                />
              </div>
            </div>
          )}

          {/* Trigger button */}
          <button
            onClick={triggerStarter}
            disabled={isGenerating}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs bg-gradient-to-r from-cyan-600/20 to-blue-600/20 hover:from-cyan-600/30 hover:to-blue-600/30 text-cyan-300 rounded-lg border border-cyan-500/30 transition-all disabled:opacity-50"
          >
            <ChatBubbleLeftEllipsisIcon className="w-4 h-4" />
            <span>{isGenerating ? 'Generating...' : 'Trigger Conversation'}</span>
          </button>

          {/* Last generated preview */}
          {lastGenerated && (
            <div className="bg-chat-bg/30 rounded-lg p-2">
              <div className="text-[10px] text-chat-text-secondary/60 mb-1">Last Generated:</div>
              <div className="text-xs text-chat-text italic">"{lastGenerated}"</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
