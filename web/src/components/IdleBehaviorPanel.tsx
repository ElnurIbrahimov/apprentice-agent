import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface IdleBehavior {
  type: string;
  intensity: string;
  status_message: string;
  breath_rate: number;
  attention_drift: number;
  age_seconds: number;
  duration_hint: number;
}

interface IdleState {
  is_idle: boolean;
  idle_duration: number;
  current_behavior: IdleBehavior | null;
  time_period: string;
}

interface IdleStats {
  behaviors_generated: number;
  total_idle_time: number;
  favorite_behavior: string | null;
  current_idle_duration: number;
}

// Behavior type icons and colors
const BEHAVIOR_CONFIG: Record<string, { icon: string; color: string; bgColor: string }> = {
  observing: { icon: '👁️', color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
  reflecting: { icon: '🪞', color: 'text-purple-400', bgColor: 'bg-purple-500/20' },
  anticipating: { icon: '👂', color: 'text-green-400', bgColor: 'bg-green-500/20' },
  drifting: { icon: '☁️', color: 'text-gray-400', bgColor: 'bg-gray-500/20' },
  focusing: { icon: '🎯', color: 'text-amber-400', bgColor: 'bg-amber-500/20' },
  relaxing: { icon: '🧘', color: 'text-teal-400', bgColor: 'bg-teal-500/20' },
  curious: { icon: '🔍', color: 'text-pink-400', bgColor: 'bg-pink-500/20' },
  processing: { icon: '⚙️', color: 'text-indigo-400', bgColor: 'bg-indigo-500/20' },
};

// Intensity indicators
const INTENSITY_CONFIG: Record<string, { label: string; bars: number }> = {
  deep: { label: 'Deep', bars: 1 },
  light: { label: 'Light', bars: 2 },
  alert: { label: 'Alert', bars: 3 },
  restless: { label: 'Restless', bars: 4 },
};

export function IdleBehaviorPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [state, setState] = useState<IdleState | null>(null);
  const [stats, setStats] = useState<IdleStats | null>(null);

  // Fetch both idle state and stats in a single polling callback
  const fetchAll = useCallback(async () => {
    try {
      const [stateRes, statsRes] = await Promise.all([
        apiFetch('/api/idle/state'),
        apiFetch('/api/idle/stats'),
      ]);

      if (stateRes.ok) {
        const data = await stateRes.json();
        setState(data);
      }
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }
    } catch (e: any) {
      // Silently ignore
    }
  }, []);

  // Poll every 10s (was 30s — now shows real system state, not just cosmetic)
  usePolling(fetchAll, 10000);

  // Format duration
  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };

  // Get time period emoji
  const getTimePeriodEmoji = (period: string): string => {
    const emojis: Record<string, string> = {
      morning: '🌅',
      afternoon: '☀️',
      evening: '🌆',
      night: '🌙',
    };
    return emojis[period] || '🕐';
  };

  const behavior = state?.current_behavior;
  const behaviorConfig = behavior ? BEHAVIOR_CONFIG[behavior.type] : null;
  const intensityConfig = behavior ? INTENSITY_CONFIG[behavior.intensity] : null;

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Behavior indicator */}
          <div className="relative">
            <span className="text-lg">
              {behaviorConfig?.icon || '💤'}
            </span>
            {state?.is_idle && (
              <span className="absolute -bottom-0.5 -right-0.5 flex h-2 w-2">
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500/70"></span>
              </span>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Ambient State</span>
              {state?.time_period && (
                <span className="text-xs opacity-60">
                  {getTimePeriodEmoji(state.time_period)}
                </span>
              )}
            </div>

            {/* Status preview when collapsed */}
            {!isExpanded && (
              <div className="text-xs text-chat-text-secondary/70 truncate mt-0.5 italic">
                {behavior ? behavior.status_message : 'Active...'}
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
            Real-time ambient state from AURA's cognitive systems.
          </div>

          {/* Current behavior display */}
          {behavior && behaviorConfig && (
            <div className={`rounded-lg p-3 ${behaviorConfig.bgColor} border border-chat-border/20`}>
              <div className="flex items-start gap-3">
                <span className="text-2xl">{behaviorConfig.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-sm font-medium capitalize ${behaviorConfig.color}`}>
                      {behavior.type}
                    </span>
                    {intensityConfig && (
                      <div className="flex gap-0.5">
                        {[...Array(4)].map((_, i) => (
                          <div
                            key={i}
                            className={`w-1 h-3 rounded-full transition-colors ${
                              i < intensityConfig.bars
                                ? `${behaviorConfig.color.replace('text-', 'bg-')} opacity-80`
                                : 'bg-chat-border/30'
                            }`}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                  <div className={`text-sm italic ${
                    behavior.status_message.includes(':')
                      ? 'text-chat-text'
                      : 'text-chat-text-secondary/70'
                  }`}>
                    "{behavior.status_message}"
                    {behavior.status_message.includes(':') && (
                      <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded bg-green-500/20 text-green-400 not-italic font-medium">
                        REAL
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-2 text-[10px] text-chat-text-secondary/60">
                    <span>Breath: {behavior.breath_rate.toFixed(1)}x</span>
                    <span>•</span>
                    <span>Age: {Math.round(behavior.age_seconds)}s</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Not idle state */}
          {!state?.is_idle && (
            <div className="text-center py-4 text-chat-text-secondary/50 text-sm">
              <span className="text-2xl mb-2 block">⚡</span>
              Actively engaged
            </div>
          )}

          {/* Idle duration indicator */}
          {state?.is_idle && (
            <div className="bg-chat-bg/30 rounded-lg p-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-chat-text-secondary/60">Idle for</span>
                <span className="text-chat-text font-mono">
                  {formatDuration(state.idle_duration)}
                </span>
              </div>
              <div className="mt-2 h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-green-500 via-blue-500 to-purple-500 transition-all duration-1000"
                  style={{
                    width: `${Math.min(100, (state.idle_duration / 300) * 100)}%`,
                  }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-chat-text-secondary/40 mt-1">
                <span>Active</span>
                <span>Deep Idle</span>
              </div>
            </div>
          )}

          {/* Stats */}
          {stats && (
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-chat-bg/30 rounded-lg p-2 text-center">
                <div className="text-chat-text font-mono">{stats.behaviors_generated}</div>
                <div className="text-chat-text-secondary/60 text-[10px]">Behaviors</div>
              </div>
              <div className="bg-chat-bg/30 rounded-lg p-2 text-center">
                <div className="text-chat-text font-mono">
                  {formatDuration(stats.total_idle_time)}
                </div>
                <div className="text-chat-text-secondary/60 text-[10px]">Total Idle</div>
              </div>
            </div>
          )}

          {/* Behavior type legend */}
          <div className="pt-2 border-t border-chat-border/20">
            <div className="text-[10px] text-chat-text-secondary/50 mb-2">Behavior Types:</div>
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {Object.entries(BEHAVIOR_CONFIG).map(([type, config]) => (
                <div
                  key={type}
                  className={`flex items-center gap-1 ${
                    behavior?.type === type ? config.color : 'text-chat-text-secondary/60'
                  }`}
                >
                  <span>{config.icon}</span>
                  <span className="capitalize">{type}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
