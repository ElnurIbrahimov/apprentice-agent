import { useState, useCallback, useRef } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface ActiveThought {
  id: string;
  type: string;
  icon: string;
  content: string;
  topics: string[];
  intensity: number;
  age_seconds: number;
  resolved: boolean;
  resolution: string | null;
  source: string;  // "brain", "engine", "memory", "dream", "emotion", "tool", "template"
  is_real: boolean;
}

interface ThinkingState {
  is_thinking: boolean;
  active_thoughts: ActiveThought[];
  thought_count: number;
  primary_thought: ActiveThought | null;
}

interface TeaserData {
  content: string;
  type: string;
  icon: string;
  intensity: number;
  topics: string[];
}

// Source labels and colors for different cognitive systems
const SOURCE_CONFIG: Record<string, { label: string; color: string; badge: string }> = {
  brain: { label: 'LLM', color: 'text-violet-400', badge: 'bg-violet-500/30 text-violet-300' },
  engine: { label: 'Engine', color: 'text-blue-400', badge: 'bg-blue-500/30 text-blue-300' },
  memory: { label: 'Memory', color: 'text-cyan-400', badge: 'bg-cyan-500/30 text-cyan-300' },
  dream: { label: 'Dream', color: 'text-indigo-400', badge: 'bg-indigo-500/30 text-indigo-300' },
  emotion: { label: 'ALMA', color: 'text-pink-400', badge: 'bg-pink-500/30 text-pink-300' },
  tool: { label: 'Tool', color: 'text-green-400', badge: 'bg-green-500/30 text-green-300' },
  agent: { label: 'Agent', color: 'text-amber-400', badge: 'bg-amber-500/30 text-amber-300' },
  service: { label: 'Service', color: 'text-orange-400', badge: 'bg-orange-500/30 text-orange-300' },
  template: { label: 'Idle', color: 'text-gray-500', badge: 'bg-gray-500/20 text-gray-400' },
};

export function ThinkingAboutTeaser() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [state, setState] = useState<ThinkingState | null>(null);
  const [teaser, setTeaser] = useState<TeaserData | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [hasRealThoughts, setHasRealThoughts] = useState(false);
  const lastThoughtCount = useRef(0);

  // Fetch both thinking state and teaser in a single polling callback
  const fetchAll = useCallback(async () => {
    try {
      const [stateRes, teaserRes] = await Promise.all([
        apiFetch('/api/thinking/state'),
        apiFetch('/api/thinking/teaser'),
      ]);

      if (stateRes.ok) {
        const data = await stateRes.json();
        setState(data);

        // Check for real (non-template) thoughts
        const realThoughts = data.active_thoughts?.filter((t: ActiveThought) => t.is_real) || [];
        setHasRealThoughts(realThoughts.length > 0);

        // Trigger animation on new thoughts
        const newCount = data.active_thoughts?.length ?? 0;
        if (newCount > lastThoughtCount.current) {
          setIsAnimating(true);
          setTimeout(() => setIsAnimating(false), 1200);
        }
        lastThoughtCount.current = newCount;
      }

      if (teaserRes.ok) {
        const data = await teaserRes.json();
        if (data.has_teaser) {
          setTeaser(data.teaser);
        } else {
          setTeaser(null);
        }
      }
    } catch (e) {
      // Silently ignore
    }
  }, []);

  // Poll every 5 seconds for real-time thought tracking
  // (was 30s — far too slow to catch real cognitive events)
  usePolling(fetchAll, 5000);

  // Generate a new thought manually
  const generateThought = async () => {
    try {
      await apiFetch('/api/thinking/generate?force=true', { method: 'POST' });
      fetchAll();
    } catch (e) {
      console.error('Failed to generate thought:', e);
    }
  };

  // Get color based on thought type
  const getThoughtColor = (type: string): string => {
    const colors: Record<string, string> = {
      connecting: 'from-purple-500 to-blue-500',
      questioning: 'from-yellow-500 to-orange-500',
      recalling: 'from-cyan-500 to-teal-500',
      analyzing: 'from-green-500 to-emerald-500',
      wondering: 'from-pink-500 to-rose-500',
      formulating: 'from-indigo-500 to-violet-500',
      observing: 'from-amber-500 to-yellow-500',
    };
    return colors[type] || 'from-gray-500 to-gray-600';
  };

  // Get background color for thought type
  const getThoughtBgColor = (type: string): string => {
    const colors: Record<string, string> = {
      connecting: 'bg-purple-500/20',
      questioning: 'bg-yellow-500/20',
      recalling: 'bg-cyan-500/20',
      analyzing: 'bg-green-500/20',
      wondering: 'bg-pink-500/20',
      formulating: 'bg-indigo-500/20',
      observing: 'bg-amber-500/20',
    };
    return colors[type] || 'bg-gray-500/20';
  };

  const hasThoughts = state?.is_thinking && (state?.active_thoughts?.length ?? 0) > 0;

  return (
    <div className={`bg-chat-assistant/60 rounded-xl border overflow-hidden transition-colors duration-500 ${
      hasRealThoughts
        ? 'border-purple-500/50 shadow-[0_0_12px_rgba(168,85,247,0.15)]'
        : 'border-chat-border/30'
    }`}>
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Thinking indicator */}
          <div className={`relative transition-all duration-300 ${isAnimating ? 'scale-110' : ''}`}>
            <span className="text-lg">{teaser?.icon || '💭'}</span>
            {hasRealThoughts && (
              <span className="absolute -top-1 -right-1 flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75 animate-ping"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-purple-500"></span>
              </span>
            )}
            {hasThoughts && !hasRealThoughts && (
              <span className="absolute -top-1 -right-1 flex h-2 w-2">
                <span className="relative inline-flex rounded-full h-2 w-2 bg-gray-500"></span>
              </span>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Thinking About</span>
              {hasRealThoughts && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-purple-500/20 text-purple-300 font-medium animate-pulse">
                  LIVE
                </span>
              )}
              {hasThoughts && (
                <span className="text-xs text-purple-400">
                  ({state?.thought_count})
                </span>
              )}
            </div>

            {/* Teaser preview when collapsed */}
            {!isExpanded && teaser && (
              <div className="text-xs text-chat-text-secondary/70 truncate mt-0.5 italic">
                {teaser.content}
              </div>
            )}
            {!isExpanded && !teaser && (
              <div className="text-xs text-chat-text-secondary/50 truncate mt-0.5">
                No active thoughts...
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
            Real-time cognitive activity from AURA's processing systems.
          </div>

          {/* Active thoughts */}
          {hasThoughts ? (
            <div className="space-y-2">
              {state?.active_thoughts.map((thought) => {
                const sourceConfig = SOURCE_CONFIG[thought.source] || SOURCE_CONFIG.template;
                return (
                  <div
                    key={thought.id}
                    className={`rounded-lg p-2.5 border transition-all duration-300 ${
                      thought.is_real
                        ? `${getThoughtBgColor(thought.type)} border-purple-500/20`
                        : 'bg-gray-500/10 border-chat-border/10'
                    }`}
                    style={{ opacity: Math.max(0.4, thought.intensity) }}
                  >
                    {/* Thought header */}
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-sm">{thought.icon}</span>
                      <span className="text-xs text-chat-text-secondary capitalize">
                        {thought.type}
                      </span>
                      {/* Source badge */}
                      <span className={`text-[9px] px-1.5 py-0.5 rounded-full font-medium ${sourceConfig.badge}`}>
                        {sourceConfig.label}
                      </span>
                      {/* Intensity bar */}
                      <div className="flex-1 h-1 bg-chat-border/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${getThoughtColor(thought.type)} transition-all duration-500`}
                          style={{ width: `${thought.intensity * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Thought content */}
                    <div className={`text-sm italic ${thought.is_real ? 'text-chat-text' : 'text-chat-text-secondary/60'}`}>
                      "{thought.content}"
                    </div>

                    {/* Topics */}
                    {thought.topics.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {thought.topics.slice(0, 3).map((topic, i) => (
                          <span
                            key={i}
                            className="text-[10px] px-1.5 py-0.5 bg-chat-bg/40 text-chat-text-secondary rounded"
                          >
                            {topic}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Age indicator */}
                    <div className="text-[10px] text-chat-text-secondary/50 mt-1.5">
                      {thought.age_seconds < 5
                        ? 'just now'
                        : thought.age_seconds < 60
                        ? `${Math.round(thought.age_seconds)}s ago`
                        : `${Math.round(thought.age_seconds / 60)}m ago`}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-4 text-chat-text-secondary/50 text-sm">
              <span className="text-2xl mb-2 block opacity-50">🧘</span>
              Mind is quiet...
            </div>
          )}

          {/* Generate thought button */}
          <button
            onClick={generateThought}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-xs bg-gradient-to-r from-purple-600/20 to-indigo-600/20 hover:from-purple-600/30 hover:to-indigo-600/30 text-purple-300 rounded-lg border border-purple-500/30 transition-all"
          >
            <span>💭</span>
            <span>Spark a Thought</span>
          </button>

          {/* Source legend */}
          <div className="pt-2 border-t border-chat-border/20">
            <div className="text-[10px] text-chat-text-secondary/50 mb-2">Sources:</div>
            <div className="grid grid-cols-3 gap-1 text-[10px]">
              {Object.entries(SOURCE_CONFIG)
                .filter(([key]) => key !== 'template')
                .map(([key, config]) => (
                  <div key={key} className={`flex items-center gap-1 ${config.color}`}>
                    <span className="w-1.5 h-1.5 rounded-full bg-current opacity-60"></span>
                    <span>{config.label}</span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
