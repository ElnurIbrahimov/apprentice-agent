import { useEffect, useState, useCallback } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

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

export function ThinkingAboutTeaser() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [state, setState] = useState<ThinkingState | null>(null);
  const [teaser, setTeaser] = useState<TeaserData | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  // Fetch thinking state
  const fetchState = useCallback(async () => {
    try {
      const response = await fetch('/api/thinking/state');
      if (response.ok) {
        const data = await response.json();
        setState(data);

        // Trigger animation on new thoughts
        if (data.is_thinking && data.active_thoughts?.length > 0) {
          setIsAnimating(true);
          setTimeout(() => setIsAnimating(false), 1000);
        }
      }
    } catch (e) {
      // Silently ignore
    }
  }, []);

  // Fetch teaser preview
  const fetchTeaser = useCallback(async () => {
    try {
      const response = await fetch('/api/thinking/teaser');
      if (response.ok) {
        const data = await response.json();
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

  // Poll for updates
  useEffect(() => {
    fetchState();
    fetchTeaser();

    const stateInterval = setInterval(fetchState, 3000);
    const teaserInterval = setInterval(fetchTeaser, 2000);

    return () => {
      clearInterval(stateInterval);
      clearInterval(teaserInterval);
    };
  }, [fetchState, fetchTeaser]);

  // Generate a new thought manually
  const generateThought = async () => {
    try {
      await fetch('/api/thinking/generate?force=true', { method: 'POST' });
      fetchState();
      fetchTeaser();
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
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Thinking indicator */}
          <div className={`relative transition-all duration-300 ${isAnimating ? 'scale-110' : ''}`}>
            <span className="text-lg">{teaser?.icon || '💭'}</span>
            {hasThoughts && (
              <span className="absolute -top-1 -right-1 flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full rounded-full bg-purple-400 opacity-75 animate-ping"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-purple-500"></span>
              </span>
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Thinking About</span>
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
            A glimpse into AURA's current thought process and associations.
          </div>

          {/* Active thoughts */}
          {hasThoughts ? (
            <div className="space-y-2">
              {state?.active_thoughts.map((thought) => (
                <div
                  key={thought.id}
                  className={`rounded-lg p-2.5 border border-chat-border/20 transition-all duration-300 ${getThoughtBgColor(thought.type)}`}
                  style={{ opacity: Math.max(0.4, thought.intensity) }}
                >
                  {/* Thought header */}
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-sm">{thought.icon}</span>
                    <span className="text-xs text-chat-text-secondary capitalize">
                      {thought.type}
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
                  <div className="text-sm text-chat-text italic">
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
              ))}
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

          {/* Thought type legend */}
          <div className="pt-2 border-t border-chat-border/20">
            <div className="text-[10px] text-chat-text-secondary/50 mb-2">Thought Types:</div>
            <div className="grid grid-cols-2 gap-1 text-[10px]">
              {[
                { type: 'connecting', icon: '🔗', label: 'Connecting' },
                { type: 'questioning', icon: '❓', label: 'Questioning' },
                { type: 'recalling', icon: '💭', label: 'Recalling' },
                { type: 'analyzing', icon: '🔍', label: 'Analyzing' },
                { type: 'wondering', icon: '🤔', label: 'Wondering' },
                { type: 'formulating', icon: '✍️', label: 'Formulating' },
                { type: 'observing', icon: '👁️', label: 'Observing' },
              ].map((item) => (
                <div key={item.type} className="flex items-center gap-1 text-chat-text-secondary/60">
                  <span>{item.icon}</span>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
