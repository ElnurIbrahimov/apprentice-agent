import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

interface Thought {
  type: string;
  content: string;
  timestamp: number;
}

const THOUGHT_ICONS: Record<string, string> = {
  observation: '👁️',
  reflection: '💭',
  inference: '🧠',
  memory: '📝',
  planning: '📋',
  emotion: '💫',
  curiosity: '🔍',
  default: '✨',
};

const THOUGHT_COLORS: Record<string, string> = {
  observation: 'text-blue-300',
  reflection: 'text-purple-300',
  inference: 'text-green-300',
  memory: 'text-amber-300',
  planning: 'text-cyan-300',
  emotion: 'text-pink-300',
  curiosity: 'text-indigo-300',
  default: 'text-chat-text-secondary',
};

export function InnerThoughtsPanel() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [currentThought, setCurrentThought] = useState<Thought | null>(null);

  // Fetch thoughts from API
  const fetchThoughts = useCallback(async () => {
    try {
      const response = await apiFetch('/api/introspection/recent?limit=5');
      if (response.ok) {
        const data = await response.json();
        const items = data.results || data.thoughts || [];
        if (items.length > 0) {
          setThoughts(items.slice(-5)); // Keep last 5
          setCurrentThought(items[items.length - 1]);
        }
      }
    } catch (e: any) {
      // Silently ignore - not critical
    }
  }, []);

  // Poll for thoughts using staggered polling (30s)
  usePolling(fetchThoughts, 30000);

  const icon = currentThought ? THOUGHT_ICONS[currentThought.type] || THOUGHT_ICONS.default : '💭';
  const colorClass = currentThought ? THOUGHT_COLORS[currentThought.type] || THOUGHT_COLORS.default : 'text-chat-text-secondary';

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="flex items-center justify-center w-8 h-8 rounded-lg bg-purple-500/15 text-base shrink-0">{icon}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Inner Thoughts</span>
              {currentThought && (
                <span className={`text-xs px-1.5 py-0.5 rounded bg-chat-bg/50 capitalize ${colorClass}`}>
                  {currentThought.type}
                </span>
              )}
            </div>
            {currentThought && !isExpanded && (
              <div className="text-xs text-chat-text-secondary truncate mt-0.5 italic">
                "{currentThought.content}"
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
          {/* Idle state */}
          {!currentThought && (
            <div className="text-chat-text-secondary/60 text-xs italic px-1">
              Processing quietly... thoughts surface during active reasoning.
            </div>
          )}

          {/* Current thought highlight */}
          {currentThought && (
            <div className="bg-gradient-to-r from-purple-900/30 to-transparent rounded-lg p-2.5 border-l-2 border-purple-500">
              <div className="flex items-center gap-2 mb-1">
                <span>{THOUGHT_ICONS[currentThought.type] || THOUGHT_ICONS.default}</span>
                <span className={`text-xs font-medium capitalize ${colorClass}`}>
                  {currentThought.type}
                </span>
                <span className="text-xs text-chat-text-secondary/50">now</span>
              </div>
              <p className="text-sm text-chat-text italic">"{currentThought.content}"</p>
            </div>
          )}

          {/* Recent thoughts stream */}
          {thoughts.length > 1 && (
            <div className="space-y-1.5">
              <div className="text-xs text-chat-text-secondary font-medium">Recent</div>
              {thoughts.slice(0, -1).reverse().map((thought, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-2 text-xs text-chat-text-secondary/70 pl-2"
                >
                  <span className="shrink-0">{THOUGHT_ICONS[thought.type] || THOUGHT_ICONS.default}</span>
                  <span className="truncate italic">"{thought.content}"</span>
                </div>
              ))}
            </div>
          )}

          {/* Thought activity indicator */}
          <div className="flex items-center gap-2 pt-1">
            <div className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse"></span>
              <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" style={{ animationDelay: '0.2s' }}></span>
              <span className="w-1.5 h-1.5 rounded-full bg-purple-300 animate-pulse" style={{ animationDelay: '0.4s' }}></span>
            </div>
            <span className="text-xs text-chat-text-secondary/50">Continuously processing...</span>
          </div>
        </div>
      )}
    </div>
  );
}
