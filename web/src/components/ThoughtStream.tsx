import { useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { usePolling } from '../hooks/usePolling';
import type { Thought } from '../types';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

const THOUGHT_ICONS: Record<string, string> = {
  perceive: '🔍',
  recall: '💾',
  reason: '🧠',
  decide: '⚡',
  execute: '🔧',
  reflect: '🪞',
  uncertain: '❓',
  eureka: '💡',
};

const THOUGHT_COLORS: Record<string, string> = {
  perceive: 'border-blue-500',
  recall: 'border-purple-500',
  reason: 'border-yellow-500',
  decide: 'border-green-500',
  execute: 'border-orange-500',
  reflect: 'border-pink-500',
  uncertain: 'border-gray-500',
  eureka: 'border-yellow-400',
};

export function ThoughtStream() {
  const [thoughts, setThoughts] = useState<Thought[]>([]);
  const [loading, setLoading] = useState(false);
  const [thoughtCount, setThoughtCount] = useState(0);
  const { isLoading: chatLoading } = useChatStore();

  const fetchThoughts = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/thoughts');
      if (res.ok) {
        const data = await res.json();
        setThoughts(data.thoughts || []);
        setThoughtCount(data.thought_count || 0);
      }
    } catch {
      // Ignore - thought stream is cosmetic
    }
    setLoading(false);
  }, []);
  usePolling(fetchThoughts, 10000);

  const clearThoughts = async () => {
    try {
      await fetch('/api/thoughts/clear', { method: 'POST' });
      setThoughts([]);
      setThoughtCount(0);
    } catch (e) {
      console.error('Failed to clear thoughts:', e);
    }
  };

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-purple-500/15 text-sm">🧠</span>
          Inner Monologue
          <span className="text-xs text-chat-text-secondary">({thoughtCount})</span>
        </h3>
        <div className="flex gap-2">
          <button
            onClick={fetchThoughts}
            className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
            disabled={loading}
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={clearThoughts}
            className="text-xs text-chat-text-secondary hover:text-chat-text px-2 py-1 rounded hover:bg-chat-assistant"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="space-y-2 max-h-64 overflow-y-auto">
        {thoughts.length === 0 ? (
          <div className="text-chat-text-secondary text-sm italic">
            {chatLoading ? 'Waiting for AURA to think...' : 'Send a message to see thoughts'}
          </div>
        ) : (
          thoughts.map((thought, i) => (
            <div
              key={`${thought.type}-${thought.timestamp || i}-${i}`}
              className={`text-sm p-2 bg-chat-assistant rounded border-l-2 ${
                THOUGHT_COLORS[thought.type] || 'border-gray-500'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="flex items-center justify-center w-5 h-5 rounded text-xs bg-white/5">
                  {THOUGHT_ICONS[thought.type] || '💭'}
                </span>
                <span className="font-medium text-chat-text uppercase text-xs tracking-wide">
                  {thought.type}
                </span>
                {thought.confidence && (
                  <span className="text-chat-text-secondary text-xs">
                    [{thought.confidence}%]
                  </span>
                )}
              </div>
              <div className="text-chat-text-secondary">
                {thought.content.length > 100
                  ? thought.content.slice(0, 100) + '...'
                  : thought.content}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
