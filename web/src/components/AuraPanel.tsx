import { useState, useRef, useEffect } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { AuraStatus } from '../types';
import { ArrowPathIcon, SparklesIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

const MOOD_EMOJIS: Record<string, string> = {
  excited: '🌟',
  happy: '😊',
  content: '🙂',
  neutral: '😐',
  thoughtful: '🤔',
  tired: '😴',
  concerned: '😟',
  frustrated: '😤',
};

export function AuraPanel() {
  const [status, setStatus] = useState<AuraStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [rememberText, setRememberText] = useState('');
  const [rememberResult, setRememberResult] = useState('');
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const res = await apiFetch('/api/aura');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e: any) {
      console.error('Failed to fetch AURA status:', e);
    } finally {
      setLoading(false);
    }
  };

  usePolling(fetchStatus, 15000);

  const handleRemember = async () => {
    if (!rememberText.trim()) return;
    try {
      const res = await apiFetch('/api/aura/remember', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fact: rememberText }),
      });
      const data = await res.json();
      if (data.success) {
        setRememberResult(`✓ Remembered: ${rememberText.slice(0, 30)}...`);
        setRememberText('');
      } else {
        setRememberResult(`✗ ${data.error || 'Failed'}`);
      }
    } catch (e: any) {
      setRememberResult('✗ Error storing memory');
    }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => setRememberResult(''), 3000);
  };

  if (!status) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-purple-500/15">
            <SparklesIcon className="w-4 h-4 text-purple-400" />
          </span>
          AURA Status
        </h3>
        <div className="text-chat-text-secondary text-sm">{loading ? 'Loading...' : 'AURA not loaded'}</div>
      </div>
    );
  }

  if (!status.enabled) {
    return (
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium mb-2 flex items-center gap-2">
          <span className="flex items-center justify-center w-6 h-6 rounded-md bg-purple-500/15">
            <SparklesIcon className="w-4 h-4 text-purple-400" />
          </span>
          AURA Status
        </h3>
        <div className="text-chat-text-secondary text-sm">AURA not loaded</div>
      </div>
    );
  }

  const emoji = MOOD_EMOJIS[status.mood] || '🤖';
  const energyPct = Math.round(status.energy * 100);
  const warmthPct = Math.round(status.warmth * 100);
  const engagementPct = Math.round(status.engagement * 100);

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <SparklesIcon className="w-5 h-5 text-purple-400" />
          AURA ALIVE
        </h3>
        <button
          onClick={fetchStatus}
          className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
          disabled={loading}
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Mood */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-3xl">{emoji}</span>
        <div>
          <div className="text-chat-text font-medium capitalize">{status.mood}</div>
          <div className="text-chat-text-secondary text-xs">Soul: {status.soul_name}</div>
        </div>
      </div>

      {/* Energy bars */}
      <div className="space-y-2 mb-4">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-chat-text-secondary">Energy</span>
            <span className="text-chat-text">{energyPct}%</span>
          </div>
          <div className="h-2 bg-chat-border rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all"
              style={{ width: `${energyPct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-chat-text-secondary">Warmth</span>
            <span className="text-chat-text">{warmthPct}%</span>
          </div>
          <div className="h-2 bg-chat-border rounded-full overflow-hidden">
            <div
              className="h-full bg-orange-500 transition-all"
              style={{ width: `${warmthPct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-chat-text-secondary">Engagement</span>
            <span className="text-chat-text">{engagementPct}%</span>
          </div>
          <div className="h-2 bg-chat-border rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${engagementPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="text-xs text-chat-text-secondary mb-4">
        <div>Patterns learned: {status.patterns_learned}</div>
        <div>Turns: {status.turns}</div>
      </div>

      {/* Remember */}
      <div className="border-t border-chat-border pt-3">
        <div className="text-xs text-chat-text-secondary mb-2">Store a memory:</div>
        <div className="flex gap-2">
          <input
            type="text"
            value={rememberText}
            onChange={(e) => setRememberText(e.target.value)}
            placeholder="Type a fact..."
            className="flex-1 bg-chat-assistant text-chat-text text-sm px-2 py-1 rounded border border-chat-border focus:border-chat-accent outline-none"
            onKeyDown={(e) => e.key === 'Enter' && handleRemember()}
          />
          <button
            onClick={handleRemember}
            className="px-2 py-1 text-xs bg-chat-accent text-white rounded hover:bg-chat-accent-hover"
          >
            Save
          </button>
        </div>
        {rememberResult && (
          <div className="text-xs mt-1 text-chat-text-secondary">{rememberResult}</div>
        )}
      </div>
    </div>
  );
}
