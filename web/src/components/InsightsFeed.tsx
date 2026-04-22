/**
 * InsightsFeed — Live feed of Aura's proactive insights.
 *
 * Combines curiosity targets, proactive suggestions, and drive-motivated actions
 * into a unified card feed. This is THE differentiator — no other AI does this.
 */

import { useState, useCallback, useRef } from 'react';
import { usePolling } from '../hooks/usePolling';
import { useChatStore } from '../store/chatStore';
import { haptics } from '../utils/haptics';
import type { CuriosityTarget, DriveAction } from '../types';
import {
  LightBulbIcon,
  SparklesIcon,
  BoltIcon,
  XMarkIcon,
  ChatBubbleLeftRightIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

// ─── Card types ───
type InsightCard = {
  id: string;
  type: 'curiosity' | 'suggestion' | 'drive';
  title: string;
  body: string;
  urgency: number; // 0-1
  color: string;
  icon: React.ComponentType<{ className?: string }>;
  data?: any;
};

const TYPE_CONFIG = {
  curiosity: { color: '#a78bfa', icon: LightBulbIcon, label: 'Curiosity' },
  suggestion: { color: '#34d399', icon: SparklesIcon, label: 'Insight' },
  drive: { color: '#60a5fa', icon: BoltIcon, label: 'Drive' },
};

export function InsightsFeed() {
  const [cards, setCards] = useState<InsightCard[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const addMessage = useChatStore(s => s.addMessage);
  const seenRef = useRef<Set<string>>(new Set());

  const fetchInsights = useCallback(async () => {
    const results = await Promise.allSettled([
      apiFetch('/api/proactive/curiosity').then(r => r.ok ? r.json() : null),
      apiFetch('/api/proactive/suggestion').then(r => r.ok ? r.json() : null),
      apiFetch('/api/motivation/actions').then(r => r.ok ? r.json() : null),
    ]);

    const newCards: InsightCard[] = [];

    // Curiosity targets
    const curiosityData = results[0].status === 'fulfilled' ? results[0].value : null;
    const targets: CuriosityTarget[] = curiosityData?.targets || [];
    for (const t of targets.slice(0, 5)) {
      const id = `curiosity-${t.entity_id || t.label}`;
      if (!seenRef.current.has(id)) seenRef.current.add(id);
      newCards.push({
        id,
        type: 'curiosity',
        title: t.gap_type ? `${t.gap_type.replace(/_/g, ' ')}` : 'Knowledge Gap',
        body: t.question || `I noticed "${t.label}" in my knowledge but it's disconnected. How does it relate to your work?`,
        urgency: t.urgency ?? 0.5,
        color: TYPE_CONFIG.curiosity.color,
        icon: TYPE_CONFIG.curiosity.icon,
        data: t,
      });
    }

    // Proactive suggestion
    const suggestionData = results[1].status === 'fulfilled' ? results[1].value : null;
    if (suggestionData?.has_suggestion && suggestionData.suggestion) {
      const id = `suggestion-${suggestionData.suggestion.slice(0, 30)}`;
      newCards.push({
        id,
        type: 'suggestion',
        title: 'Proactive Insight',
        body: suggestionData.suggestion,
        urgency: 0.6,
        color: TYPE_CONFIG.suggestion.color,
        icon: TYPE_CONFIG.suggestion.icon,
        data: suggestionData,
      });
    }

    // Drive actions
    const actionsData = results[2].status === 'fulfilled' ? results[2].value : null;
    const actions: DriveAction[] = actionsData?.actions || [];
    for (const a of actions.slice(0, 3)) {
      const id = `drive-${a.drive}-${a.action?.slice(0, 20)}`;
      newCards.push({
        id,
        type: 'drive',
        title: `${a.drive} Drive`,
        body: a.description || a.action,
        urgency: (a.priority ?? 50) / 100,
        color: TYPE_CONFIG.drive.color,
        icon: TYPE_CONFIG.drive.icon,
        data: a,
      });
    }

    // Sort by urgency descending
    newCards.sort((a, b) => b.urgency - a.urgency);
    setCards(newCards);
    setLoading(false);
  }, []);

  usePolling(fetchInsights, 15000);

  const handleDismiss = useCallback((id: string) => {
    haptics.light();
    setDismissed(prev => new Set(prev).add(id));
    apiFetch('/api/proactive/dismiss', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' }).catch(() => {});
  }, []);

  const handleTellMore = useCallback((card: InsightCard) => {
    haptics.medium();
    // Send to chat
    const prompt = card.type === 'curiosity'
      ? card.body
      : card.type === 'suggestion'
        ? `Tell me more about: ${card.body}`
        : `${card.body}`;
    addMessage({ role: 'user', content: prompt });
    // Switch to chat tab
    document.dispatchEvent(new CustomEvent('aura:switch-tab', { detail: 'chat' }));
    handleDismiss(card.id);
  }, [addMessage, handleDismiss]);

  const visibleCards = cards.filter(c => !dismissed.has(c.id));

  if (loading && cards.length === 0) {
    return (
      <div className="p-4 space-y-3 animate-pulse">
        {[1, 2, 3].map(i => (
          <div key={i} className="h-20 rounded-xl" style={{ background: 'var(--surface-2)' }} />
        ))}
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-3 tab-panel-scroll">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-chat-text">Aura's Mind</h3>
        <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: 'var(--surface-2)', color: 'var(--text-secondary)' }}>
          {visibleCards.length} insight{visibleCards.length !== 1 ? 's' : ''}
        </span>
      </div>

      {visibleCards.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <SparklesIcon className="w-8 h-8 mb-3 text-chat-text-tertiary" />
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>No active insights right now.</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-tertiary)' }}>
            Keep chatting — Aura is always watching for things to share.
          </p>
        </div>
      ) : (
        visibleCards.map((card, i) => {
          const Icon = card.icon;
          return (
            <div
              key={card.id}
              className="rounded-xl overflow-hidden active:scale-[0.98] transition-transform"
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border-default)',
                borderLeft: `3px solid ${card.color}`,
                animation: `spring-up 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) ${i * 60}ms both`,
              }}
            >
              {/* Header */}
              <div className="flex items-center gap-2 px-3 pt-3 pb-1">
                <span className="flex-shrink-0" style={{ color: card.color }}><Icon className="w-4 h-4" /></span>
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: card.color }}>
                  {TYPE_CONFIG[card.type].label}
                </span>
                <span className="text-[10px] capitalize" style={{ color: 'var(--text-tertiary)' }}>
                  {card.title}
                </span>
                <div className="flex-1" />
                <button
                  onClick={() => handleDismiss(card.id)}
                  className="p-1 rounded-lg transition-colors"
                  style={{ color: 'var(--text-tertiary)' }}
                >
                  <XMarkIcon className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Body */}
              <p className="px-3 pb-2 text-sm leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                {card.body}
              </p>

              {/* Actions */}
              <div className="flex items-center gap-2 px-3 pb-3">
                <button
                  onClick={() => handleTellMore(card)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all active:scale-95"
                  style={{ background: `${card.color}20`, color: card.color }}
                >
                  <ChatBubbleLeftRightIcon className="w-3.5 h-3.5" />
                  Tell me more
                </button>
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}
