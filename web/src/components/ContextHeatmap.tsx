import { useState, useCallback, useEffect, useRef } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface FocusItem {
  name: string;
  category: string;
  weight: number;
  size: string;
  color: string;
  opacity: number;
}

interface HeatmapData {
  items: FocusItem[];
  timestamp: string;
}

const SIZE_CLASSES: Record<string, string> = {
  xl: 'text-sm px-3 py-1.5',
  lg: 'text-xs px-2.5 py-1',
  md: 'text-xs px-2 py-0.5',
  sm: 'text-[10px] px-1.5 py-0.5',
};

const CATEGORY_ICONS: Record<string, string> = {
  topic: '💡',
  entity: '🏷️',
  keyword: '🔤',
  memory: '🧠',
  emotion: '💫',
  action: '⚡',
};

export function ContextHeatmap() {
  const [isExpanded, setIsExpanded] = useState(false);
  const [data, setData] = useState<HeatmapData | null>(null);
  const [pulsingItems, setPulsingItems] = useState<Set<string>>(new Set());
  const prevItemsRef = useRef<FocusItem[] | null>(null);

  // Fetch heatmap data
  const fetchData = useCallback(async () => {
    try {
      const response = await fetch('/api/context/heatmap');
      if (response.ok) {
        const newData = await response.json();

        // Detect new/boosted items for pulse animation
        const prevItems = prevItemsRef.current;
        if (prevItems) {
          const newPulsing = new Set<string>();
          newData.items.forEach((item: FocusItem) => {
            const existing = prevItems.find(i => i.name === item.name);
            if (!existing || item.weight > existing.weight + 0.1) {
              newPulsing.add(item.name);
            }
          });
          if (newPulsing.size > 0) {
            setPulsingItems(newPulsing);
            setTimeout(() => setPulsingItems(new Set()), 2000);
          }
        }

        prevItemsRef.current = newData.items;
        setData(newData);
      }
    } catch (e) {
      // Silently ignore - not critical
    }
  }, []);

  // Poll for data (30s - context awareness is cosmetic)
  usePolling(fetchData, 30000);

  // Refresh data when expanded
  useEffect(() => {
    if (isExpanded) {
      fetchData();
    }
  }, [isExpanded, fetchData]);

  const topItems = data?.items?.slice(0, 3) || [];
  const hasItems = data?.items && data.items.length > 0;

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-lg">🎯</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Focus</span>
              {hasItems && (
                <span className="text-xs text-chat-text-secondary">
                  {data?.items.length} active
                </span>
              )}
            </div>

            {/* Preview of top focus items */}
            {!isExpanded && topItems.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {topItems.map((item, idx) => (
                  <span
                    key={idx}
                    className="text-[10px] px-1.5 py-0.5 rounded-full bg-chat-bg/50 text-chat-text-secondary truncate max-w-[80px]"
                    style={{
                      borderLeft: `2px solid ${item.color}`,
                    }}
                  >
                    {item.name}
                  </span>
                ))}
                {data && data.items.length > 3 && (
                  <span className="text-[10px] text-chat-text-secondary/50">
                    +{data.items.length - 3}
                  </span>
                )}
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

      {/* Expanded content - Heatmap visualization */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3">
          {hasItems ? (
            <>
              {/* Focus bubbles/tags */}
              <div className="flex flex-wrap gap-1.5">
                {data?.items.map((item, idx) => (
                  <div
                    key={`${item.name}-${idx}`}
                    className={`
                      rounded-full font-medium transition-all duration-300
                      ${SIZE_CLASSES[item.size] || SIZE_CLASSES.sm}
                      ${pulsingItems.has(item.name) ? 'animate-pulse ring-2 ring-white/30' : ''}
                    `}
                    style={{
                      backgroundColor: `${item.color}${Math.round(Math.min(item.opacity, 1) * 200).toString(16).padStart(2, '0')}`,
                      color: item.color,
                      borderLeft: `3px solid ${item.color}`,
                    }}
                    title={`${item.category}: ${item.name} (${Math.round(item.weight * 100)}%)`}
                  >
                    <span className="mr-1">{CATEGORY_ICONS[item.category] || '•'}</span>
                    {item.name}
                  </div>
                ))}
              </div>

              {/* Category legend */}
              <div className="flex flex-wrap gap-2 pt-2 border-t border-chat-border/30">
                {Object.entries(CATEGORY_ICONS).map(([category, icon]) => {
                  const count = data?.items.filter(i => i.category === category).length || 0;
                  if (count === 0) return null;
                  return (
                    <div key={category} className="flex items-center gap-1 text-[10px] text-chat-text-secondary/60">
                      <span>{icon}</span>
                      <span className="capitalize">{category}</span>
                      <span className="text-chat-text-secondary/40">({count})</span>
                    </div>
                  );
                })}
              </div>

              {/* Focus intensity bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-chat-text-secondary/60">
                  <span>Focus Intensity</span>
                  <span>{data?.items.length} topics</span>
                </div>
                <div className="h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${Math.min(100, (data?.items.reduce((sum, i) => sum + i.weight, 0) || 0) * 20)}%`,
                      background: 'linear-gradient(90deg, #8b5cf6, #ec4899, #f59e0b)',
                    }}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-4">
              <div className="text-2xl mb-2">🎯</div>
              <div className="text-xs text-chat-text-secondary">
                Start chatting — this map shows AURA's attention focus.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
