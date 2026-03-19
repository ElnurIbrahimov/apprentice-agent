import { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface DreamData {
  insights: string[];
  merged: number;
  pruned: number;
  patterns?: Record<string, { total: number; success_rate: number }>;
}

interface DreamJournalProps {
  dreamData: DreamData | null;
}

export function DreamJournal({ dreamData }: DreamJournalProps) {
  const [dismissed, setDismissed] = useState(false);
  const [visible, setVisible] = useState(false);

  // Trigger entrance animation on mount / data change
  useEffect(() => {
    if (dreamData && !dismissed) {
      const timer = setTimeout(() => setVisible(true), 50);
      return () => clearTimeout(timer);
    }
    setVisible(false);
  }, [dreamData, dismissed]);

  if (!dreamData || dismissed) return null;

  // Get top 3 tools sorted by total usage
  const topTools = dreamData.patterns
    ? Object.entries(dreamData.patterns)
        .sort(([, a], [, b]) => b.total - a.total)
        .slice(0, 3)
    : [];

  return (
    <div
      className={`dream-card ${visible ? 'dream-card-visible' : ''}`}
    >
      {/* Dismiss button */}
      <button
        onClick={() => setDismissed(true)}
        className="absolute top-3 right-3 p-1 rounded-lg text-white/40 hover:text-white/80 hover:bg-white/10 transition-all"
        aria-label="Dismiss dream journal"
      >
        <XMarkIcon className="w-4 h-4" />
      </button>

      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <span className="text-lg">🌙</span>
        <h3
          className="text-sm font-semibold"
          style={{
            background: 'linear-gradient(135deg, #c4b5fd 0%, #93c5fd 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          Dream Consolidation
        </h3>
        <span className="text-[10px] text-white/30 ml-auto mr-6">Morning Briefing</span>
      </div>

      {/* Insights */}
      {dreamData.insights.length > 0 && (
        <div className="mb-4">
          <div className="text-[11px] uppercase tracking-wider text-white/40 mb-2">Insights</div>
          <ul className="space-y-1.5">
            {dreamData.insights.map((insight, i) => (
              <li
                key={i}
                className="dream-insight flex items-start gap-2 text-sm text-white/80"
                style={{ animationDelay: `${150 + i * 120}ms` }}
              >
                <span className="flex-shrink-0 mt-1 w-1 h-1 rounded-full bg-purple-400/60" />
                <span>{insight}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Activity */}
      <div className="mb-4">
        <div className="text-[11px] uppercase tracking-wider text-white/40 mb-1.5">Activity</div>
        <p className="text-sm text-white/70">
          Merged <span className="text-purple-300 font-medium">{dreamData.merged}</span> memories, pruned{' '}
          <span className="text-blue-300 font-medium">{dreamData.pruned}</span> stale entries
        </p>
      </div>

      {/* Patterns */}
      {topTools.length > 0 && (
        <div>
          <div className="text-[11px] uppercase tracking-wider text-white/40 mb-2">Top Patterns</div>
          <div className="space-y-2">
            {topTools.map(([name, stats]) => (
              <div key={name} className="flex items-center gap-3">
                <span className="text-xs text-white/60 w-20 truncate" title={name}>{name}</span>
                <div className="flex-1 dream-progress">
                  <div
                    className="dream-progress-fill"
                    style={{ width: `${Math.round(stats.success_rate * 100)}%` }}
                  />
                </div>
                <span className="text-[11px] text-white/50 w-14 text-right">
                  {Math.round(stats.success_rate * 100)}% ({stats.total})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
