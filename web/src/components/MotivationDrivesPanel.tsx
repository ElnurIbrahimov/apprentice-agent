import { useState, useCallback } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface DriveData {
  drive_type: string;
  intensity: number;
  satisfaction: number;
  urgency: number;
  hours_since_satisfied: number;
  triggers: string[];
}

interface MotivationStatus {
  active: boolean;
  drives: Record<string, DriveData>;
  dominant_drive: string;
  dominant_urgency: number;
  pending_actions: number;
  stats: {
    assessments_run: number;
    actions_generated: number;
    preferences_pushed: number;
  };
}

const DRIVE_CONFIG: Record<string, { icon: string; color: string; barColor: string; label: string }> = {
  curiosity:  { icon: '\uD83D\uDD0D', color: 'text-blue-400',   barColor: 'bg-blue-500',   label: 'Curiosity' },
  competence: { icon: '\uD83C\uDFAF', color: 'text-green-400',  barColor: 'bg-green-500',  label: 'Competence' },
  social:     { icon: '\uD83E\uDD1D', color: 'text-pink-400',   barColor: 'bg-pink-500',   label: 'Social' },
  coherence:  { icon: '\uD83E\uDDE9', color: 'text-purple-400', barColor: 'bg-purple-500', label: 'Coherence' },
};

export function MotivationDrivesPanel() {
  const [status, setStatus] = useState<MotivationStatus | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/motivation/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch {
      // Silently ignore
    }
  }, []);

  usePolling(fetchStatus, 15000); // 15 second polling

  if (!status) {
    return (
      <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 p-3 animate-pulse">
        <div className="h-4 bg-chat-border/30 rounded w-24"></div>
      </div>
    );
  }

  if (!status.active) {
    return (
      <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
        <div className="p-3">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-lg">🧠</span>
            <span className="text-chat-text font-medium text-sm">Motivation Drives</span>
          </div>
          <p className="text-chat-text-secondary text-xs italic mb-3">
            Idle — activates during downtime and self-directed thinking.
          </p>
          <div className="space-y-1.5 opacity-40">
            {Object.entries(DRIVE_CONFIG).map(([key, config]) => (
              <div key={key} className="flex items-center gap-2">
                <span className="text-xs text-chat-text-secondary w-20 capitalize">{config.label}</span>
                <div className="flex-1 bg-chat-bg rounded-full h-1.5">
                  <div className="h-full w-0 rounded-full bg-purple-500" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const dominant = status.dominant_drive;
  const dominantConfig = DRIVE_CONFIG[dominant] || DRIVE_CONFIG.curiosity;
  const dominantDrive = status.drives[dominant];

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-lg">{dominantConfig.icon}</span>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-chat-text font-medium text-sm">Drives</span>
              <span className={`text-xs ${dominantConfig.color}`}>
                {dominantConfig.label} {Math.round(dominantDrive?.urgency * 100 || 0)}%
              </span>
            </div>
            {/* Mini drive bars - always visible */}
            <div className="flex gap-1 mt-1">
              {Object.entries(DRIVE_CONFIG).map(([key, config]) => {
                const drive = status.drives[key];
                const urgency = drive?.urgency || 0;
                return (
                  <div key={key} className="flex-1 h-1 bg-chat-border/30 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${config.barColor}`}
                      style={{ width: `${urgency * 100}%`, opacity: urgency > 0.3 ? 1 : 0.4 }}
                    />
                  </div>
                );
              })}
            </div>
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
          {/* Drive details */}
          {Object.entries(DRIVE_CONFIG).map(([key, config]) => {
            const drive = status.drives[key];
            if (!drive) return null;
            const isDominant = key === dominant;

            return (
              <div
                key={key}
                className={`p-2 rounded-lg ${isDominant ? 'bg-chat-bg/40 border border-chat-border/30' : 'bg-chat-bg/20'}`}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{config.icon}</span>
                    <span className={`text-xs font-medium ${config.color}`}>
                      {config.label}
                    </span>
                    {isDominant && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-yellow-500/20 text-yellow-300 border border-yellow-500/30">
                        dominant
                      </span>
                    )}
                  </div>
                  <span className="text-xs text-chat-text-secondary font-mono">
                    {Math.round(drive.urgency * 100)}%
                  </span>
                </div>

                {/* Urgency bar */}
                <div className="h-1.5 bg-chat-border/30 rounded-full overflow-hidden mb-1">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${config.barColor}`}
                    style={{ width: `${drive.urgency * 100}%` }}
                  />
                </div>

                {/* Trigger info */}
                {drive.triggers.length > 0 && (
                  <div className="text-[10px] text-chat-text-secondary truncate mt-0.5">
                    {drive.triggers[drive.triggers.length - 1]}
                  </div>
                )}
              </div>
            );
          })}

          {/* Stats */}
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-chat-bg/30 rounded px-2 py-1.5 text-center" title="Motivation evaluation cycles run">
              <div className="text-chat-text-secondary">Cycles</div>
              <div className="text-chat-text font-mono">{status.stats.assessments_run}</div>
            </div>
            <div className="bg-chat-bg/30 rounded px-2 py-1.5 text-center" title="Actions triggered by drives">
              <div className="text-chat-text-secondary">Actions</div>
              <div className="text-chat-text font-mono">{status.stats.actions_generated}</div>
            </div>
            <div className="bg-chat-bg/30 rounded px-2 py-1.5 text-center" title="Times a drive exceeded its threshold">
              <div className="text-chat-text-secondary">Triggered</div>
              <div className="text-chat-text font-mono">{status.stats.preferences_pushed}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
