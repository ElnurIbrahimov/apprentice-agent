import { useCallback, useRef, useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { EMOTION_COLORS, NEURO_INFO, PERSONALITY_INFO } from '../utils/emotionConstants';
import { apiFetch } from '../utils/apiFetch';

interface ALMAState {
  available: boolean;
  dominant_emotion: string;
  intensity: number;
  pad: {
    pleasure: number;
    arousal: number;
    dominance: number;
  };
  mood: {
    label?: string;
    intensity?: number;
  };
  active_emotions: Array<{
    name: string;
    intensity: number;
    current_intensity: number;
    trigger: string;
  }>;
  neuromodulators: {
    dopamine: number;
    serotonin: number;
    norepinephrine: number;
    oxytocin: number;
  };
  personality: {
    openness: number;
    conscientiousness: number;
    extraversion: number;
    agreeableness: number;
    neuroticism: number;
  };
}

export function EmotionPanel() {
  const [almaState, setAlmaState] = useState<ALMAState | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeSection, setActiveSection] = useState<'emotions' | 'neuro' | 'personality'>('emotions');
  const [fetchError, setFetchError] = useState(false);
  const failCount = useRef(0);

  // Fetch ALMA state periodically (10s - emotional state doesn't change that fast)
  const fetchState = useCallback(async () => {
    try {
      const response = await apiFetch('/api/alma/state');
      if (response.ok) {
        const data = await response.json();
        setAlmaState(data);
        setFetchError(false);
        failCount.current = 0;
      } else {
        failCount.current++;
        if (failCount.current >= 3) setFetchError(true);
      }
    } catch {
      failCount.current++;
      if (failCount.current >= 3) setFetchError(true);
    }
  }, []);
  usePolling(fetchState, 30000);

  if (!almaState) {
    if (fetchError) {
      return (
        <div className="p-4 text-center space-y-2">
          <p className="text-chat-text-secondary text-sm">Emotion system unavailable</p>
          <button
            onClick={() => { setFetchError(false); failCount.current = 0; }}
            className="text-xs text-purple-400 hover:text-purple-300 underline"
          >
            Retry
          </button>
        </div>
      );
    }
    return (
      <div className="p-4 bg-chat-assistant/50 rounded-xl animate-pulse">
        <div className="h-4 bg-chat-border/30 rounded w-24 mb-2"></div>
        <div className="h-8 bg-chat-border/30 rounded w-full"></div>
      </div>
    );
  }

  const emotionColor = EMOTION_COLORS[almaState.dominant_emotion] || EMOTION_COLORS.neutral;

  return (
    <div className="bg-chat-assistant/60 rounded-xl border border-chat-border/30 overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-chat-assistant/80 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div
            className="w-3 h-3 rounded-full animate-pulse"
            style={{ backgroundColor: emotionColor, boxShadow: `0 0 8px ${emotionColor}` }}
          />
          <span className="text-chat-text font-medium capitalize">
            {almaState.dominant_emotion}
          </span>
          <span className="text-chat-text-secondary text-xs">
            {(almaState.intensity * 100).toFixed(0)}%
          </span>
        </div>
        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4 text-chat-text-secondary" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 text-chat-text-secondary" />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-3 pb-3 space-y-3">
          {/* Section tabs */}
          <div className="flex gap-1 p-1 bg-chat-bg/50 rounded-lg">
            {(['emotions', 'neuro', 'personality'] as const).map((section) => (
              <button
                key={section}
                onClick={() => setActiveSection(section)}
                className={`
                  flex-1 px-2 py-1 text-xs font-medium rounded transition-all
                  ${activeSection === section
                    ? 'bg-purple-600/30 text-purple-300'
                    : 'text-chat-text-secondary hover:text-chat-text'
                  }
                `}
              >
                {section === 'emotions' ? 'Emotions' : section === 'neuro' ? 'Neuro' : 'Personality'}
              </button>
            ))}
          </div>

          {/* Emotions Section */}
          {activeSection === 'emotions' && (
            <div className="space-y-3">
              {/* PAD Values */}
              <div className="space-y-2">
                <div className="text-xs text-chat-text-secondary font-medium">PAD Space</div>
                <PADBar label="Pleasure" value={almaState.pad.pleasure} color="#10b981" />
                <PADBar label="Arousal" value={almaState.pad.arousal} color="#f59e0b" />
                <PADBar label="Dominance" value={almaState.pad.dominance} color="#3b82f6" />
              </div>

              {/* Active Emotions */}
              {almaState.active_emotions.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-chat-text-secondary font-medium">Active Emotions</div>
                  <div className="space-y-1.5">
                    {almaState.active_emotions.slice(0, 4).map((emotion, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: EMOTION_COLORS[emotion.name] || '#6b7280' }}
                        />
                        <span className="text-xs text-chat-text capitalize flex-1">{emotion.name}</span>
                        <div className="w-16 h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${emotion.current_intensity * 100}%`,
                              backgroundColor: EMOTION_COLORS[emotion.name] || '#6b7280'
                            }}
                          />
                        </div>
                        <span className="text-xs text-chat-text-secondary w-8 text-right">
                          {(emotion.current_intensity * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Neuromodulators Section */}
          {activeSection === 'neuro' && (
            <div className="space-y-2">
              <div className="text-xs text-chat-text-secondary font-medium">Neuromodulator Levels</div>
              {Object.entries(almaState.neuromodulators).map(([key, value]) => {
                const info = NEURO_INFO[key];
                return (
                  <div key={key} className="space-y-1">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-chat-text">{info.label}</span>
                      <span className="text-xs text-chat-text-secondary">{(value * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-2 bg-chat-border/30 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${value * 100}%`,
                          backgroundColor: info.color,
                          boxShadow: `0 0 6px ${info.color}40`
                        }}
                      />
                    </div>
                    <div className="text-xs text-chat-text-secondary/60 italic">{info.effect}</div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Personality Section */}
          {activeSection === 'personality' && (
            <div className="space-y-3">
              <div className="text-xs text-chat-text-secondary font-medium">OCEAN Personality</div>
              {Object.entries(almaState.personality).map(([key, value]) => {
                const info = PERSONALITY_INFO[key];
                return (
                  <div key={key} className="space-y-1">
                    {/* Trait name and value */}
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-chat-text font-medium">{info.label}</span>
                      <span className="text-xs text-chat-text-secondary">{(value * 100).toFixed(0)}%</span>
                    </div>
                    {/* Progress bar */}
                    <div className="h-1.5 bg-chat-border/30 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700 bg-gradient-to-r from-purple-600 to-blue-500"
                        style={{ width: `${value * 100}%` }}
                      />
                    </div>
                    {/* Low/High labels below */}
                    <div className="flex justify-between text-xs text-chat-text-secondary/50">
                      <span>{info.low}</span>
                      <span>{info.high}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// PAD Value Bar Component
function PADBar({ label, value, color }: { label: string; value: number; color: string }) {
  // PAD values range from -1 to 1, normalize to 0-100 for display
  const normalizedValue = ((value + 1) / 2) * 100;
  const isPositive = value >= 0;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-chat-text-secondary w-16">{label}</span>
      <div className="flex-1 h-2 bg-chat-border/30 rounded-full overflow-hidden relative">
        {/* Center marker for 0 */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-chat-text-secondary/40" />
        {/* Value bar - grows from center */}
        <div
          className="absolute top-0 bottom-0 rounded-full transition-all duration-500"
          style={{
            left: isPositive ? '50%' : `${normalizedValue}%`,
            width: `${Math.abs(value) * 50}%`,
            backgroundColor: color,
            boxShadow: `0 0 4px ${color}60`
          }}
        />
      </div>
      <span className="text-xs text-chat-text-secondary w-10 text-right">
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}
