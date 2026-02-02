import type { MoodState } from '../types';

interface MoodIndicatorProps {
  mood: MoodState | null;
  compact?: boolean;
}

const EMOTION_EMOJIS: Record<string, string> = {
  happy: '😊',
  sad: '😢',
  angry: '😠',
  fearful: '😨',
  surprised: '😲',
  disgusted: '🤢',
  neutral: '😐',
  curious: '🤔',
  excited: '🤩',
  calm: '😌',
  confident: '😎',
  confused: '😕',
  thoughtful: '🧐',
};

export function MoodIndicator({ mood, compact = false }: MoodIndicatorProps) {
  if (!mood) {
    return compact ? (
      <span className="text-lg">😐</span>
    ) : (
      <div className="text-chat-text-secondary text-sm">Mood: Unknown</div>
    );
  }

  const emoji = EMOTION_EMOJIS[mood.emotion || 'neutral'] || '😐';
  const confidence = mood.confidence || 0;

  if (compact) {
    return (
      <div className="flex items-center gap-1" title={`${mood.emotion || 'neutral'} (${confidence}%)`}>
        <span className="text-lg">{emoji}</span>
        <div className="w-8 h-1.5 bg-chat-border rounded-full overflow-hidden">
          <div
            className="h-full bg-chat-accent transition-all duration-300"
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 bg-chat-assistant rounded-lg">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-2xl">{emoji}</span>
        <div>
          <div className="text-chat-text font-medium capitalize">
            {mood.emotion || 'Neutral'}
          </div>
          <div className="text-chat-text-secondary text-xs">
            {confidence}% confident
          </div>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="w-full h-2 bg-chat-border rounded-full overflow-hidden">
        <div
          className="h-full bg-chat-accent transition-all duration-500 ease-out"
          style={{ width: `${confidence}%` }}
        />
      </div>

      {/* Valence/Arousal mini-display */}
      {(mood.valence !== 0 || mood.arousal !== 0) && (
        <div className="mt-2 text-xs text-chat-text-secondary flex gap-4">
          <span>Valence: {(mood.valence * 100).toFixed(0)}%</span>
          <span>Arousal: {(mood.arousal * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}
