import { useEffect, useState } from 'react';
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

// Mood-based gradient colors for progress bar
const MOOD_GRADIENTS: Record<string, { start: string; end: string }> = {
  happy: { start: '#eab308', end: '#f59e0b' },      // yellow to amber
  excited: { start: '#f97316', end: '#eab308' },    // orange to yellow
  calm: { start: '#10a37f', end: '#14b8a6' },       // green to teal
  confident: { start: '#10a37f', end: '#3b82f6' },  // green to blue
  sad: { start: '#3b82f6', end: '#6366f1' },        // blue to indigo
  fearful: { start: '#6366f1', end: '#8b5cf6' },    // indigo to purple
  angry: { start: '#ef4444', end: '#f97316' },      // red to orange
  disgusted: { start: '#ef4444', end: '#ec4899' },  // red to pink
  curious: { start: '#8b5cf6', end: '#3b82f6' },    // purple to blue
  thoughtful: { start: '#8b5cf6', end: '#6366f1' }, // purple to indigo
  surprised: { start: '#f59e0b', end: '#8b5cf6' },  // amber to purple
  neutral: { start: '#6b7280', end: '#8e8ea0' },    // gray
  confused: { start: '#9ca3af', end: '#6b7280' },   // light gray to gray
};

export function MoodIndicator({ mood, compact = false }: MoodIndicatorProps) {
  const [animatedWidth, setAnimatedWidth] = useState(0);

  // Animate progress bar on mood change
  useEffect(() => {
    setAnimatedWidth(0);
    const timer = setTimeout(() => {
      setAnimatedWidth(mood?.confidence || 0);
    }, 50);
    return () => clearTimeout(timer);
  }, [mood?.confidence, mood?.emotion]);

  if (!mood) {
    return compact ? (
      <span className="text-lg">😐</span>
    ) : (
      <div className="text-chat-text-secondary text-sm">Mood: Unknown</div>
    );
  }

  const emoji = EMOTION_EMOJIS[mood.emotion || 'neutral'] || '😐';
  const confidence = mood.confidence || 0;
  const gradient = MOOD_GRADIENTS[mood.emotion || 'neutral'] || MOOD_GRADIENTS.neutral;

  if (compact) {
    return (
      <div className="flex items-center gap-1" title={`${mood.emotion || 'neutral'} (${confidence}%)`}>
        <span className="text-lg">{emoji}</span>
        <div className="w-8 h-1.5 bg-chat-border rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500 ease-out"
            style={{
              width: `${animatedWidth}%`,
              background: `linear-gradient(90deg, ${gradient.start} 0%, ${gradient.end} 100%)`,
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      className="mood-glow p-4 bg-chat-assistant/80 rounded-xl backdrop-blur-sm transition-all duration-500"
      data-mood={mood.emotion || 'neutral'}
    >
      <div className="flex items-center gap-3 mb-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-600/20 to-blue-600/10 flex items-center justify-center">
          <span className="text-2xl">{emoji}</span>
        </div>
        <div>
          <div className="text-chat-text font-semibold capitalize text-lg">
            {mood.emotion || 'Neutral'}
          </div>
          <div className="text-chat-text-secondary text-xs font-medium">
            {confidence}% confident
          </div>
        </div>
      </div>

      {/* Animated confidence bar with mood gradient */}
      <div className="w-full h-2.5 bg-chat-border/50 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${animatedWidth}%`,
            background: `linear-gradient(90deg, ${gradient.start} 0%, ${gradient.end} 100%)`,
            boxShadow: `0 0 10px ${gradient.start}40`,
          }}
        />
      </div>

      {/* Valence/Arousal mini-display */}
      {(mood.valence !== 0 || mood.arousal !== 0) && (
        <div className="mt-3 text-xs text-chat-text-secondary flex gap-4 font-medium">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-500"></span>
            Valence: {(mood.valence * 100).toFixed(0)}%
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500"></span>
            Arousal: {(mood.arousal * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  );
}
