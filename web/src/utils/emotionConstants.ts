export const EMOTION_COLORS: Record<string, string> = {
  joy: '#eab308',
  happy: '#eab308',
  excited: '#f97316',
  curious: '#8b5cf6',
  engaged: '#3b82f6',
  confident: '#10b981',
  calm: '#14b8a6',
  neutral: '#6b7280',
  sad: '#60a5fa',
  fearful: '#6366f1',
  angry: '#ef4444',
  surprised: '#f59e0b',
  thoughtful: '#8b5cf6',
};

export const NEURO_INFO: Record<string, { color: string; label: string; effect: string }> = {
  dopamine: { color: '#f59e0b', label: 'Dopamine', effect: 'Motivation & Reward' },
  serotonin: { color: '#10b981', label: 'Serotonin', effect: 'Mood Stability' },
  norepinephrine: { color: '#ef4444', label: 'Norepinephrine', effect: 'Alertness' },
  oxytocin: { color: '#ec4899', label: 'Oxytocin', effect: 'Social Bonding' },
};

export const PERSONALITY_INFO: Record<string, { label: string; low: string; high: string }> = {
  openness: { label: 'Openness', low: 'Conventional', high: 'Creative' },
  conscientiousness: { label: 'Conscientiousness', low: 'Flexible', high: 'Organized' },
  extraversion: { label: 'Extraversion', low: 'Reserved', high: 'Outgoing' },
  agreeableness: { label: 'Agreeableness', low: 'Analytical', high: 'Empathetic' },
  neuroticism: { label: 'Neuroticism', low: 'Stable', high: 'Sensitive' },
};
