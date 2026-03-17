import { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useSettingsStore, type Settings } from '../store/settingsStore';
import { toast } from './Toast';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface PersonalityTraits {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

interface TraitDescription {
  name: string;
  low: string;
  high: string;
  description: string;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { settings, updateSettings, resetSettings } = useSettingsStore();

  // Personality state
  const [personality, setPersonality] = useState<PersonalityTraits>({
    openness: 0.8,
    conscientiousness: 0.7,
    extraversion: 0.5,
    agreeableness: 0.75,
    neuroticism: 0.25,
  });
  const [traitDescriptions, setTraitDescriptions] = useState<Record<string, TraitDescription>>({});
  const [personalityLoading, setPersonalityLoading] = useState(false);
  const [personalityAvailable, setPersonalityAvailable] = useState(false);

  // Fetch personality when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchPersonality();
    }
  }, [isOpen]);

  const fetchPersonality = async () => {
    try {
      const response = await fetch('/api/alma/personality');
      if (response.ok) {
        const data = await response.json();
        setPersonalityAvailable(data.available);
        setPersonality(data.traits);
        setTraitDescriptions(data.descriptions || {});
      }
    } catch (e) {
      console.error('Failed to fetch personality:', e);
    }
  };

  const handlePersonalityChange = (trait: keyof PersonalityTraits, value: number) => {
    setPersonality(prev => ({ ...prev, [trait]: value }));
  };

  const savePersonality = async () => {
    setPersonalityLoading(true);
    try {
      const response = await fetch('/api/alma/personality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(personality),
      });
      if (response.ok) {
        toast.success('Personality updated', 'AURA\'s personality traits have been saved');
      } else {
        toast.error('Failed to save', 'Could not update personality');
      }
    } catch (e) {
      toast.error('Error', 'Failed to save personality');
    } finally {
      setPersonalityLoading(false);
    }
  };

  const resetPersonality = async () => {
    setPersonalityLoading(true);
    try {
      const response = await fetch('/api/alma/personality/reset', { method: 'POST' });
      if (response.ok) {
        const data = await response.json();
        setPersonality(data.traits);
        toast.info('Personality reset', 'Restored to AURA defaults');
      }
    } catch (e) {
      toast.error('Error', 'Failed to reset personality');
    } finally {
      setPersonalityLoading(false);
    }
  };

  const handleReset = () => {
    resetSettings();
    toast.info('Settings reset', 'All settings restored to defaults');
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-chat-sidebar border border-chat-border rounded-xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-chat-border">
          <h2 className="text-lg font-semibold text-chat-text">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 text-chat-text-secondary hover:text-chat-text rounded-lg transition-colors"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6 max-h-[60vh] overflow-y-auto">
          {/* Appearance */}
          <div>
            <h3 className="text-sm font-medium text-chat-text mb-3">Appearance</h3>
            <div className="space-y-3">
              {/* Theme */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-chat-text-secondary">Theme</label>
                <select
                  value={settings.theme}
                  onChange={(e) => updateSettings({ theme: e.target.value as Settings['theme'] })}
                  className="bg-chat-bg border border-chat-border rounded-lg px-3 py-1.5 text-sm text-chat-text focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="dark">Dark</option>
                  <option value="light" disabled>Light (Coming Soon)</option>
                  <option value="system">System</option>
                </select>
              </div>

              {/* Font Size */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-chat-text-secondary">Font Size</label>
                <select
                  value={settings.fontSize}
                  onChange={(e) => updateSettings({ fontSize: e.target.value as Settings['fontSize'] })}
                  className="bg-chat-bg border border-chat-border rounded-lg px-3 py-1.5 text-sm text-chat-text focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="small">Small</option>
                  <option value="medium">Medium</option>
                  <option value="large">Large</option>
                </select>
              </div>
            </div>
          </div>

          {/* Behavior */}
          <div>
            <h3 className="text-sm font-medium text-chat-text mb-3">Behavior</h3>
            <div className="space-y-3">
              {/* Show Thinking */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm text-chat-text-secondary">Show Thinking</label>
                  <p className="text-xs text-chat-text-secondary/70">Display AURA's thought process</p>
                </div>
                <button
                  onClick={() => updateSettings({ showThinking: !settings.showThinking })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    settings.showThinking ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      settings.showThinking ? 'translate-x-5' : ''
                    }`}
                  />
                </button>
              </div>

              {/* Auto Scroll */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm text-chat-text-secondary">Auto Scroll</label>
                  <p className="text-xs text-chat-text-secondary/70">Scroll to new messages</p>
                </div>
                <button
                  onClick={() => updateSettings({ autoScroll: !settings.autoScroll })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    settings.autoScroll ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      settings.autoScroll ? 'translate-x-5' : ''
                    }`}
                  />
                </button>
              </div>

              {/* Sound */}
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-sm text-chat-text-secondary">Sound Effects</label>
                  <p className="text-xs text-chat-text-secondary/70">Play sounds for notifications</p>
                </div>
                <button
                  onClick={() => updateSettings({ soundEnabled: !settings.soundEnabled })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    settings.soundEnabled ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      settings.soundEnabled ? 'translate-x-5' : ''
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>

          {/* AURA Personality */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-chat-text">AURA Personality (OCEAN)</h3>
              {personalityAvailable && (
                <button
                  onClick={resetPersonality}
                  disabled={personalityLoading}
                  className="text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
                >
                  Reset
                </button>
              )}
            </div>
            {personalityAvailable ? (
              <div className="space-y-4">
                {/* Openness */}
                <PersonalitySlider
                  trait="openness"
                  value={personality.openness}
                  onChange={(v) => handlePersonalityChange('openness', v)}
                  description={traitDescriptions.openness}
                  color="purple"
                />
                {/* Conscientiousness */}
                <PersonalitySlider
                  trait="conscientiousness"
                  value={personality.conscientiousness}
                  onChange={(v) => handlePersonalityChange('conscientiousness', v)}
                  description={traitDescriptions.conscientiousness}
                  color="blue"
                />
                {/* Extraversion */}
                <PersonalitySlider
                  trait="extraversion"
                  value={personality.extraversion}
                  onChange={(v) => handlePersonalityChange('extraversion', v)}
                  description={traitDescriptions.extraversion}
                  color="yellow"
                />
                {/* Agreeableness */}
                <PersonalitySlider
                  trait="agreeableness"
                  value={personality.agreeableness}
                  onChange={(v) => handlePersonalityChange('agreeableness', v)}
                  description={traitDescriptions.agreeableness}
                  color="green"
                />
                {/* Neuroticism */}
                <PersonalitySlider
                  trait="neuroticism"
                  value={personality.neuroticism}
                  onChange={(v) => handlePersonalityChange('neuroticism', v)}
                  description={traitDescriptions.neuroticism}
                  color="red"
                />

                <button
                  onClick={savePersonality}
                  disabled={personalityLoading}
                  className="w-full mt-2 px-4 py-2 text-sm bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded-lg transition-colors border border-purple-500/30"
                >
                  {personalityLoading ? 'Saving...' : 'Save Personality'}
                </button>
              </div>
            ) : (
              <div className="text-sm text-chat-text-secondary/70 italic">
                ALMA emotional system not available
              </div>
            )}
          </div>

          {/* About */}
          <div>
            <h3 className="text-sm font-medium text-chat-text mb-3">About</h3>
            <div className="text-sm text-chat-text-secondary space-y-1">
              <p>AURA - Autonomous Universal Reasoning Agent</p>
              <p className="text-xs">Version 4.3.0 with Multi-Agent System</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-chat-border">
          <button
            onClick={handleReset}
            className="px-4 py-2 text-sm text-chat-text-secondary hover:text-chat-text transition-colors"
          >
            Reset to Defaults
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// Personality trait slider component
interface PersonalitySliderProps {
  trait: string;
  value: number;
  onChange: (value: number) => void;
  description?: {
    name: string;
    low: string;
    high: string;
    description: string;
  };
  color: 'purple' | 'blue' | 'green' | 'yellow' | 'red';
}

const TRAIT_COLORS = {
  purple: { bg: 'bg-purple-600', glow: 'shadow-purple-500/30' },
  blue: { bg: 'bg-blue-600', glow: 'shadow-blue-500/30' },
  green: { bg: 'bg-emerald-600', glow: 'shadow-emerald-500/30' },
  yellow: { bg: 'bg-amber-500', glow: 'shadow-amber-500/30' },
  red: { bg: 'bg-red-500', glow: 'shadow-red-500/30' },
};

function PersonalitySlider({ trait, value, onChange, description, color }: PersonalitySliderProps) {
  const colors = TRAIT_COLORS[color];
  const percentage = Math.round(value * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-chat-text capitalize">
          {description?.name || trait}
        </span>
        <span className="text-xs text-chat-text-secondary font-mono">
          {percentage}%
        </span>
      </div>

      {/* Labels */}
      <div className="flex justify-between text-xs text-chat-text-secondary/60">
        <span>{description?.low || 'Low'}</span>
        <span>{description?.high || 'High'}</span>
      </div>

      {/* Slider track */}
      <div className="relative h-2 bg-chat-border/50 rounded-full overflow-hidden">
        {/* Fill */}
        <div
          className={`absolute inset-y-0 left-0 ${colors.bg} rounded-full transition-all duration-150`}
          style={{ width: `${percentage}%` }}
        />
        {/* Slider input */}
        <input
          type="range"
          min="0"
          max="100"
          value={percentage}
          onChange={(e) => onChange(parseInt(e.target.value) / 100)}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        {/* Thumb indicator */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 w-3 h-3 ${colors.bg} rounded-full shadow-lg ${colors.glow} transition-all duration-150 pointer-events-none`}
          style={{ left: `calc(${percentage}% - 6px)` }}
        />
      </div>

      {/* Description tooltip */}
      {description?.description && (
        <p className="text-xs text-chat-text-secondary/50 italic">
          {description.description}
        </p>
      )}
    </div>
  );
}
