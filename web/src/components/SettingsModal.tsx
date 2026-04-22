import { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useSettingsStore, type Settings } from '../store/settingsStore';
import { toast } from './Toast';
import { apiFetch } from '../utils/apiFetch';

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

const API_PROVIDERS = [
  { name: 'anthropic', display: 'Anthropic', placeholder: 'sk-ant-...' },
  { name: 'openai', display: 'OpenAI', placeholder: 'sk-...' },
  { name: 'gemini', display: 'Google Gemini', placeholder: 'AIza...' },
  { name: 'grok', display: 'xAI (Grok)', placeholder: 'xai-...' },
  { name: 'perplexity', display: 'Perplexity', placeholder: 'pplx-...' },
  { name: 'deepseek', display: 'DeepSeek', placeholder: 'sk-...' },
];

function ApiProviderManager() {
  const [providers, setProviders] = useState<Record<string, boolean>>({});
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [keyInput, setKeyInput] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    apiFetch('/api/providers').then(r => r.json()).then(data => {
      const status: Record<string, boolean> = {};
      if (Array.isArray(data)) {
        data.forEach((p: any) => { status[p.name] = p.configured; });
      } else if (data.providers) {
        data.providers.forEach((p: any) => { status[p.name] = p.configured; });
      }
      setProviders(status);
    }).catch(() => {});
  }, []);

  const saveKey = async (providerName: string) => {
    if (!keyInput.trim()) return;
    setSaving(true);
    try {
      const resp = await apiFetch(`/api/providers/${providerName}/key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: keyInput.trim() }),
      });
      if (resp.ok) {
        setProviders(prev => ({ ...prev, [providerName]: true }));
        toast.success('API key saved', `${providerName} is now active`);
        setEditingKey(null);
        setKeyInput('');
      } else {
        toast.error('Failed', 'Could not save API key');
      }
    } catch {
      toast.error('Error', 'Network error saving key');
    }
    setSaving(false);
  };

  const removeKey = async (providerName: string) => {
    try {
      const res = await apiFetch(`/api/providers/${providerName}/key`, { method: 'DELETE' });
      if (res.ok) {
        setProviders(prev => ({ ...prev, [providerName]: false }));
        toast.success('Removed', `${providerName} key removed`);
      } else {
        toast.error('Error', `Failed to remove ${providerName} key`);
      }
    } catch {
      toast.error('Error', 'Network error removing key');
    }
  };

  return (
    <div className="space-y-2">
      {API_PROVIDERS.map(p => (
        <div key={p.name} className="flex items-center gap-2 p-2 rounded-lg bg-chat-assistant/20 border border-chat-border/30">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-xs text-chat-text font-medium">{p.display}</span>
              {providers[p.name] ? (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">Active</span>
              ) : (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-chat-border/30 text-chat-text-secondary">Not set</span>
              )}
            </div>
            {editingKey === p.name && (
              <div className="flex gap-1 mt-1.5">
                <input
                  type="password"
                  value={keyInput}
                  onChange={e => setKeyInput(e.target.value)}
                  placeholder={p.placeholder}
                  className="flex-1 px-2 py-1 text-xs bg-chat-bg border border-chat-border/50 rounded text-chat-text placeholder:text-chat-text-secondary/40 outline-none focus:border-purple-500/50"
                  autoFocus
                  onKeyDown={e => e.key === 'Enter' && saveKey(p.name)}
                />
                <button
                  onClick={() => saveKey(p.name)}
                  disabled={saving}
                  className="px-2 py-1 text-[10px] bg-purple-600/30 text-purple-300 rounded hover:bg-purple-600/40"
                >
                  Save
                </button>
                <button
                  onClick={() => { setEditingKey(null); setKeyInput(''); }}
                  className="px-2 py-1 text-[10px] bg-chat-border/30 text-chat-text-secondary rounded hover:bg-chat-border/50"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
          {editingKey !== p.name && (
            <div className="flex gap-1 flex-shrink-0">
              <button
                onClick={() => { setEditingKey(p.name); setKeyInput(''); }}
                className="px-2 py-1 text-[10px] bg-chat-border/30 text-chat-text-secondary rounded hover:bg-chat-border/50"
              >
                {providers[p.name] ? 'Change' : 'Set Key'}
              </button>
              {providers[p.name] && (
                <button
                  onClick={() => removeKey(p.name)}
                  className="px-2 py-1 text-[10px] bg-red-500/20 text-red-400 rounded hover:bg-red-500/30"
                >
                  Remove
                </button>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
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
      const response = await apiFetch('/api/alma/personality');
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
      const response = await apiFetch('/api/alma/personality', {
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
      const response = await apiFetch('/api/alma/personality/reset', { method: 'POST' });
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

          {/* API Providers */}
          <div>
            <h3 className="text-sm font-medium text-chat-text mb-3">API Providers</h3>
            <p className="text-xs text-chat-text-secondary/70 mb-3">
              Add API keys for direct model access. Models appear in the picker once a key is set.
            </p>
            <ApiProviderManager />
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
