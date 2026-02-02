import { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Settings {
  theme: 'dark' | 'light' | 'system';
  fontSize: 'small' | 'medium' | 'large';
  showThinking: boolean;
  autoScroll: boolean;
  soundEnabled: boolean;
}

const defaultSettings: Settings = {
  theme: 'dark',
  fontSize: 'medium',
  showThinking: true,
  autoScroll: true,
  soundEnabled: false,
};

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [saved, setSaved] = useState(false);

  // Load settings from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('aura-settings');
    if (stored) {
      try {
        setSettings({ ...defaultSettings, ...JSON.parse(stored) });
      } catch (e) {
        console.error('Failed to load settings:', e);
      }
    }
  }, []);

  const handleSave = () => {
    localStorage.setItem('aura-settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setSettings(defaultSettings);
    localStorage.removeItem('aura-settings');
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
                  onChange={(e) => setSettings({ ...settings, theme: e.target.value as Settings['theme'] })}
                  className="bg-chat-bg border border-chat-border rounded-lg px-3 py-1.5 text-sm text-chat-text focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="dark">Dark</option>
                  <option value="light">Light (Coming Soon)</option>
                  <option value="system">System</option>
                </select>
              </div>

              {/* Font Size */}
              <div className="flex items-center justify-between">
                <label className="text-sm text-chat-text-secondary">Font Size</label>
                <select
                  value={settings.fontSize}
                  onChange={(e) => setSettings({ ...settings, fontSize: e.target.value as Settings['fontSize'] })}
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
                  onClick={() => setSettings({ ...settings, showThinking: !settings.showThinking })}
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
                  onClick={() => setSettings({ ...settings, autoScroll: !settings.autoScroll })}
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
                  onClick={() => setSettings({ ...settings, soundEnabled: !settings.soundEnabled })}
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

          {/* About */}
          <div>
            <h3 className="text-sm font-medium text-chat-text mb-3">About</h3>
            <div className="text-sm text-chat-text-secondary space-y-1">
              <p>AURA - Autonomous Universal Reasoning Agent</p>
              <p className="text-xs">Version 3.0 with Multi-Agent System</p>
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
          <div className="flex items-center gap-2">
            {saved && (
              <span className="text-sm text-green-500">Saved!</span>
            )}
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              Save Changes
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
