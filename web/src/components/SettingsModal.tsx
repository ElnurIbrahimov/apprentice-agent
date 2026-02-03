import { useState } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { useSettingsStore, type Settings } from '../store/settingsStore';
import { toast } from './Toast';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { settings, updateSettings, resetSettings } = useSettingsStore();
  const [localSettings, setLocalSettings] = useState<Settings>(settings);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    updateSettings(localSettings);
    setSaved(true);
    toast.success('Settings saved', 'Your preferences have been updated');
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    resetSettings();
    setLocalSettings({
      theme: 'dark',
      fontSize: 'medium',
      showThinking: true,
      autoScroll: true,
      soundEnabled: false,
    });
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
                  value={localSettings.theme}
                  onChange={(e) => setLocalSettings({ ...localSettings, theme: e.target.value as Settings['theme'] })}
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
                  value={localSettings.fontSize}
                  onChange={(e) => setLocalSettings({ ...localSettings, fontSize: e.target.value as Settings['fontSize'] })}
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
                  onClick={() => setLocalSettings({ ...localSettings, showThinking: !localSettings.showThinking })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    localSettings.showThinking ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      localSettings.showThinking ? 'translate-x-5' : ''
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
                  onClick={() => setLocalSettings({ ...localSettings, autoScroll: !localSettings.autoScroll })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    localSettings.autoScroll ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      localSettings.autoScroll ? 'translate-x-5' : ''
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
                  onClick={() => setLocalSettings({ ...localSettings, soundEnabled: !localSettings.soundEnabled })}
                  className={`relative w-11 h-6 rounded-full transition-colors ${
                    localSettings.soundEnabled ? 'bg-purple-600' : 'bg-chat-border'
                  }`}
                >
                  <span
                    className={`absolute top-1 left-1 w-4 h-4 bg-white rounded-full transition-transform ${
                      localSettings.soundEnabled ? 'translate-x-5' : ''
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
