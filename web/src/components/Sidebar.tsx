import { useEffect, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { MoodIndicator } from './MoodIndicator';
import { SettingsModal } from './SettingsModal';
import {
  XMarkIcon,
  TrashIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { SparklesIcon } from '@heroicons/react/24/solid';

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
  const [showSettings, setShowSettings] = useState(false);

  const {
    mood,
    status,
    setStatus,
    connectionStatus,
    clearMessages,
  } = useChatStore();

  // Poll status every 5 seconds
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/status');
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
        }
      } catch (e) {
        console.error('Failed to fetch status:', e);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [setStatus]);

  const handleClearHistory = async () => {
    if (window.confirm('Clear all messages?')) {
      try {
        await fetch('/api/chat/clear', { method: 'POST' });
        clearMessages();
      } catch (e) {
        console.error('Failed to clear history:', e);
      }
    }
  };

  return (
    <>
      <div className="h-full glass flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-chat-border/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-600 to-blue-600 flex items-center justify-center shadow-glow-purple transition-all duration-300 hover:scale-105 hover:shadow-glow-purple-lg">
              <SparklesIcon className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="text-chat-text font-bold text-lg">AURA</span>
              <div className="text-xs text-chat-text-secondary">v3.0 ALIVE</div>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-lg transition-all duration-200 lg:hidden"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Status section */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Connection status */}
          <div className="flex items-center gap-3 text-sm">
            {connectionStatus === 'connected' ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="connected-dot absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <span className="text-green-400 font-medium">Connected</span>
              </>
            ) : connectionStatus === 'connecting' ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
                </span>
                <span className="text-yellow-400 font-medium">Connecting...</span>
              </>
            ) : (
              <>
                <span className="h-3 w-3 rounded-full bg-red-500"></span>
                <span className="text-red-400 font-medium">Disconnected</span>
              </>
            )}
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Mood indicator */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Current Mood
            </h3>
            <MoodIndicator mood={mood} />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Stats */}
          {status && (
            <div>
              <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
                System Stats
              </h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between items-center p-2 rounded-lg hover:bg-chat-assistant/30 transition-colors duration-200">
                  <span className="text-chat-text-secondary">Model</span>
                  <span className="text-chat-text font-medium">{status.model}</span>
                </div>
                <div className="flex justify-between items-center p-2 rounded-lg hover:bg-chat-assistant/30 transition-colors duration-200">
                  <span className="text-chat-text-secondary">Last Model</span>
                  <span className="text-chat-text text-xs truncate max-w-[120px] font-medium">
                    {status.last_model_used || '-'}
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 rounded-lg hover:bg-chat-assistant/30 transition-colors duration-200">
                  <span className="text-chat-text-secondary">AURA</span>
                  <span className={`font-medium ${status.aura_enabled ? 'text-green-400' : 'text-chat-text-secondary'}`}>
                    {status.aura_enabled ? 'Enabled' : 'Disabled'}
                  </span>
                </div>
                <div className="flex justify-between items-center p-2 rounded-lg hover:bg-chat-assistant/30 transition-colors duration-200">
                  <span className="text-chat-text-secondary">Memories</span>
                  <span className="text-chat-text font-medium">{status.memory_count}</span>
                </div>
                <div className="flex justify-between items-center p-2 rounded-lg hover:bg-chat-assistant/30 transition-colors duration-200">
                  <span className="text-chat-text-secondary">Queries</span>
                  <span className="text-chat-text font-medium">{status.query_count}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="p-4 border-t border-chat-border/50 space-y-2">
          <button
            onClick={handleClearHistory}
            className="w-full flex items-center gap-3 px-4 py-3 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-xl transition-all duration-200 group"
          >
            <TrashIcon className="w-5 h-5 transition-transform duration-200 group-hover:scale-110" />
            <span>Clear History</span>
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="w-full flex items-center gap-3 px-4 py-3 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-xl transition-all duration-200 group"
          >
            <Cog6ToothIcon className="w-5 h-5 transition-transform duration-200 group-hover:rotate-90" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </>
  );
}
