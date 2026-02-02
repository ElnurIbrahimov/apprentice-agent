import { useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { MoodIndicator } from './MoodIndicator';
import {
  XMarkIcon,
  TrashIcon,
  Cog6ToothIcon,
  SignalIcon,
  SignalSlashIcon,
} from '@heroicons/react/24/outline';
import { SparklesIcon } from '@heroicons/react/24/solid';

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
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
    <div className="h-full bg-chat-sidebar flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-chat-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
            <SparklesIcon className="w-5 h-5 text-white" />
          </div>
          <span className="text-chat-text font-semibold">AURA</span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-1 text-chat-text-secondary hover:text-chat-text rounded lg:hidden"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        )}
      </div>

      {/* Status section */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Connection status */}
        <div className="flex items-center gap-2 text-sm">
          {connectionStatus === 'connected' ? (
            <>
              <SignalIcon className="w-4 h-4 text-green-500" />
              <span className="text-green-500">Connected</span>
            </>
          ) : connectionStatus === 'connecting' ? (
            <>
              <SignalIcon className="w-4 h-4 text-yellow-500 animate-pulse" />
              <span className="text-yellow-500">Connecting...</span>
            </>
          ) : (
            <>
              <SignalSlashIcon className="w-4 h-4 text-red-500" />
              <span className="text-red-500">Disconnected</span>
            </>
          )}
        </div>

        {/* Mood indicator */}
        <div>
          <h3 className="text-chat-text-secondary text-xs uppercase tracking-wide mb-2">
            Mood
          </h3>
          <MoodIndicator mood={mood} />
        </div>

        {/* Stats */}
        {status && (
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wide mb-2">
              Stats
            </h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-chat-text-secondary">Model</span>
                <span className="text-chat-text">{status.model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-chat-text-secondary">Last Model</span>
                <span className="text-chat-text text-xs truncate max-w-[120px]">
                  {status.last_model_used || '-'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-chat-text-secondary">AURA</span>
                <span className={status.aura_enabled ? 'text-green-500' : 'text-chat-text-secondary'}>
                  {status.aura_enabled ? 'Enabled' : 'Disabled'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-chat-text-secondary">Memories</span>
                <span className="text-chat-text">{status.memory_count}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-chat-text-secondary">Queries</span>
                <span className="text-chat-text">{status.query_count}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer actions */}
      <div className="p-4 border-t border-chat-border space-y-2">
        <button
          onClick={handleClearHistory}
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant rounded-lg transition-colors"
        >
          <TrashIcon className="w-4 h-4" />
          Clear History
        </button>
        <button
          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant rounded-lg transition-colors opacity-50 cursor-not-allowed"
          disabled
        >
          <Cog6ToothIcon className="w-4 h-4" />
          Settings
        </button>
      </div>
    </div>
  );
}
