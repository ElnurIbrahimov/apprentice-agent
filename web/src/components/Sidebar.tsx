import { useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { usePolling } from '../hooks/usePolling';
import { AuraBreathingAvatar, AuraStatusLine } from './AuraBreathingAvatar';
import { ConversationList } from './ConversationList';
import {
  XMarkIcon,
  PlusIcon,
} from '@heroicons/react/24/outline';
import pkg from '../../package.json';

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
  const {
    setStatus,
    connectionStatus,
    isLoading,
  } = useChatStore();

  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch {
      // Ignore
    }
  }, [setStatus]);
  usePolling(fetchStatus, 30000);

  return (
    <>
      <div className="h-full glass flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-chat-border/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="mood-avatar-glow rounded-full">
              <AuraBreathingAvatar
                isActive={connectionStatus === 'connected'}
                isThinking={isLoading}
                size="md"
              />
            </div>
            <div>
              <span className="text-chat-text font-bold text-lg" title={`AURA v${pkg.version}`}>AURA</span>
              <AuraStatusLine
                status={isLoading ? 'Thinking...' : null}
                isVisible={connectionStatus === 'connected'}
              />
            </div>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => document.dispatchEvent(new CustomEvent('aura:new-chat'))}
              className="p-2 text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-lg transition-all duration-200 bg-chat-accent/10"
              aria-label="New Chat"
              title="New Chat"
            >
              <PlusIcon className="w-5 h-5" />
            </button>
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-lg transition-all duration-200 lg:hidden"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Conversation list — main body */}
        <div className="flex-1 overflow-y-auto px-4 pt-2 pb-2">
          <ConversationList />
        </div>
      </div>
    </>
  );
}
