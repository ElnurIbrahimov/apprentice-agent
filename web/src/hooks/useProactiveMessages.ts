import { useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { usePolling } from './usePolling';
import { hasSeenProactive, markProactiveSeen } from './proactiveDedup';
import { apiFetch } from '../utils/apiFetch';

interface ProactiveMessageResponse {
  action: string;
  content: string;
  priority: string;
  timestamp: string;
  delivered: boolean;
  metadata: Record<string, unknown>;
}

interface ProactiveMessagesResponse {
  count: number;
  messages: ProactiveMessageResponse[];
}

/**
 * Hook to poll for proactive messages from the Gateway Daemon
 * and inject them into the chat as AURA-initiated messages.
 */
export function useProactiveMessages(enabled: boolean = true) {
  const { addMessage, connectionStatus } = useChatStore();
  const lastCheckRef = useRef<number>(0);
  const isPollingRef = useRef<boolean>(false);

  const fetchAndAddMessages = useCallback(async () => {
    // Prevent concurrent fetches
    if (isPollingRef.current) return;
    isPollingRef.current = true;

    try {
      const response = await apiFetch('/api/proactive/messages');
      if (!response.ok) return;

      const data: ProactiveMessagesResponse = await response.json();

      if (data.messages && data.messages.length > 0) {
        for (const msg of data.messages) {
          // Deduplicate by timestamp + action fingerprint
          const msgId = `${msg.action}:${msg.content.slice(0, 40)}`;
          if (hasSeenProactive(msgId)) continue;
          markProactiveSeen(msgId);

          // Create a chat message from the proactive message
          const actionLabels: Record<string, string> = {
            notify: 'noticed something',
            suggest: 'has a suggestion',
            remind: 'wants to remind you',
            ask: 'is curious',
            intervene: 'needs your attention',
            prepare: 'prepared something',
          };

          const triggerLabel = actionLabels[msg.action] || 'wants to share';

          addMessage({
            role: 'assistant',
            content: msg.content,
            proactive: {
              action: msg.action,
              trigger: triggerLabel,
              confidence: typeof msg.metadata?.confidence === 'number'
                ? msg.metadata.confidence
                : undefined,
            },
          });

          // Also record interaction with daemon
          apiFetch('/api/proactive/interaction', { method: 'POST' }).catch(() => {});
        }
      }

      lastCheckRef.current = Date.now();
    } catch (error) {
      console.error('[Proactive] Failed to fetch messages:', error);
    } finally {
      isPollingRef.current = false;
    }
  }, [addMessage]);

  // Poll every 15 seconds for proactive messages
  usePolling(fetchAndAddMessages, 15000, { enabled: enabled && connectionStatus === 'connected' });

  return {
    lastCheck: lastCheckRef.current,
  };
}
