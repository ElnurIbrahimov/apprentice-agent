import { useEffect, useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { usePolling } from './usePolling';

interface ConversationStarter {
  type: string;
  content: string;
  timestamp: string;
  metadata: {
    focus_topics?: string[];
    emotion?: string;
    idle_seconds?: number;
  };
}

/**
 * Hook to poll for spontaneous conversation starters from AURA
 * and inject them into the chat as proactive messages.
 */
export function useConversationStarters(enabled: boolean = true) {
  const { addMessage, isLoading } = useChatStore();
  const lastStarterTime = useRef<string | null>(null);
  const lastActivityTime = useRef<number>(Date.now());

  // Track user activity
  useEffect(() => {
    const handleActivity = () => {
      lastActivityTime.current = Date.now();
    };

    window.addEventListener('mousemove', handleActivity);
    window.addEventListener('keydown', handleActivity);
    window.addEventListener('click', handleActivity);

    return () => {
      window.removeEventListener('mousemove', handleActivity);
      window.removeEventListener('keydown', handleActivity);
      window.removeEventListener('click', handleActivity);
    };
  }, []);

  // Generate starter based on context
  const generateStarter = useCallback(async () => {
    if (!enabled || isLoading) return;

    try {
      const idleSeconds = (Date.now() - lastActivityTime.current) / 1000;

      const response = await fetch('/api/conversation/starter/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          idle_seconds: idleSeconds,
          force: false,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.generated && data.starter) {
          const starter: ConversationStarter = data.starter;

          // Avoid duplicates
          if (starter.timestamp !== lastStarterTime.current) {
            lastStarterTime.current = starter.timestamp;

            // Add as a proactive message
            addMessage({
              role: 'assistant',
              content: starter.content,
              proactive: {
                action: 'conversation_starter',
                trigger: starter.type,
                confidence: 0.8,
              },
            });
          }
        }
      }
    } catch (e) {
      // Silently ignore - not critical
    }
  }, [enabled, isLoading, addMessage]);

  // Poll for pending starters
  const checkPending = useCallback(async () => {
    if (!enabled || isLoading) return;

    try {
      const response = await fetch('/api/conversation/starter/pending');
      if (response.ok) {
        const data = await response.json();
        if (data.has_starter && data.starter) {
          const starter: ConversationStarter = data.starter;

          // Avoid duplicates
          if (starter.timestamp !== lastStarterTime.current) {
            lastStarterTime.current = starter.timestamp;

            // Add as a proactive message
            addMessage({
              role: 'assistant',
              content: starter.content,
              proactive: {
                action: 'conversation_starter',
                trigger: starter.type,
                confidence: 0.8,
              },
            });
          }
        }
      }
    } catch (e) {
      // Silently ignore
    }
  }, [enabled, isLoading, addMessage]);

  // Poll for pending starters every 30 seconds
  usePolling(checkPending, 30000, { enabled });

  // Try to generate starters every 2 minutes (if conditions are met)
  usePolling(generateStarter, 120000, { enabled });

  // Manual trigger for testing
  const triggerStarter = useCallback(async (force: boolean = true) => {
    try {
      const response = await fetch('/api/conversation/starter/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.generated && data.starter) {
          addMessage({
            role: 'assistant',
            content: data.starter.content,
            proactive: {
              action: 'conversation_starter',
              trigger: data.starter.type,
              confidence: 0.8,
            },
          });
          return true;
        }
      }
    } catch (e) {
      console.error('Failed to trigger starter:', e);
    }
    return false;
  }, [addMessage]);

  return { triggerStarter };
}
