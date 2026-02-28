import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import type { WebSocketMessage, FileAttachment } from '../types';

// Connect directly to backend — CORS does not apply to WebSocket in Starlette 0.52+
// (CORSMiddleware passes websocket scope through unchanged, and auth middleware also bypasses ws)
const WS_URL = 'ws://127.0.0.1:8000/api/chat/stream';

const INITIAL_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const MAX_RECONNECT_ATTEMPTS = 10;
const HEARTBEAT_INTERVAL = 30000;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentMessageId = useRef<string | null>(null);
  const isManualDisconnect = useRef(false);
  const mountedRef = useRef(true);

  const {
    addMessage,
    appendToMessage,
    setMessageStreaming,
    setConnectionStatus,
    setMood,
    setIsLoading,
    setError,
  } = useChatStore();

  // Calculate exponential backoff delay
  const getReconnectDelay = useCallback(() => {
    const delay = INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttempts.current);
    return Math.min(delay, MAX_RECONNECT_DELAY);
  }, []);

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (heartbeatInterval.current) {
      clearInterval(heartbeatInterval.current);
      heartbeatInterval.current = null;
    }
  }, []);

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    stopHeartbeat();
    heartbeatInterval.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'ping' }));
        } catch (e) {
          console.warn('[WebSocket] Heartbeat failed');
        }
      }
    }, HEARTBEAT_INTERVAL);
  }, [stopHeartbeat]);

  // Handle incoming messages - defined before connect to avoid closure issues
  const handleMessage = useCallback((data: WebSocketMessage) => {
    switch (data.type) {
      case 'chunk':
        if (data.content) {
          if (!currentMessageId.current) {
            currentMessageId.current = addMessage({
              role: 'assistant',
              content: data.content,
              isStreaming: true,
            });
          } else {
            appendToMessage(currentMessageId.current, data.content);
          }
        }
        break;

      case 'done':
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
        }
        if (data.mood) {
          setMood(data.mood);
        }
        currentMessageId.current = null;
        setIsLoading(false);
        break;

      case 'error':
        console.error('[WebSocket] Server error:', data.error);
        setError(data.error || 'Unknown error');
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          appendToMessage(currentMessageId.current, `\n\n*Error: ${data.error}*`);
        }
        currentMessageId.current = null;
        setIsLoading(false);
        break;

      case 'stopped':
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          appendToMessage(currentMessageId.current, '\n\n*[Generation stopped]*');
        }
        currentMessageId.current = null;
        setIsLoading(false);
        break;

      case 'proactive': {
        // Real-time push from Gateway Daemon (instant, no polling delay)
        const actionLabels: Record<string, string> = {
          notify: 'noticed something',
          suggest: 'has a suggestion',
          remind: 'wants to remind you',
          ask: 'is curious',
          intervene: 'needs your attention',
          prepare: 'prepared something',
        };
        const triggerLabel = actionLabels[data.action || ''] || 'wants to share';
        addMessage({
          role: 'assistant',
          content: data.content || '',
          proactive: {
            action: data.action || 'notify',
            trigger: triggerLabel,
            confidence: typeof data.metadata?.confidence === 'number'
              ? data.metadata.confidence
              : undefined,
          },
        });
        break;
      }
    }
  }, [addMessage, appendToMessage, setMessageStreaming, setMood, setIsLoading, setError]);

  // Connect function
  const connect = useCallback(() => {
    // Check if already connected or connecting
    if (wsRef.current) {
      const state = wsRef.current.readyState;
      if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) {
        return;
      }
    }

    // Check if manually disconnected
    if (isManualDisconnect.current) {
      return;
    }

    // Check if unmounted
    if (!mountedRef.current) {
      return;
    }

    setConnectionStatus('connecting');

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        if (!mountedRef.current) {
          ws.close();
          return;
        }
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        setError(null);
        startHeartbeat();
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          if (data.type === 'pong') return;
          handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Parse error:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error);
        setConnectionStatus('error');
      };

      ws.onclose = () => {
        // Only handle if this is our current socket
        if (wsRef.current !== ws) {
          return;
        }

        setConnectionStatus('disconnected');
        wsRef.current = null;
        stopHeartbeat();

        // Don't reconnect if manually disconnected or unmounted
        if (isManualDisconnect.current || !mountedRef.current) {
          return;
        }

        // Reconnect with backoff
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = getReconnectDelay();
          reconnectAttempts.current++;
          reconnectTimeout.current = setTimeout(connect, delay);
        } else {
          setError('Connection lost. Click to reconnect.');
        }
      };

      wsRef.current = ws;

    } catch (e) {
      console.error('[WebSocket] Failed to create WebSocket:', e);
      setConnectionStatus('error');
      setError('Failed to create WebSocket connection');
    }
  }, [setConnectionStatus, setError, startHeartbeat, stopHeartbeat, getReconnectDelay, handleMessage]);

  // Send message
  const sendMessage = useCallback((message: string, attachments?: FileAttachment[], modelOverride?: string | null, actionMode?: string | null) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return false;
    }

    addMessage({
      role: 'user',
      content: message,
      attachments,
    });

    setIsLoading(true);
    setError(null);
    currentMessageId.current = null;

    const selectedModel = modelOverride !== undefined
      ? modelOverride
      : (actionMode ? null : useChatStore.getState().selectedModel);

    const payload: {
      type: string;
      message: string;
      model?: string;
      action_mode?: string;
      conversation_id?: string;
      attachments?: Array<{ id: string; filename: string; type: string; path?: string }>;
    } = { type: 'chat', message };

    if (selectedModel) payload.model = selectedModel;
    if (actionMode) payload.action_mode = actionMode;

    // Include conversation_id for multi-conversation support
    const conversationId = useChatStore.getState().currentConversationId;
    if (conversationId) payload.conversation_id = conversationId;
    if (attachments?.length) {
      payload.attachments = attachments.map(a => ({
        id: a.id,
        filename: a.filename,
        type: a.type,
        path: a.path,
      }));
    }

    wsRef.current.send(JSON.stringify(payload));
    return true;
  }, [addMessage, setIsLoading, setError]);

  // Disconnect
  const disconnect = useCallback(() => {
    isManualDisconnect.current = true;
    reconnectAttempts.current = 0;

    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    stopHeartbeat();

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, [setConnectionStatus, stopHeartbeat]);

  // Reconnect
  const reconnect = useCallback(() => {
    isManualDisconnect.current = false;
    reconnectAttempts.current = 0;
    connect();
  }, [connect]);

  // Stop generation
  const stopGeneration = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }
    wsRef.current.send(JSON.stringify({ type: 'stop' }));
    return true;
  }, []);

  // Connect on mount
  useEffect(() => {
    mountedRef.current = true;
    isManualDisconnect.current = false;

    // Small delay to ensure DOM is ready
    const timeoutId = setTimeout(() => {
      connect();
    }, 100);

    return () => {
      mountedRef.current = false;
      isManualDisconnect.current = true;
      clearTimeout(timeoutId);
      stopHeartbeat();

      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }

      if (wsRef.current) {
        const ws = wsRef.current;
        wsRef.current = null;
        ws.close();
      }
    };
  }, [connect, stopHeartbeat]);

  return {
    sendMessage,
    stopGeneration,
    connect: reconnect,
    disconnect,
    isConnected: useChatStore((state) => state.connectionStatus === 'connected'),
    connectionStatus: useChatStore((state) => state.connectionStatus),
  };
}
