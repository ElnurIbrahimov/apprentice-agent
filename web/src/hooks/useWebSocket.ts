import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import type { WebSocketMessage, FileAttachment } from '../types';

const WS_URL = `ws://${window.location.hostname}:${window.location.port || '8000'}/api/chat/stream`;
const INITIAL_RECONNECT_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const MAX_RECONNECT_ATTEMPTS = 10;
const HEARTBEAT_INTERVAL = 30000; // 30 seconds

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const currentMessageId = useRef<string | null>(null);
  const isManualDisconnect = useRef(false);

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

  // Start heartbeat to detect stale connections
  const startHeartbeat = useCallback(() => {
    if (heartbeatInterval.current) {
      clearInterval(heartbeatInterval.current);
    }
    heartbeatInterval.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        try {
          wsRef.current.send(JSON.stringify({ type: 'ping' }));
        } catch (e) {
          console.warn('[WebSocket] Heartbeat failed, connection may be stale');
        }
      }
    }, HEARTBEAT_INTERVAL);
  }, []);

  const stopHeartbeat = useCallback(() => {
    if (heartbeatInterval.current) {
      clearInterval(heartbeatInterval.current);
      heartbeatInterval.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    // Prevent multiple simultaneous connections
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Don't reconnect if manually disconnected
    if (isManualDisconnect.current) {
      return;
    }

    setConnectionStatus('connecting');
    console.log('[WebSocket] Connecting to', WS_URL);

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setConnectionStatus('connected');
        reconnectAttempts.current = 0; // Reset on successful connection
        setError(null);
        startHeartbeat();
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          // Ignore pong responses
          if (data.type === 'pong') return;
          handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setConnectionStatus('error');
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;
        stopHeartbeat();

        // Don't reconnect if manually disconnected or clean close
        if (isManualDisconnect.current) {
          return;
        }

        // Attempt to reconnect with exponential backoff
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          const delay = getReconnectDelay();
          reconnectAttempts.current++;
          console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current}/${MAX_RECONNECT_ATTEMPTS})`);

          // Clear any existing timeout
          if (reconnectTimeout.current) {
            clearTimeout(reconnectTimeout.current);
          }
          reconnectTimeout.current = setTimeout(connect, delay);
        } else {
          setError('Connection lost. Click to reconnect.');
        }
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('[WebSocket] Failed to create connection:', e);
      setConnectionStatus('error');
      setError('Failed to create WebSocket connection');
    }
  }, [setConnectionStatus, setError, startHeartbeat, stopHeartbeat, getReconnectDelay]);

  const handleMessage = useCallback((data: WebSocketMessage) => {
    switch (data.type) {
      case 'chunk':
        if (data.content) {
          if (!currentMessageId.current) {
            // First chunk - create message
            currentMessageId.current = addMessage({
              role: 'assistant',
              content: data.content,
              isStreaming: true,
            });
          } else {
            // Append to existing message
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
        console.log('[WebSocket] Generation stopped by user');
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          appendToMessage(currentMessageId.current, '\n\n*[Generation stopped]*');
        }
        currentMessageId.current = null;
        setIsLoading(false);
        break;
    }
  }, [addMessage, appendToMessage, setMessageStreaming, setMood, setIsLoading, setError]);

  const sendMessage = useCallback((message: string, attachments?: FileAttachment[], modelOverride?: string | null, actionMode?: string | null) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return false;
    }

    // Add user message to store (with attachments for display)
    addMessage({
      role: 'user',
      content: message,
      attachments,
    });

    setIsLoading(true);
    setError(null);
    currentMessageId.current = null;

    // Get selected model from store if not provided
    // Note: If actionMode is set, backend will auto-select the best model
    const selectedModel = modelOverride !== undefined
      ? modelOverride
      : (actionMode ? null : useChatStore.getState().selectedModel); // Don't send user model if action mode is active

    // Build payload with attachments metadata for server processing
    const payload: {
      type: string;
      message: string;
      model?: string;
      action_mode?: string;
      attachments?: Array<{
        id: string;
        filename: string;
        type: string;
        path?: string;
      }>;
    } = {
      type: 'chat',
      message,
    };

    if (selectedModel) {
      payload.model = selectedModel;
    }

    // Include action mode for auto-model selection
    if (actionMode) {
      payload.action_mode = actionMode;
    }

    // Include attachment metadata for server-side processing
    if (attachments && attachments.length > 0) {
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

  const disconnect = useCallback(() => {
    isManualDisconnect.current = true;
    reconnectAttempts.current = 0; // Reset attempts on manual disconnect

    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    stopHeartbeat();

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected'); // Clean close
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, [setConnectionStatus, stopHeartbeat]);

  const reconnect = useCallback(() => {
    isManualDisconnect.current = false;
    reconnectAttempts.current = 0;
    connect();
  }, [connect]);

  const stopGeneration = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot stop - not connected');
      return false;
    }

    console.log('[WebSocket] Sending stop request');
    wsRef.current.send(JSON.stringify({ type: 'stop' }));
    return true;
  }, []);

  // Connect on mount
  useEffect(() => {
    isManualDisconnect.current = false;
    connect();

    return () => {
      isManualDisconnect.current = true;
      stopHeartbeat();
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
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
