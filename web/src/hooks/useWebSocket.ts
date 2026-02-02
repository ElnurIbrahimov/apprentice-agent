import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import type { WebSocketMessage } from '../types';

const WS_URL = `ws://${window.location.hostname}:${window.location.port || '8000'}/api/chat/stream`;
const RECONNECT_DELAY = 3000;
const MAX_RECONNECT_ATTEMPTS = 5;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const currentMessageId = useRef<string | null>(null);

  const {
    addMessage,
    appendToMessage,
    setMessageStreaming,
    setConnectionStatus,
    setMood,
    setIsLoading,
    setError,
  } = useChatStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionStatus('connecting');
    console.log('[WebSocket] Connecting to', WS_URL);

    try {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data: WebSocketMessage = JSON.parse(event.data);
          handleMessage(data);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
        setConnectionStatus('error');
        setError('Connection error');
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Closed:', event.code, event.reason);
        setConnectionStatus('disconnected');
        wsRef.current = null;

        // Attempt to reconnect
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          console.log(`[WebSocket] Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts.current})`);
          reconnectTimeout.current = setTimeout(connect, RECONNECT_DELAY);
        } else {
          setError('Failed to connect after multiple attempts');
        }
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('[WebSocket] Failed to create connection:', e);
      setConnectionStatus('error');
      setError('Failed to create WebSocket connection');
    }
  }, [setConnectionStatus, setError]);

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
    }
  }, [addMessage, appendToMessage, setMessageStreaming, setMood, setIsLoading, setError]);

  const sendMessage = useCallback((message: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      return false;
    }

    // Add user message to store
    addMessage({
      role: 'user',
      content: message,
    });

    setIsLoading(true);
    setError(null);
    currentMessageId.current = null;

    // Send to server
    wsRef.current.send(JSON.stringify({
      type: 'chat',
      message,
    }));

    return true;
  }, [addMessage, setIsLoading, setError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, [setConnectionStatus]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    sendMessage,
    connect,
    disconnect,
    isConnected: useChatStore((state) => state.connectionStatus === 'connected'),
    connectionStatus: useChatStore((state) => state.connectionStatus),
  };
}
