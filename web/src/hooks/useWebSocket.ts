import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { hasSeenProactive, markProactiveSeen } from './proactiveDedup';
import {
  enqueueMessage, listQueued, removeQueued, queuedCount as readQueuedCount,
} from '../utils/messageQueue';
import type { WebSocketMessage, FileAttachment } from '../types';

const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/chat/stream`;

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
  const responseTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);

  const {
    addMessage,
    appendToMessage,
    setMessageStreaming,
    setMessageModelUsed,
    setCitationsForMessage,
    appendToolTrace,
    setConnectionStatus,
    setMood,
    setIsLoading,
    setError,
    setToolStatus,
    addResearchStep,
    clearResearchProgress,
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

  // Clear response timeout
  const clearResponseTimeout = useCallback(() => {
    if (responseTimeout.current) {
      clearTimeout(responseTimeout.current);
      responseTimeout.current = null;
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
      case 'research_progress':
        if (data.stage) {
          addResearchStep({
            stage: data.stage,
            data: data.data || {},
            timestamp: Date.now(),
          });
        }
        break;

      case 'chunk':
        // When streaming starts, deactivate research progress
        if (!currentMessageId.current) {
          useChatStore.getState().addResearchStep({
            stage: 'synthesis',
            data: {},
            timestamp: Date.now(),
          });
        }
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
        clearResponseTimeout();
        clearResearchProgress();
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          if (data.model_used) {
            setMessageModelUsed(currentMessageId.current, data.model_used);
          }
        }
        if (data.mood) {
          setMood(data.mood);
        }
        if (data.audio_url && useSettingsStore.getState().settings.soundEnabled) {
          try {
            const audio = new Audio(data.audio_url);
            audio.play().catch((e) => console.warn('[TTS] Audio play failed:', e));
          } catch (e) {
            console.warn('[TTS] Failed to create Audio:', e);
          }
        }
        currentMessageId.current = null;
        setIsLoading(false);
        setToolStatus(null);
        break;

      case 'error':
        clearResponseTimeout();
        console.error('[WebSocket] Server error:', data.error);
        setError(data.error || 'Unknown error');
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          appendToMessage(currentMessageId.current, `\n\n*Error: ${data.error}*`);
        }
        currentMessageId.current = null;
        setIsLoading(false);
        setToolStatus(null);
        break;

      case 'stopped':
        clearResponseTimeout();
        if (currentMessageId.current) {
          setMessageStreaming(currentMessageId.current, false);
          appendToMessage(currentMessageId.current, '\n\n*[Generation stopped]*');
        }
        currentMessageId.current = null;
        setIsLoading(false);
        setToolStatus(null);
        break;

      case 'tool_status':
        useChatStore.getState().setToolStatus(
          data.tool_name
            ? { name: data.tool_name, action: data.tool_action || 'working' }
            : null
        );
        break;

      case 'citations':
        if (currentMessageId.current && data.citations) {
          setCitationsForMessage(currentMessageId.current, data.citations);
        }
        break;

      case 'tool_trace':
        if (currentMessageId.current && data.event && data.tool) {
          appendToolTrace(currentMessageId.current, {
            event: data.event,
            tool: data.tool,
            detail: data.detail,
            elapsed_ms: data.elapsed_ms,
            timestamp: Date.now(),
          });
        }
        break;

      case 'proactive': {
        // Real-time push from Gateway Daemon (instant, no polling delay)
        const proactiveId = data.id
          ? String(data.id)
          : `${data.action}:${(data.content || '').slice(0, 40)}`;
        if (hasSeenProactive(proactiveId)) break;
        markProactiveSeen(proactiveId);

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

      case 'hand_event': {
        // Real-time Hand execution result — dispatch custom event for HandsDashboard
        window.dispatchEvent(new CustomEvent('aura:hand_event', { detail: data }));
        // Also show notable results as proactive messages
        const handSuccess = data.success;
        const handName = data.hand || 'unknown';
        const handSummary = data.summary || '';
        if (!handSuccess || (handName === 'guardian' && !handSummary.toUpperCase().includes('ALL CLEAR'))) {
          addMessage({
            role: 'assistant',
            content: `**Hand '${handName}' ${handSuccess ? 'completed' : 'failed'}:**\n\n${handSummary}`,
            proactive: { action: 'notify', trigger: `hand:${handName}` },
          });
        }
        break;
      }

      case 'hand_approval_request': {
        // Surface approval requests as urgent proactive messages
        window.dispatchEvent(new CustomEvent('aura:hand_approval', { detail: data }));
        addMessage({
          role: 'assistant',
          content: `**Approval Required:** Hand '${data.hand_name}' wants to use **${data.tool_name}**. Check the Hands dashboard to approve or deny.`,
          proactive: { action: 'intervene', trigger: 'hand_approval' },
        });
        break;
      }

      case 'conv_sync': {
        // Extension or another surface created/updated/deleted a conversation
        window.dispatchEvent(new CustomEvent('aura:conv-sync', { detail: data }));
        break;
      }

      case 'action_trace': {
        // Live Hand execution step — dispatch for HandsDashboard
        window.dispatchEvent(new CustomEvent('aura:action_trace', { detail: data }));
        break;
      }

      default:
        console.warn('[WebSocket] Unknown message type:', data.type);
    }
  }, [addMessage, appendToMessage, setMessageStreaming, setMessageModelUsed, setCitationsForMessage, appendToolTrace, setMood, setIsLoading, setError, clearResponseTimeout, setToolStatus, addResearchStep, clearResearchProgress]);

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
        // Fire-and-forget queue drain — if there are messages we couldn't send
        // while disconnected, replay them now in order.
        void drainQueue();
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

        clearResponseTimeout();
        setIsLoading(false);  // Clear stuck loading state on disconnect
        setConnectionStatus('disconnected');
        wsRef.current = null;
        stopHeartbeat();

        // Don't reconnect if manually disconnected or unmounted
        if (isManualDisconnect.current || !mountedRef.current) {
          return;
        }

        // Reconnect with backoff
        if (reconnectAttempts.current < MAX_RECONNECT_ATTEMPTS) {
          reconnectAttempts.current++;
          const delay = getReconnectDelay();
          reconnectTimeout.current = setTimeout(() => connectRef.current(), delay);
        } else {
          setConnectionStatus('disconnected');
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

  // Actually push a payload over the wire — assumes the socket is OPEN.
  const wirePayload = useCallback((message: string, attachments: FileAttachment[] | undefined, modelOverride: string | null | undefined, actionMode: string | null | undefined) => {
    const selectedModel = modelOverride !== undefined
      ? modelOverride
      : useChatStore.getState().selectedModel;

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
    wsRef.current?.send(JSON.stringify(payload));
  }, []);

  // Send message — queues to localStorage if offline so nothing gets lost.
  const sendMessage = useCallback((message: string, attachments?: FileAttachment[], modelOverride?: string | null, actionMode?: string | null) => {
    const connected = wsRef.current && wsRef.current.readyState === WebSocket.OPEN;

    addMessage({
      role: 'user',
      content: message,
      attachments,
      actionMode: actionMode ?? null,
    });

    if (!connected) {
      // Queue for replay on reconnect. Don't set loading — we can't show
      // progress for something that hasn't started.
      enqueueMessage({
        message,
        attachments,
        modelOverride,
        actionMode,
        conversationId: useChatStore.getState().currentConversationId,
      });
      setError(`You're offline — message queued (${readQueuedCount()} pending). Will send on reconnect.`);
      return false;
    }

    setIsLoading(true);
    setError(null);
    currentMessageId.current = null;

    clearResponseTimeout();
    responseTimeout.current = setTimeout(() => {
      setIsLoading(false);
      setError('Response timed out. AURA may be busy — please try again.');
      currentMessageId.current = null;
    }, 300000);

    wirePayload(message, attachments, modelOverride, actionMode);
    return true;
  }, [addMessage, setIsLoading, setError, clearResponseTimeout, wirePayload]);

  // Drain any queued messages on reconnect. Paces sends to avoid slamming
  // the server with a burst.
  const drainQueue = useCallback(async () => {
    const queue = listQueued();
    if (queue.length === 0) return;
    for (const q of queue) {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) break;
      wirePayload(q.message, q.attachments, q.modelOverride, q.actionMode);
      removeQueued(q.id);
      await new Promise((r) => setTimeout(r, 400));
    }
  }, [wirePayload]);

  // Disconnect
  const disconnect = useCallback(() => {
    isManualDisconnect.current = true;
    reconnectAttempts.current = 0;
    clearResponseTimeout();

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
  }, [setConnectionStatus, stopHeartbeat, clearResponseTimeout]);

  // Reconnect
  const reconnect = useCallback(() => {
    isManualDisconnect.current = false;
    reconnectAttempts.current = 0;
    connect();
  }, [connect]);

  const connectRef = useRef(connect);
  useEffect(() => { connectRef.current = connect; }, [connect]);

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
      connectRef.current();
    }, 100);

    return () => {
      mountedRef.current = false;
      isManualDisconnect.current = true;
      clearTimeout(timeoutId);
      clearResponseTimeout();
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
  }, [stopHeartbeat, clearResponseTimeout]);

  return {
    sendMessage,
    stopGeneration,
    connect: reconnect,
    disconnect,
    isConnected: useChatStore((state) => state.connectionStatus === 'connected'),
    connectionStatus: useChatStore((state) => state.connectionStatus),
  };
}
