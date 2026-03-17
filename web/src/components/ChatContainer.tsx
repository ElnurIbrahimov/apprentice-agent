import { useRef, useEffect, useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { useWebSocket } from '../hooks/useWebSocket';
import { useProactiveMessages } from '../hooks/useProactiveMessages';
import type { FileAttachment, ModelResult } from '../types';
import {
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  CalculatorIcon,
  CpuChipIcon,
  ChevronDownIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

// Quick action button configurations with icons
const QUICK_ACTIONS = [
  {
    text: 'What can you do?',
    icon: ChatBubbleLeftRightIcon,
  },
  {
    text: 'Search online for AI news',
    icon: MagnifyingGlassIcon,
  },
  {
    text: 'Calculate factorial of 20',
    icon: CalculatorIcon,
  },
  {
    text: 'Tell me about yourself',
    icon: CpuChipIcon,
  },
];

export function ChatContainer() {
  const { messages, isLoading, error, setError, connectionStatus, toolStatus, isSwitchingConversation } = useChatStore();
  const { sendMessage, stopGeneration, connect: reconnect } = useWebSocket();
  const { settings } = useSettingsStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);

  useProactiveMessages(connectionStatus === 'connected');

  // Handle scroll events to detect if user scrolled up
  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    setIsUserScrolledUp(distFromBottom > 150);
  }, []);

  // Smart auto-scroll: only scroll if user is already at bottom and autoScroll is enabled
  useEffect(() => {
    if (settings.autoScroll && !isUserScrolledUp) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isUserScrolledUp, settings.autoScroll]);

  // Auto-scroll to bottom when user sends a new message (always, regardless of autoScroll setting)
  useEffect(() => {
    if (isLoading) {
      setIsUserScrolledUp(false);
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isLoading]);

  // Auto-dismiss error after 10 seconds
  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(null), 10000);
    return () => clearTimeout(timer);
  }, [error, setError]);

  const handleSend = async (message: string, attachments?: FileAttachment[], actionMode?: string | null) => {
    if (actionMode === 'compare') {
      // Route through REST /api/compare instead of WebSocket
      const store = useChatStore.getState();
      store.addMessage({ role: 'user', content: message });
      const assistantId = store.addMessage({ role: 'assistant', content: 'Comparing models...', isStreaming: true });
      store.setIsLoading(true);

      try {
        const res = await fetch('/api/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message }),
        });
        if (!res.ok) {
          throw new Error(`Compare request failed: ${res.status}`);
        }
        const data = await res.json();
        const results: ModelResult[] = data.results ?? [];
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: '', compareResults: results, isStreaming: false }
              : m
          ),
          isLoading: false,
        }));
      } catch {
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: 'Compare failed. Please try again.', isStreaming: false }
              : m
          ),
          isLoading: false,
        }));
      }
      store.setToolStatus(null);
      return;
    }
    // Pass actionMode for auto-model selection (null = use user's selected model)
    sendMessage(message, attachments, undefined, actionMode);
  };

  const isDisabled = isLoading || connectionStatus !== 'connected';

  return (
    <div className="flex flex-col h-full" style={{ background: 'transparent' }}>
      {/* Connection status banner */}
      {connectionStatus !== 'connected' && (
        <div className={`px-4 py-2 text-center text-sm transition-all duration-300 ${
          connectionStatus === 'connecting'
            ? 'bg-yellow-600/90 text-white backdrop-blur-sm'
            : 'bg-red-600/90 text-white backdrop-blur-sm'
        }`}>
          {connectionStatus === 'connecting' ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              Connecting to AURA...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-white rounded-full" />
              Disconnected.
              <button onClick={reconnect} className="underline hover:no-underline font-medium ml-1">
                Reconnect now
              </button>
            </span>
          )}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900/80 text-red-200 text-sm backdrop-blur-sm animate-slide-up-fade flex items-center gap-2">
          <span className="flex-1 text-center">{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss error" className="ml-auto flex-shrink-0">
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto relative">
        {isSwitchingConversation && messages.length === 0 ? (
          <div className="animate-pulse space-y-4 p-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-16 bg-chat-border/20 rounded-lg" />
            ))}
          </div>
        ) : messages.length === 0 ? (
          // Empty state with enhanced styling
          <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary relative px-6">
            {/* NextGen welcome heading */}
            <h1 className="text-5xl font-light tracking-tight mb-4 text-center animate-fade-in"
              style={{
                background: 'linear-gradient(180deg, #fff 0%, rgba(255,255,255,0.4) 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '-0.04em',
              }}
            >
              What will we build today?
            </h1>

            <p className="text-center max-w-md text-chat-text-secondary mb-12 leading-relaxed animate-fade-in animation-delay-100">
              AURA is ready to assist with complex design, deep research, and architectural engineering.
            </p>

            {/* NextGen suggestion cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
              {QUICK_ACTIONS.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.text}
                    onClick={() => handleSend(action.text)}
                    disabled={isDisabled}
                    aria-label={action.text}
                    className={`
                      group flex flex-col gap-3 p-5 text-left
                      disabled:opacity-50 disabled:cursor-not-allowed
                      animate-slide-up-fade transition-all duration-300
                      ${index === 0 ? 'animation-delay-100' : ''}
                      ${index === 1 ? 'animation-delay-200' : ''}
                      ${index === 2 ? 'animation-delay-300' : ''}
                      ${index === 3 ? 'animation-delay-400' : ''}
                    `}
                    style={{
                      background: 'rgba(20, 20, 25, 0.4)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      borderRadius: '12px',
                      backdropFilter: 'blur(12px)',
                    }}
                    onMouseEnter={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)';
                      (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.15)';
                    }}
                    onMouseLeave={e => {
                      (e.currentTarget as HTMLElement).style.background = 'rgba(20, 20, 25, 0.4)';
                      (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.06)';
                    }}
                  >
                    <div style={{
                      background: 'rgba(255,255,255,0.1)',
                      width: 36, height: 36, borderRadius: 8,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      <Icon className="w-4 h-4 text-white" />
                    </div>
                    <span className="text-sm font-medium text-chat-text group-hover:text-white transition-colors">
                      {action.text}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          // Message list
          <div className="pb-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
        {isUserScrolledUp && messages.length > 0 && (
          <button
            onClick={() => {
              setIsUserScrolledUp(false);
              messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="absolute bottom-24 right-6 z-10 p-2 bg-purple-600 hover:bg-purple-500 text-white rounded-full shadow-lg transition-all"
            aria-label="Scroll to bottom"
          >
            <ChevronDownIcon className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Tool status indicator */}
      {toolStatus && (
        <div className="px-4 pb-1 flex items-center gap-2 text-xs text-chat-text-secondary">
          <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-ping inline-block" />
          {toolStatus.action === 'thinking'
            ? 'AURA is thinking...'
            : `Using ${toolStatus.name}...`}
        </div>
      )}

      {/* Input area */}
      <MessageInput
        onSend={handleSend}
        onStop={stopGeneration}
        disabled={isDisabled}
        isLoading={isLoading}
        placeholder={
          connectionStatus !== 'connected'
            ? 'Connecting...'
            : isLoading
            ? 'AURA is thinking...'
            : 'Message AURA...'
        }
      />
    </div>
  );
}
