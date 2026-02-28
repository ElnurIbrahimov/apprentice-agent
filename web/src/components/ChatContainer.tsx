import { useRef, useEffect, useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { useWebSocket } from '../hooks/useWebSocket';
import { useProactiveMessages } from '../hooks/useProactiveMessages';
// import { useConversationStarters } from '../hooks/useConversationStarters';
import type { FileAttachment } from '../types';
import {
  SparklesIcon,
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  CalculatorIcon,
  CpuChipIcon,
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
  const { messages, isLoading, error, connectionStatus } = useChatStore();
  const { sendMessage, stopGeneration } = useWebSocket();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);

  useProactiveMessages(connectionStatus === 'connected');
  // useConversationStarters(connectionStatus === 'connected');

  // Check if user is near the bottom of scroll area
  const checkIfNearBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;

    const threshold = 100; // pixels from bottom
    const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
    return isNearBottom;
  }, []);

  // Handle scroll events to detect if user scrolled up
  const handleScroll = useCallback(() => {
    const isNearBottom = checkIfNearBottom();
    setIsUserScrolledUp(!isNearBottom);
  }, [checkIfNearBottom]);

  // Smart auto-scroll: only scroll if user is already at bottom
  useEffect(() => {
    if (!isUserScrolledUp) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isUserScrolledUp]);

  // Auto-scroll to bottom when user sends a new message
  useEffect(() => {
    if (isLoading) {
      setIsUserScrolledUp(false);
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isLoading]);

  const handleSend = (message: string, attachments?: FileAttachment[], actionMode?: string | null) => {
    // Pass actionMode for auto-model selection (null = use user's selected model)
    sendMessage(message, attachments, undefined, actionMode);
  };

  const isDisabled = isLoading || connectionStatus !== 'connected';

  return (
    <div className="flex flex-col h-full bg-radial-content">
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
              Disconnected. Attempting to reconnect...
            </span>
          )}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900/80 text-red-200 text-center text-sm backdrop-blur-sm animate-slide-up-fade">
          {error}
        </div>
      )}

      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          // Empty state with enhanced styling
          <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary relative">
            {/* Radial purple glow behind welcome area */}
            <div className="absolute inset-0 bg-radial-welcome pointer-events-none" />

            {/* Animated sparkle icon */}
            <div className="relative z-10 w-20 h-20 rounded-full bg-gradient-to-br from-purple-600/30 to-blue-600/20 flex items-center justify-center mb-6 animate-pulse-glow">
              <SparklesIcon className="w-10 h-10 text-purple-400 icon-glow-pulse" />
            </div>

            {/* Enhanced welcome heading */}
            <h2 className="relative z-10 text-3xl font-bold mb-3 text-gradient-purple animate-fade-in">
              Welcome to AURA
            </h2>

            {/* Subtitle with better spacing */}
            <p className="relative z-10 text-center max-w-md px-4 text-chat-text-secondary font-light tracking-wide leading-relaxed animate-fade-in animation-delay-100">
              Autonomous Universal Reasoning Agent. Ask me anything, request a web search,
              run code, or just chat!
            </p>

            {/* Quick action buttons with staggered animation */}
            <div className="relative z-10 mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4 px-4 max-w-2xl">
              {QUICK_ACTIONS.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.text}
                    onClick={() => handleSend(action.text)}
                    disabled={isDisabled}
                    className={`
                      group flex items-center gap-3 px-5 py-4
                      btn-glass rounded-xl text-left text-sm text-chat-text
                      disabled:opacity-50 disabled:cursor-not-allowed
                      animate-slide-up-fade
                      ${index === 0 ? 'animation-delay-100' : ''}
                      ${index === 1 ? 'animation-delay-200' : ''}
                      ${index === 2 ? 'animation-delay-300' : ''}
                      ${index === 3 ? 'animation-delay-400' : ''}
                    `}
                  >
                    <div className="w-8 h-8 rounded-lg bg-purple-600/20 flex items-center justify-center flex-shrink-0 transition-all duration-300 group-hover:bg-purple-600/40 group-hover:scale-110">
                      <Icon className="w-4 h-4 text-purple-400" />
                    </div>
                    <span className="transition-colors duration-200 group-hover:text-white">
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
      </div>

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
