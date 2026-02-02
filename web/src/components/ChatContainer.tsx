import { useRef, useEffect } from 'react';
import { useChatStore } from '../store/chatStore';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { useWebSocket } from '../hooks/useWebSocket';
import { SparklesIcon } from '@heroicons/react/24/outline';

export function ChatContainer() {
  const { messages, isLoading, error, connectionStatus } = useChatStore();
  const { sendMessage } = useWebSocket();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = (message: string) => {
    sendMessage(message);
  };

  const isDisabled = isLoading || connectionStatus !== 'connected';

  return (
    <div className="flex flex-col h-full">
      {/* Connection status banner */}
      {connectionStatus !== 'connected' && (
        <div className={`px-4 py-2 text-center text-sm ${
          connectionStatus === 'connecting'
            ? 'bg-yellow-600 text-white'
            : 'bg-red-600 text-white'
        }`}>
          {connectionStatus === 'connecting' ? (
            <>Connecting to AURA...</>
          ) : (
            <>Disconnected. Attempting to reconnect...</>
          )}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900 text-red-200 text-center text-sm">
          {error}
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          // Empty state
          <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary">
            <div className="w-16 h-16 rounded-full bg-purple-600/20 flex items-center justify-center mb-4">
              <SparklesIcon className="w-8 h-8 text-purple-400" />
            </div>
            <h2 className="text-xl font-semibold text-chat-text mb-2">
              Welcome to AURA
            </h2>
            <p className="text-center max-w-md px-4">
              Autonomous Universal Reasoning Agent. Ask me anything, request a web search,
              run code, or just chat!
            </p>
            <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-3 px-4 max-w-2xl">
              {[
                'What can you do?',
                'Search online for AI news',
                'Calculate factorial of 20',
                'Tell me about yourself',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => handleSend(suggestion)}
                  disabled={isDisabled}
                  className="px-4 py-3 bg-chat-assistant hover:bg-chat-border rounded-lg text-left text-sm text-chat-text transition-colors disabled:opacity-50"
                >
                  {suggestion}
                </button>
              ))}
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
        disabled={isDisabled}
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
