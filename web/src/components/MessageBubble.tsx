import ReactMarkdown from 'react-markdown';
import type { Message } from '../types';
import { UserCircleIcon, SparklesIcon, BoltIcon } from '@heroicons/react/24/solid';
import { AttachmentList } from './AttachmentPreview';

interface MessageBubbleProps {
  message: Message;
}

// Action icons for proactive messages
const PROACTIVE_ICONS: Record<string, string> = {
  notify: '💡',
  suggest: '✨',
  remind: '⏰',
  ask: '🤔',
  intervene: '⚡',
  prepare: '📋',
};

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;
  const isProactive = !!message.proactive;

  return (
    <div
      className={`py-6 px-4 md:px-8 ${
        isUser ? 'bg-chat-user' : isProactive ? 'bg-gradient-to-r from-purple-900/20 to-chat-assistant' : 'bg-chat-assistant'
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0 relative">
          {isUser ? (
            <div className="w-8 h-8 rounded-full bg-chat-accent flex items-center justify-center">
              <UserCircleIcon className="w-6 h-6 text-white" />
            </div>
          ) : (
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isProactive ? 'bg-gradient-to-br from-purple-500 to-pink-500' : 'bg-purple-600'
            }`}>
              {isProactive ? (
                <BoltIcon className="w-5 h-5 text-white" />
              ) : (
                <SparklesIcon className="w-5 h-5 text-white" />
              )}
            </div>
          )}
          {/* Proactive indicator pulse */}
          {isProactive && (
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-pink-500"></span>
            </span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Role label with proactive badge */}
          <div className="flex items-center gap-2 mb-1">
            <span className="text-chat-text font-medium">
              {isUser ? 'You' : 'AURA'}
            </span>
            {isProactive && message.proactive && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-purple-500/20 text-purple-300 border border-purple-500/30">
                <span>{PROACTIVE_ICONS[message.proactive.action] || '💭'}</span>
                <span>{message.proactive.trigger || 'initiated'}</span>
              </span>
            )}
          </div>

          {/* Attachments (for user messages) */}
          {isUser && message.attachments && message.attachments.length > 0 && (
            <AttachmentList attachments={message.attachments} compact />
          )}

          {/* Message content */}
          <div className="prose prose-invert max-w-none text-chat-text">
            {isUser ? (
              <p className="whitespace-pre-wrap">{message.content}</p>
            ) : (
              <ReactMarkdown
                components={{
                  // Custom rendering for code blocks
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const isInline = !match;

                    if (isInline) {
                      return (
                        <code
                          className="bg-gray-800 px-1.5 py-0.5 rounded text-sm"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }

                    return (
                      <div className="relative">
                        <div className="absolute top-0 right-0 px-2 py-1 text-xs text-gray-400 bg-gray-700 rounded-bl">
                          {match[1]}
                        </div>
                        <pre className="bg-gray-900 p-4 rounded-lg overflow-x-auto">
                          <code className={className} {...props}>
                            {children}
                          </code>
                        </pre>
                      </div>
                    );
                  },
                  // Custom link rendering
                  a({ href, children }) {
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-chat-accent hover:text-chat-accent-hover underline"
                      >
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}

            {/* Streaming cursor */}
            {isStreaming && (
              <span className="typing-cursor inline-block w-2 h-4 bg-chat-accent ml-1" />
            )}
          </div>

          {/* Timestamp */}
          <div className="mt-2 text-xs text-chat-text-secondary">
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
}
