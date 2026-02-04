import ReactMarkdown from 'react-markdown';
import type { Message } from '../types';
import { UserCircleIcon, SparklesIcon } from '@heroicons/react/24/solid';
import { AttachmentList } from './AttachmentPreview';

interface MessageBubbleProps {
  message: Message;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;

  return (
    <div
      className={`py-6 px-4 md:px-8 ${
        isUser ? 'bg-chat-user' : 'bg-chat-assistant'
      }`}
    >
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar */}
        <div className="flex-shrink-0">
          {isUser ? (
            <div className="w-8 h-8 rounded-full bg-chat-accent flex items-center justify-center">
              <UserCircleIcon className="w-6 h-6 text-white" />
            </div>
          ) : (
            <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
              <SparklesIcon className="w-5 h-5 text-white" />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Role label */}
          <div className="text-chat-text font-medium mb-1">
            {isUser ? 'You' : 'AURA'}
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
