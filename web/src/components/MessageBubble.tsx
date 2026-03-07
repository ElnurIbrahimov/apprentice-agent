import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Message, Citation } from '../types';
import { ModelCompare } from './ModelCompare';
import { UserCircleIcon, SparklesIcon, BoltIcon } from '@heroicons/react/24/solid';
import { ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';
import { AttachmentList } from './AttachmentPreview';
import { ToolTrace } from './ToolTrace';

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [codeCopied, setCodeCopied] = useState(false);
  const codeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => { if (codeTimeoutRef.current) clearTimeout(codeTimeoutRef.current); };
  }, []);

  const handleCodeCopy = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(children).catch((e) => console.warn('[Copy] Failed:', e));
    }
    if (codeTimeoutRef.current) clearTimeout(codeTimeoutRef.current);
    setCodeCopied(true);
    codeTimeoutRef.current = setTimeout(() => setCodeCopied(false), 2000);
  };

  return (
    <div className="relative">
      <div className="absolute top-0 right-0 flex items-center gap-1 z-10">
        <button
          onClick={handleCodeCopy}
          aria-label="Copy code"
          className="px-2 py-1 text-xs text-gray-400 bg-gray-700 hover:bg-gray-600 hover:text-gray-200 transition-colors rounded-bl flex items-center gap-1"
        >
          {codeCopied ? (
            <><CheckIcon className="w-3 h-3 text-green-400" /><span className="text-green-400">Copied</span></>
          ) : (
            <><ClipboardDocumentIcon className="w-3 h-3" /><span>{language}</span></>
          )}
        </button>
      </div>
      <pre className="bg-gray-900 p-4 rounded-lg overflow-x-auto pt-8">
        <code className={`language-${language}`}>
          {children}
        </code>
      </pre>
    </div>
  );
}

function CitationList({ citations }: { citations: Citation[] }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? citations : citations.slice(0, 3);
  const hasMore = citations.length > 3;

  return (
    <div className="mt-3 border-t border-gray-700/50 pt-2 text-xs">
      <div className="text-gray-500 mb-1.5 font-medium">Sources</div>
      <ol className="space-y-1">
        {visible.map((c) => (
          <li key={c.id} className="flex items-start gap-1.5">
            <span className="flex-shrink-0 text-gray-500">[{c.id}]</span>
            <a
              href={c.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-chat-accent hover:underline hover:text-chat-accent-hover truncate block"
              title={c.snippet || c.title}
            >
              {c.title || c.url}
            </a>
          </li>
        ))}
      </ol>
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1 text-gray-500 hover:text-gray-300 transition-colors"
        >
          {expanded ? 'Show fewer' : `Show ${citations.length - 3} more source${citations.length - 3 !== 1 ? 's' : ''}...`}
        </button>
      )}
    </div>
  );
}

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
  const [copied, setCopied] = useState(false);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleCopy = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(message.content).catch((e) => console.warn('[Copy] Failed:', e));
    }
    if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    setCopied(true);
    copyTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
  };

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
        <div className="flex-1 min-w-0 group">
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

          {/* Tool trace (above content) */}
          {!isUser && message.toolTrace && message.toolTrace.length > 0 && (
            <ToolTrace traces={message.toolTrace} isStreaming={isStreaming} />
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
                      <CodeBlock language={match[1]}>
                        {String(children).replace(/\n$/, '')}
                      </CodeBlock>
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

          {/* Citations */}
          {!isUser && message.citations && message.citations.length > 0 && !isStreaming && (
            <CitationList citations={message.citations} />
          )}

          {/* Model comparison results */}
          {!isUser && message.compareResults && message.compareResults.length > 0 && (
            <ModelCompare
              results={message.compareResults}
              query={message.content}
              onUseResponse={(response) => navigator.clipboard.writeText(response)}
            />
          )}

          {/* Model badge for assistant messages */}
          {!isUser && message.model_used && !isStreaming && (
            <span className="text-xs text-chat-text-secondary/60 mt-1 inline-block">
              {message.model_used}
            </span>
          )}

          {/* Timestamp + copy */}
          <div className="mt-2 flex items-center gap-2 text-xs text-chat-text-secondary">
            <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
            {!isStreaming && (
              <button
                onClick={handleCopy}
                aria-label="Copy message"
                className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:text-chat-text"
                title="Copy message"
              >
                {copied ? (
                  <CheckIcon className="w-3.5 h-3.5 text-green-400" />
                ) : (
                  <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
