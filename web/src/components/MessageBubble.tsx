import { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import type { Message, Citation } from '../types';
import { useChatStore } from '../store/chatStore';
import { ModelCompare } from './ModelCompare';
import { SparklesIcon, BoltIcon } from '@heroicons/react/24/solid';
import { ClipboardDocumentIcon, CheckIcon, ClipboardIcon, ArrowPathIcon, ShareIcon } from '@heroicons/react/24/outline';
import { AttachmentList } from './AttachmentPreview';
import { ToolTrace } from './ToolTrace';
import { MemoryIndicator } from './MemoryIndicator';
import { haptic } from '../utils/haptics';

function CodeBlock({ language, children }: { language: string; children: string }) {
  const [codeCopied, setCodeCopied] = useState(false);
  const codeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => { if (codeTimeoutRef.current) clearTimeout(codeTimeoutRef.current); };
  }, []);

  const handleCodeCopy = async () => {
    try {
      await navigator.clipboard.writeText(children);
      if (codeTimeoutRef.current) clearTimeout(codeTimeoutRef.current);
      setCodeCopied(true);
      codeTimeoutRef.current = setTimeout(() => setCodeCopied(false), 2000);
    } catch (e) {
      console.warn('[Copy] Failed:', e);
    }
  };

  return (
    <div className="relative rounded-lg border border-white/[0.06] overflow-hidden" style={{ background: '#0d0d14' }}>
      {/* Top bar: language badge left, copy button right */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06]">
        <span
          className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full"
          style={{ background: 'rgba(139,92,246,0.25)', color: '#c4b5fd' }}
        >
          {language}
        </span>
        <button
          onClick={handleCodeCopy}
          aria-label="Copy code"
          className="flex items-center gap-1 px-2 py-1 text-xs text-gray-400 hover:text-gray-200 hover:bg-white/[0.06] active:bg-white/[0.1] transition-colors rounded code-copy-btn touch-target"
        >
          {codeCopied ? (
            <><CheckIcon className="w-3.5 h-3.5 text-green-400" /><span className="text-green-400">Copied</span></>
          ) : (
            <><ClipboardDocumentIcon className="w-3.5 h-3.5" /><span>Copy</span></>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto m-0" style={{ background: '#0d0d14' }}>
        <code className={`language-${language}`}>
          {children}
        </code>
      </pre>
    </div>
  );
}

/* ── Citation tooltip shown on hover of inline [N] badges ── */
function CitationTooltip({ citation, position }: { citation: Citation; position: { x: number; y: number } }) {
  // Clamp position to viewport
  const left = Math.min(position.x, window.innerWidth - 280);
  const top = position.y + 8;

  return (
    <div
      className="fixed z-[1100] pointer-events-none animate-fade-in"
      style={{
        left,
        top,
        width: 260,
        background: 'rgba(15, 15, 22, 0.95)',
        border: '1px solid rgba(139, 92, 246, 0.3)',
        borderRadius: 10,
        padding: '10px 12px',
        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
        backdropFilter: 'blur(16px)',
      }}
    >
      <div className="text-xs font-medium text-purple-300 truncate">{citation.title || citation.url}</div>
      {citation.snippet && (
        <div className="text-[11px] text-gray-400 mt-1.5 leading-relaxed line-clamp-3">{citation.snippet}</div>
      )}
      <div className="text-[10px] text-gray-600 mt-1.5 truncate">{citation.url}</div>
    </div>
  );
}

/* ── Inline [N] badge rendered inside message text ── */
function InlineCitationBadge({
  num,
  citation,
  messageId,
}: {
  num: number;
  citation: Citation | undefined;
  messageId: string;
}) {
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const setHoveredCitation = useChatStore((s) => s.setHoveredCitation);
  const setCitationsPanelOpen = useChatStore((s) => s.setCitationsPanelOpen);
  const setActiveCitationRef = useChatStore((s) => s.setActiveCitationRef);
  const hoveredCitation = useChatStore((s) => s.hoveredCitation);

  const isHighlighted =
    hoveredCitation !== null &&
    hoveredCitation.id === num &&
    hoveredCitation.messageId === messageId;

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent) => {
      const rect = (e.target as HTMLElement).getBoundingClientRect();
      setTooltipPos({ x: rect.left, y: rect.bottom });
      setHoveredCitation({ id: num, messageId });
    },
    [num, messageId, setHoveredCitation]
  );

  const handleMouseLeave = useCallback(() => {
    setTooltipPos(null);
    setHoveredCitation(null);
  }, [setHoveredCitation]);

  const handleClick = useCallback(() => {
    // Open panel and scroll to this citation
    setCitationsPanelOpen(true);
    setActiveCitationRef({ id: num, messageId });
  }, [num, messageId, setCitationsPanelOpen, setActiveCitationRef]);

  return (
    <>
      <button
        type="button"
        onClick={handleClick}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        data-citation-id={num}
        data-message-id={messageId}
        className="inline-flex items-center justify-center align-super cursor-pointer transition-all duration-150"
        style={{
          width: 18,
          height: 18,
          fontSize: 10,
          fontWeight: 700,
          lineHeight: 1,
          borderRadius: '50%',
          background: isHighlighted ? 'rgba(139, 92, 246, 0.5)' : 'rgba(139, 92, 246, 0.25)',
          color: isHighlighted ? '#e9d5ff' : '#c4b5fd',
          border: `1px solid ${isHighlighted ? 'rgba(139, 92, 246, 0.7)' : 'rgba(139, 92, 246, 0.35)'}`,
          marginLeft: 1,
          marginRight: 1,
          verticalAlign: 'super',
          transform: isHighlighted ? 'scale(1.15)' : 'scale(1)',
        }}
        aria-label={citation ? `Source ${num}: ${citation.title}` : `Source ${num}`}
      >
        {num}
      </button>
      {tooltipPos && citation && <CitationTooltip citation={citation} position={tooltipPos} />}
    </>
  );
}

/* ── Compact source list below message (enhanced) ── */
function CitationList({ citations, messageId }: { citations: Citation[]; messageId: string }) {
  const [expanded, setExpanded] = useState(false);
  const visible = expanded ? citations : citations.slice(0, 3);
  const hasMore = citations.length > 3;
  const setCitationsPanelOpen = useChatStore((s) => s.setCitationsPanelOpen);
  const hoveredCitation = useChatStore((s) => s.hoveredCitation);
  const setHoveredCitation = useChatStore((s) => s.setHoveredCitation);

  return (
    <div className="mt-3 border-t border-gray-700/50 pt-2 text-xs">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-gray-500 font-medium">Sources</span>
        <button
          onClick={() => setCitationsPanelOpen(true)}
          className="text-[10px] text-purple-400 hover:text-purple-300 transition-colors"
        >
          View all
        </button>
      </div>
      <ol className="space-y-1">
        {visible.map((c) => {
          const isHighlighted =
            hoveredCitation !== null &&
            hoveredCitation.id === c.id &&
            hoveredCitation.messageId === messageId;
          return (
            <li
              key={c.id}
              className="flex items-start gap-1.5 rounded px-1 py-0.5 transition-colors"
              style={{
                background: isHighlighted ? 'rgba(139, 92, 246, 0.12)' : 'transparent',
              }}
              onMouseEnter={() => setHoveredCitation({ id: c.id, messageId })}
              onMouseLeave={() => setHoveredCitation(null)}
            >
              <span
                className="flex-shrink-0 inline-flex items-center justify-center rounded-full text-[10px] font-bold"
                style={{
                  width: 18,
                  height: 18,
                  background: isHighlighted ? 'rgba(139, 92, 246, 0.4)' : 'rgba(255, 255, 255, 0.08)',
                  color: isHighlighted ? '#e9d5ff' : '#a1a1aa',
                }}
              >
                {c.id}
              </span>
              <a
                href={/^https?:\/\//i.test(c.url) ? c.url : '#'}
                target="_blank"
                rel="noopener noreferrer"
                className="text-chat-accent hover:underline hover:text-chat-accent-hover truncate block"
                title={c.snippet || c.title}
              >
                {c.title || c.url}
              </a>
            </li>
          );
        })}
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

/* ── Parse message content to inject inline citation badges ── */
function renderContentWithCitations(
  content: string,
  citations: Citation[] | undefined,
  messageId: string,
): React.ReactNode[] {
  if (!citations || citations.length === 0) return [content];

  const citationMap = new Map(citations.map((c) => [c.id, c]));
  // Split on [N] patterns, keeping the delimiters
  const parts = content.split(/(\[\d+\])/g);

  return parts.map((part, i) => {
    const match = /^\[(\d+)\]$/.exec(part);
    if (match) {
      const num = parseInt(match[1], 10);
      return (
        <InlineCitationBadge
          key={`cite-${messageId}-${num}-${i}`}
          num={num}
          citation={citationMap.get(num)}
          messageId={messageId}
        />
      );
    }
    return part;
  });
}

/* ── Markdown renderer that injects inline citation badges into text nodes ── */
function CitationAwareMarkdown({
  content,
  citations,
  messageId,
}: {
  content: string;
  citations: Citation[];
  messageId: string;
}) {
  return (
    <ReactMarkdown
      components={{
        code({ className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          if (!match) {
            return (
              <code className="bg-gray-800 px-1.5 py-0.5 rounded text-sm" {...props}>
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
        a({ href, children }) {
          return (
            <a href={href} target="_blank" rel="noopener noreferrer"
              className="text-purple-400 hover:text-purple-300 underline">
              {children}
            </a>
          );
        },
        // Inject citation badges into paragraph text
        p({ children }) {
          return <p>{injectCitationsIntoChildren(children, citations, messageId)}</p>;
        },
        li({ children }) {
          return <li>{injectCitationsIntoChildren(children, citations, messageId)}</li>;
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/* ── Walk React children and replace [N] patterns in text strings ── */
function injectCitationsIntoChildren(
  children: React.ReactNode,
  citations: Citation[],
  messageId: string,
): React.ReactNode {
  if (!Array.isArray(children)) {
    if (typeof children === 'string') {
      return renderContentWithCitations(children, citations, messageId);
    }
    return children;
  }
  return children.map((child, i) => {
    if (typeof child === 'string') {
      const parts = renderContentWithCitations(child, citations, messageId);
      // If no citations found, return the string as-is
      if (parts.length === 1 && parts[0] === child) return child;
      return <span key={i}>{parts}</span>;
    }
    return child;
  });
}

interface MessageBubbleProps {
  message: Message;
  animateIn?: boolean;
  animationIndex?: number;
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

export function MessageBubble({ message, animateIn = false, animationIndex = 0 }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;
  const isProactive = !!message.proactive;
  const [copied, setCopied] = useState(false);
  const [cursorExiting, setCursorExiting] = useState(false);
  const prevStreamingRef = useRef(isStreaming);
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Long-press context menu state
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);
  const longPressTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const touchStartPosRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Cursor fade-out when streaming ends
  useEffect(() => {
    if (prevStreamingRef.current && !isStreaming) {
      setCursorExiting(true);
      const timer = setTimeout(() => setCursorExiting(false), 200);
      return () => clearTimeout(timer);
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
    };
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current);
      setCopied(true);
      copyTimeoutRef.current = setTimeout(() => setCopied(false), 1500);
    } catch (e) {
      console.warn('[Copy] Failed:', e);
    }
  };

  // Clean up long-press timer on unmount
  useEffect(() => {
    return () => { if (longPressTimerRef.current) clearTimeout(longPressTimerRef.current); };
  }, []);

  // Close context menu on scroll or outside click
  useEffect(() => {
    if (!contextMenu) return;
    const close = () => setContextMenu(null);
    window.addEventListener('scroll', close, true);
    return () => window.removeEventListener('scroll', close, true);
  }, [contextMenu]);

  const handleTouchStart = useCallback((e: React.TouchEvent) => {
    const touch = e.touches[0];
    touchStartPosRef.current = { x: touch.clientX, y: touch.clientY };
    longPressTimerRef.current = setTimeout(() => {
      haptic(30);
      setContextMenu({ x: touch.clientX, y: touch.clientY });
    }, 500);
  }, []);

  const handleTouchMove = useCallback((e: React.TouchEvent) => {
    if (!longPressTimerRef.current) return;
    const touch = e.touches[0];
    const dx = touch.clientX - touchStartPosRef.current.x;
    const dy = touch.clientY - touchStartPosRef.current.y;
    if (Math.sqrt(dx * dx + dy * dy) > 10) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  const handleTouchEnd = useCallback(() => {
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }
  }, []);

  const handleCtxCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
    } catch (e) {
      console.warn('[CtxMenu Copy] Failed:', e);
    }
    setContextMenu(null);
  }, [message.content]);

  const handleCtxRegenerate = useCallback(() => {
    console.log('[CtxMenu] Regenerate requested for message:', message.id);
    setContextMenu(null);
  }, [message.id]);

  const handleCtxShare = useCallback(async () => {
    if (navigator.share) {
      try {
        await navigator.share({ text: message.content });
      } catch (e) {
        // User cancelled or share failed — not an error
        if ((e as DOMException).name !== 'AbortError') {
          console.warn('[CtxMenu Share] Failed:', e);
        }
      }
    } else {
      // Fallback: copy to clipboard
      try {
        await navigator.clipboard.writeText(message.content);
      } catch (e) {
        console.warn('[CtxMenu Share fallback] Failed:', e);
      }
    }
    setContextMenu(null);
  }, [message.content]);

  const hasCitations = !!(message.citations && message.citations.length > 0);

  const staggerClass = animationIndex <= 4 ? `msg-stagger-${animationIndex}` : 'msg-stagger-4';
  const animClass = animateIn ? `msg-animate-in ${staggerClass}` : '';

  if (isUser) {
    return (
      <div className={`py-4 px-4 md:px-8 ${animClass}`}>
        <div className="max-w-3xl mx-auto flex justify-end">
          <div style={{
            background: '#fff',
            color: '#000',
            padding: '12px 22px',
            borderRadius: '24px 24px 4px 24px',
            fontSize: '1rem',
            fontWeight: 500,
            maxWidth: '85%',
            lineHeight: 1.6,
            boxShadow: '0 4px 20px rgba(0,0,0,0.25)',
          }}>
            {message.attachments && message.attachments.length > 0 && (
              <AttachmentList attachments={message.attachments} compact />
            )}
            <p className="whitespace-pre-wrap">{message.content}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`py-5 px-4 md:px-8 ${isProactive ? 'bg-gradient-to-r from-purple-900/10 to-transparent' : ''} ${animClass}`}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Long-press context menu */}
      {contextMenu && (
        <>
          <div
            className="fixed inset-0 z-[999]"
            onClick={() => setContextMenu(null)}
            onTouchEnd={() => setContextMenu(null)}
          />
          <div
            className="ctx-menu fixed z-[1000]"
            style={{
              left: Math.min(contextMenu.x, window.innerWidth - 180),
              top: Math.min(contextMenu.y, window.innerHeight - 140),
            }}
          >
            <button className="ctx-menu-item w-full" onClick={handleCtxCopy}>
              <ClipboardIcon className="w-4 h-4" />
              Copy
            </button>
            <button className="ctx-menu-item w-full" onClick={handleCtxRegenerate}>
              <ArrowPathIcon className="w-4 h-4" />
              Regenerate
            </button>
            <button className="ctx-menu-item w-full" onClick={handleCtxShare}>
              <ShareIcon className="w-4 h-4" />
              Share
            </button>
          </div>
        </>
      )}
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* AI Avatar */}
        <div className="flex-shrink-0 relative mt-1">
          <div
            className="w-9 h-9 flex items-center justify-center"
            style={{
              borderRadius: 8,
              background: isProactive
                ? 'linear-gradient(135deg, rgba(168,85,247,0.2) 0%, rgba(255,255,255,0.02) 100%)'
                : 'linear-gradient(135deg, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0.02) 100%)',
              border: '1px solid rgba(255,255,255,0.07)',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
            }}
          >
            {isProactive
              ? <BoltIcon className="w-4 h-4 text-purple-300" />
              : <SparklesIcon className="w-4 h-4 text-white" />
            }
          </div>
          {isProactive && (
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-pink-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-pink-500" />
            </span>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0 group">
          {/* Role / badge */}
          <div className="flex items-center gap-2 mb-2">
            <span className="text-chat-text font-medium text-sm">AURA</span>
            {isProactive && message.proactive && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-purple-500/20 text-purple-300 border border-purple-500/30">
                <span>{PROACTIVE_ICONS[message.proactive.action] || '💭'}</span>
                <span>{message.proactive.trigger || 'initiated'}</span>
              </span>
            )}
          </div>

          {/* Tool trace */}
          {message.toolTrace && message.toolTrace.length > 0 && (
            <ToolTrace traces={message.toolTrace} isStreaming={isStreaming} />
          )}

          {/* Message content with inline citation badges */}
          <div className={`prose prose-invert max-w-none text-chat-text ${isStreaming ? 'stream-container-active' : ''}`}>
            {hasCitations && !isStreaming ? (
              <CitationAwareMarkdown content={message.content} citations={message.citations!} messageId={message.id} />
            ) : (
              <ReactMarkdown
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    if (!match) {
                      return (
                        <code className="bg-gray-800 px-1.5 py-0.5 rounded text-sm" {...props}>
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
                  a({ href, children }) {
                    return (
                      <a href={href} target="_blank" rel="noopener noreferrer"
                        className="text-purple-400 hover:text-purple-300 underline">
                        {children}
                      </a>
                    );
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            )}
            {(isStreaming || cursorExiting) && (
              <span className={`stream-cursor ${cursorExiting ? 'cursor-exiting' : ''}`} />
            )}
          </div>

          {/* Citations */}
          {hasCitations && !isStreaming && (
            <CitationList citations={message.citations!} messageId={message.id} />
          )}

          {/* Memory transparency */}
          {message.memoriesUsed && message.memoriesUsed.length > 0 && !isStreaming && (
            <MemoryIndicator memories={message.memoriesUsed} />
          )}

          {/* Model compare */}
          {message.compareResults && message.compareResults.length > 0 && (
            <ModelCompare
              results={message.compareResults}
              query={message.content}
              onUseResponse={(response) => navigator.clipboard.writeText(response).catch(e => console.warn('[Copy] Failed:', e))}
            />
          )}

          {/* Footer: model + timestamp + copy */}
          <div className="mt-2 flex items-center gap-3 text-xs text-chat-text-secondary">
            {message.model_used && !isStreaming && (
              <span className="opacity-50">{message.model_used}</span>
            )}
            <span>{new Date(message.timestamp).toLocaleTimeString()}</span>
            {!isStreaming && (
              <button
                onClick={handleCopy}
                aria-label="Copy message"
                className="opacity-0 group-hover:opacity-100 transition-opacity p-0.5 rounded hover:text-chat-text"
              >
                {copied
                  ? <CheckIcon className="w-3.5 h-3.5 text-green-400" />
                  : <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                }
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
