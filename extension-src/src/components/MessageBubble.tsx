import React, { useState, useEffect, useRef } from 'react';
import { Copy, Check, Volume2, VolumeX, ChevronDown, Download, Bot, ThumbsUp, ThumbsDown, RotateCw } from 'lucide-react';
import type { Message } from '../types';
import { md } from '../markdown';
import { speak, stopSpeaking, isSpeaking } from '../tts';
import { useStore } from '../store';
import { messageAsMarkdown, messageAsText, downloadFile } from '../exportChat';
import AgentStepRow from './AgentStepRow';
import { routing } from '../api/client';

interface Props {
  message: Message;
  isLatest?: boolean;
  isStreaming?: boolean;
}

export default function MessageBubble({ message, isLatest, isStreaming: isStreamingProp }: Props) {
  const isUser = message.role === 'user';
  const { activeStream, featureModels } = useStore();
  const [speaking, setSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [thinkExpanded, setThinkExpanded] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const exportRef = useRef<HTMLDivElement>(null);
  const thinkContentRef = useRef<HTMLDivElement>(null);

  const sendFeedback = (signal: 'up' | 'down' | 'regen') => {
    const model = featureModels['chat'] || 'unknown';
    const conversationId = useStore.getState().activeConversationId || undefined;
    routing
      .feedback({
        signal: signal === 'up' ? 'positive' : signal === 'down' ? 'negative' : 'regenerate',
        model,
        conversation_id: conversationId,
      })
      .catch(() => {});
    if (signal !== 'regen') setFeedback(signal);
  };

  const isStreaming = isStreamingProp || (isLatest && !!activeStream);
  const hasThinking = !!message.thinkingContent;
  const hasAgent = !!message.agentSteps;
  const agentActive = hasAgent && !message.agentDone;
  const [agentExpanded, setAgentExpanded] = useState(true);

  useEffect(() => {
    if (hasAgent && message.agentDone) setAgentExpanded(false);
  }, [hasAgent, message.agentDone]);

  // Get active model name for the badge
  const modelDisplay = featureModels['chat']
    ? featureModels['chat'].replace(/:cloud$/, '').split('/').pop() || 'AI'
    : null;

  // Poll speaking state so the button updates when speech ends naturally
  useEffect(() => {
    if (!speaking) return;
    const interval = setInterval(() => {
      if (!isSpeaking()) setSpeaking(false);
    }, 300);
    return () => clearInterval(interval);
  }, [speaking]);

  // Close export dropdown on outside click
  useEffect(() => {
    if (!exportOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, [exportOpen]);

  const handleExportCopyMd = async () => {
    try {
      await navigator.clipboard.writeText(messageAsMarkdown(message));
    } catch { /* noop */ }
    setExportOpen(false);
  };

  const handleExportCopyText = async () => {
    try {
      await navigator.clipboard.writeText(messageAsText(message));
    } catch { /* noop */ }
    setExportOpen(false);
  };

  const handleExportSaveMd = () => {
    const role = message.role === 'user' ? 'user' : 'aura';
    const ts = new Date(message.timestamp).toISOString().slice(0, 10);
    downloadFile(messageAsMarkdown(message), `${role}-${ts}.md`, 'text/markdown;charset=utf-8');
    setExportOpen(false);
  };

  const handleTts = () => {
    if (speaking) {
      stopSpeaking();
      setSpeaking(false);
    } else {
      setSpeaking(true);
      speak(message.text, {
        onEnd: () => setSpeaking(false),
      });
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard not available
    }
  };

  /* ───── User message: right-aligned glass bubble ───── */
  if (isUser) {
    return (
      <div
        className="flex mb-3 justify-end msg-slide-in"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <div className="user-bubble" style={{
          background: 'linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(245,243,255,0.92) 100%)',
          color: '#1a1a2e',
          padding: '10px 16px',
          borderRadius: '18px 18px 4px 18px',
          fontSize: '12.5px',
          fontWeight: 500,
          maxWidth: '82%',
          lineHeight: 1.55,
          boxShadow: '0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.6)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          position: 'relative',
        }}>
          {message.text}

          {/* Hover timestamp */}
          <div
            className="user-msg-time"
            style={{
              position: 'absolute',
              bottom: -18,
              right: 4,
              fontSize: '9px',
              color: 'var(--mu)',
              opacity: hovered ? 1 : 0,
              transition: 'opacity 0.2s ease',
              pointerEvents: 'none',
              whiteSpace: 'nowrap',
            }}
          >
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      </div>
    );
  }

  /* ───── AI message: full-width with left accent border ───── */
  return (
    <div
      className="flex gap-2.5 mb-4 msg-slide-in"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* Avatar */}
      <div
        className="flex-shrink-0 flex items-center justify-center"
        style={{
          width: 28, height: 28, borderRadius: 7, marginTop: 2,
          background: 'linear-gradient(135deg, rgba(124,58,237,0.2) 0%, rgba(124,58,237,0.05) 100%)',
          border: '1px solid rgba(124,58,237,0.15)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          fontSize: '11px', color: 'var(--pl)', fontWeight: 600,
        }}
      >
        A
      </div>

      <div
        className="flex-1 min-w-0"
        style={{
          borderLeft: '2px solid transparent',
          borderImage: 'linear-gradient(to bottom, var(--pl), var(--p3)) 1',
          paddingLeft: 10,
        }}
      >
        {/* Model badge */}
        {modelDisplay && (
          <span
            style={{
              display: 'inline-block',
              fontSize: '9px',
              color: 'var(--mu)',
              background: 'var(--s2)',
              border: '1px solid var(--b3)',
              borderRadius: 4,
              padding: '1px 5px',
              marginBottom: 4,
              letterSpacing: '0.02em',
              fontWeight: 500,
            }}
          >
            {modelDisplay}
          </span>
        )}

        {/* Collapsible thinking section */}
        {hasThinking && (
          <div className="think-block">
            <button
              className="think-block-toggle"
              onClick={() => setThinkExpanded(!thinkExpanded)}
            >
              <span className="think-block-icon">{'\uD83D\uDCAD'}</span>
              <span>{thinkExpanded ? 'Thinking' : 'Thinking... (click to expand)'}</span>
              <ChevronDown
                size={12}
                style={{
                  marginLeft: 'auto',
                  transition: 'transform 0.2s ease',
                  transform: thinkExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                }}
              />
            </button>
            <div
              ref={thinkContentRef}
              className="think-block-content"
              style={{
                maxHeight: thinkExpanded ? (thinkContentRef.current?.scrollHeight || 2000) + 'px' : '0px',
              }}
            >
              <div
                className="think-block-text md-body"
                dangerouslySetInnerHTML={{ __html: md(message.thinkingContent!) }}
              />
            </div>
          </div>
        )}

        {/* Agent run — task header + step list */}
        {hasAgent && (
          <div style={{ marginBottom: message.text ? 8 : 0 }}>
            <button
              onClick={() => setAgentExpanded(v => !v)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                background: 'transparent',
                border: 'none',
                padding: '2px 0 6px',
                cursor: 'pointer',
                color: 'var(--pl)',
                fontSize: 12,
                fontWeight: 500,
                fontFamily: 'inherit',
              }}
            >
              <Bot size={13} />
              <span>Agent: {message.agentTask || 'running'}</span>
              {agentActive && (
                <span style={{ color: 'var(--mu)', fontSize: 10.5, marginLeft: 4 }}>
                  · step {message.agentSteps!.length}
                </span>
              )}
              <ChevronDown
                size={11}
                style={{ marginLeft: 'auto', transition: 'transform 0.2s', transform: agentExpanded ? 'rotate(180deg)' : 'rotate(0deg)' }}
              />
            </button>
            {agentExpanded && message.agentSteps!.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {message.agentSteps!.map((s, i) => <AgentStepRow key={i} step={s} />)}
              </div>
            )}
            {agentActive && (
              <div className="aura-thinking" style={{ marginTop: 6 }}>
                <span /><span /><span />
              </div>
            )}
          </div>
        )}

        {/* Message body */}
        {message.text && (
          <div
            className="md-body"
            style={{ fontSize: '12.5px', lineHeight: 1.65, color: 'var(--tx)' }}
            dangerouslySetInnerHTML={{ __html: md(message.text) }}
          />
        )}

        {/* Streaming cursor — blinking purple block */}
        {isStreaming && !hasAgent && message.text && (
          <span className="streaming-cursor" />
        )}

        {/* Branded thinking indicator — three staggered gradient circles */}
        {isStreaming && !hasAgent && !message.text && (
          <div className="aura-thinking">
            <span /><span /><span />
          </div>
        )}

        {/* Footer: timestamp + actions — fade in on hover */}
        <div
          className="flex items-center gap-2 mt-1"
          style={{
            opacity: hovered ? 1 : 0,
            transition: 'opacity 0.25s ease',
            height: hovered ? 'auto' : 0,
            overflow: 'hidden',
          }}
        >
          <span style={{ fontSize: '10px', color: 'var(--di)' }}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="msg-action-btn"
            title={copied ? 'Copied!' : 'Copy text'}
            style={{
              background: 'none',
              border: 'none',
              color: copied ? 'var(--gr)' : 'var(--di)',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              gap: 3,
              fontSize: '10px',
              borderRadius: 4,
              transition: 'all 0.15s ease',
            }}
          >
            {copied ? <Check size={11} /> : <Copy size={11} />}
            {copied && <span style={{ color: 'var(--gr)' }}>Copied!</span>}
          </button>

          {/* Export dropdown */}
          <div ref={exportRef} style={{ position: 'relative', display: 'inline-flex' }}>
            <button
              onClick={() => setExportOpen(!exportOpen)}
              className="msg-action-btn"
              title="Export message"
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--di)',
                cursor: 'pointer',
                padding: 2,
                display: 'flex',
                alignItems: 'center',
                borderRadius: 4,
                transition: 'all 0.15s ease',
              }}
            >
              <Download size={11} />
            </button>
            {exportOpen && (
              <div
                style={{
                  position: 'absolute',
                  bottom: '100%',
                  left: 0,
                  marginBottom: 4,
                  background: 'var(--s2)',
                  border: '1px solid var(--b3)',
                  borderRadius: 6,
                  boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                  zIndex: 50,
                  minWidth: 140,
                  overflow: 'hidden',
                }}
              >
                {[
                  { label: 'Copy as Markdown', fn: handleExportCopyMd },
                  { label: 'Copy as Text', fn: handleExportCopyText },
                  { label: 'Save as .md file', fn: handleExportSaveMd },
                ].map((item) => (
                  <button
                    key={item.label}
                    onClick={item.fn}
                    style={{
                      display: 'block',
                      width: '100%',
                      padding: '6px 10px',
                      background: 'none',
                      border: 'none',
                      color: 'var(--tx)',
                      fontSize: '11px',
                      textAlign: 'left',
                      cursor: 'pointer',
                      fontFamily: 'inherit',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--b1)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'none')}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Speak button */}
          <button
            onClick={handleTts}
            className="msg-action-btn"
            title={speaking ? 'Stop speaking' : 'Read aloud'}
            style={{
              background: 'none',
              border: 'none',
              color: speaking ? 'var(--rd)' : 'var(--di)',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              borderRadius: 4,
              transition: 'all 0.15s ease',
            }}
          >
            {speaking ? <VolumeX size={11} /> : <Volume2 size={11} />}
          </button>

          {/* Routing feedback — thumbs up / down / regenerate (AI messages only) */}
          <button
            onClick={() => sendFeedback('up')}
            className="msg-action-btn"
            title="Good response"
            style={{
              background: 'none',
              border: 'none',
              color: feedback === 'up' ? 'var(--gr)' : 'var(--di)',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              borderRadius: 4,
            }}
          >
            <ThumbsUp size={11} />
          </button>
          <button
            onClick={() => sendFeedback('down')}
            className="msg-action-btn"
            title="Bad response"
            style={{
              background: 'none',
              border: 'none',
              color: feedback === 'down' ? 'var(--rd)' : 'var(--di)',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              borderRadius: 4,
            }}
          >
            <ThumbsDown size={11} />
          </button>
          <button
            onClick={() => sendFeedback('regen')}
            className="msg-action-btn"
            title="Regenerate"
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--di)',
              cursor: 'pointer',
              padding: 2,
              display: 'flex',
              alignItems: 'center',
              borderRadius: 4,
            }}
          >
            <RotateCw size={11} />
          </button>
        </div>
      </div>
    </div>
  );
}
