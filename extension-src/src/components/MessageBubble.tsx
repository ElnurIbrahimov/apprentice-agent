import React, { useState, useEffect } from 'react';
import { Copy, Check, Volume2, VolumeX } from 'lucide-react';
import type { Message } from '../types';
import { md } from '../markdown';
import { speak, stopSpeaking, isSpeaking } from '../tts';
import { useStore } from '../store';

interface Props {
  message: Message;
  isLatest?: boolean;
}

export default function MessageBubble({ message, isLatest }: Props) {
  const isUser = message.role === 'user';
  const { activeStream, featureModels } = useStore();
  const [speaking, setSpeaking] = useState(false);
  const [copied, setCopied] = useState(false);
  const [hovered, setHovered] = useState(false);

  const isStreaming = isLatest && !!activeStream;

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

  if (isUser) {
    return (
      <div className="flex mb-3 justify-end msg-slide-in">
        <div style={{
          background: '#fff',
          color: '#000',
          padding: '9px 16px',
          borderRadius: '18px 18px 3px 18px',
          fontSize: '12.5px',
          fontWeight: 500,
          maxWidth: '82%',
          lineHeight: 1.55,
          boxShadow: '0 4px 16px rgba(0,0,0,0.3)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {message.text}
        </div>
      </div>
    );
  }

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

      <div className="flex-1 min-w-0">
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

        {/* Message body */}
        <div
          className="md-body"
          style={{ fontSize: '12.5px', lineHeight: 1.65, color: 'var(--tx)' }}
          dangerouslySetInnerHTML={{ __html: md(message.text) }}
        />

        {/* Streaming typing indicator */}
        {isStreaming && (
          <div className="dots" style={{ marginTop: 2 }}>
            <span />
            <span />
            <span />
          </div>
        )}

        {/* Footer: timestamp + actions */}
        <div
          className="flex items-center gap-2 mt-1"
          style={{
            opacity: hovered ? 1 : 0.5,
            transition: 'opacity 0.2s ease',
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
        </div>
      </div>
    </div>
  );
}
