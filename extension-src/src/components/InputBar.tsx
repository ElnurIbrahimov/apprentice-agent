import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Send, Brain, Globe, FileText } from 'lucide-react';
import { useStore } from '../store';
import { getPageContentCached } from '../ext';
import ModelPill from './ModelPill';

interface Props {
  onSend: (text: string) => void;
  featureKey?: string;
  placeholder?: string;
  disabled?: boolean;
}

const PLACEHOLDER_SUGGESTIONS = [
  'Ask anything...',
  'Summarize this page...',
  'Translate to...',
  'Explain this code...',
  'Find key points...',
];

export default function InputBar({ onSend, featureKey = 'chat', placeholder, disabled }: Props) {
  const { thinkingMode, setThinkingMode, deepResearch, setDeepResearch, activeStream, setPendingCtx, pendingCtx } = useStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [hasText, setHasText] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [placeholderFade, setPlaceholderFade] = useState(true);

  const isStreaming = !!activeStream;

  // Cycle placeholder suggestions
  useEffect(() => {
    if (placeholder) return; // skip cycling if explicit placeholder given
    const interval = setInterval(() => {
      setPlaceholderFade(false);
      setTimeout(() => {
        setPlaceholderIdx(prev => (prev + 1) % PLACEHOLDER_SUGGESTIONS.length);
        setPlaceholderFade(true);
      }, 200);
    }, 3500);
    return () => clearInterval(interval);
  }, [placeholder]);

  const currentPlaceholder = placeholder || PLACEHOLDER_SUGGESTIONS[placeholderIdx];

  const addPageCtx = async () => {
    setPageLoading(true);
    try {
      const resp = await getPageContentCached();
      if (resp?.ok && resp.text) {
        setPendingCtx({ text: resp.text.slice(0, 20000), title: resp.title, url: resp.url, action: 'ask' });
      }
    } finally {
      setPageLoading(false);
    }
  };

  const handleSend = () => {
    const text = textareaRef.current?.value.trim();
    if (!text || isStreaming || disabled) return;
    onSend(text);
    if (textareaRef.current) {
      textareaRef.current.value = '';
      setHasText(false);
      setCharCount(0);
      autoResize();
    }
  };

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  };

  const handleInput = useCallback(() => {
    autoResize();
    const val = textareaRef.current?.value || '';
    setHasText(val.trim().length > 0);
    setCharCount(val.length);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = hasText && !isStreaming && !disabled;

  // Derive favicon URL from pending context
  const faviconUrl = pendingCtx?.url
    ? (() => {
        try {
          const u = new URL(pendingCtx.url);
          return `${u.origin}/favicon.ico`;
        } catch {
          return null;
        }
      })()
    : null;

  return (
    <div
      className="flex flex-col flex-shrink-0"
      style={{ padding: '6px 8px 10px', background: 'transparent' }}
    >
      {/* Context bar - prominent with favicon */}
      {pendingCtx && (
        <div
          className="input-ctx-bar flex items-center gap-2 mb-2 px-3 py-2"
          style={{
            background: 'rgba(124, 58, 237, 0.1)',
            border: '1px solid rgba(124, 58, 237, 0.2)',
            borderRadius: 12,
            fontSize: '11px',
          }}
        >
          {faviconUrl && (
            <img
              src={faviconUrl}
              alt=""
              width={14}
              height={14}
              style={{ borderRadius: 2, flexShrink: 0 }}
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          )}
          <FileText size={12} style={{ color: 'var(--pl)', flexShrink: 0 }} />
          <span className="truncate flex-1" style={{ color: 'var(--pl)' }}>
            {pendingCtx.title || pendingCtx.url || 'Page context'}
          </span>
          <button
            onClick={() => setPendingCtx(null)}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 0, fontSize: '14px', lineHeight: 1 }}
          >
            &times;
          </button>
        </div>
      )}

      {/* Mode pills row */}
      <div className="flex items-center gap-1.5 mb-2">
        <button
          onClick={() => setThinkingMode(!thinkingMode)}
          className="flex items-center gap-1 px-2 py-1 mode-pill"
          style={{
            background: thinkingMode ? 'var(--pg2)' : 'var(--s2)',
            border: `1px solid ${thinkingMode ? 'var(--p)' : 'var(--b1)'}`,
            borderRadius: 'var(--r-pill)',
            color: thinkingMode ? 'var(--pl)' : 'var(--mu)',
            fontSize: '11px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          <Brain size={11} />
          <span>Think</span>
        </button>
        <button
          onClick={() => setDeepResearch(!deepResearch)}
          className="flex items-center gap-1 px-2 py-1 mode-pill"
          style={{
            background: deepResearch ? 'var(--pg2)' : 'var(--s2)',
            border: `1px solid ${deepResearch ? 'var(--p)' : 'var(--b1)'}`,
            borderRadius: 'var(--r-pill)',
            color: deepResearch ? 'var(--pl)' : 'var(--mu)',
            fontSize: '11px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          <Globe size={11} />
          <span>Research</span>
        </button>
        <button
          onClick={addPageCtx}
          disabled={pageLoading}
          className="flex items-center gap-1 px-2 py-1 mode-pill"
          style={{
            background: pendingCtx ? 'var(--pg2)' : 'var(--s2)',
            border: `1px solid ${pendingCtx ? 'var(--p)' : 'var(--b1)'}`,
            borderRadius: 'var(--r-pill)',
            color: pendingCtx ? 'var(--pl)' : 'var(--mu)',
            fontSize: '11px',
            cursor: pageLoading ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
            opacity: pageLoading ? 0.6 : 1,
          }}
          title="Add page content as context"
        >
          <FileText size={11} />
          <span>{pageLoading ? '...' : 'Page'}</span>
        </button>
        <div className="flex-1" />
        <ModelPill featureKey={featureKey} />
      </div>

      {/* Glass input wrapper */}
      <div
        className={`input-glass-wrap ${isFocused ? 'input-focused' : ''}`}
        style={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          background: 'rgba(20, 20, 24, 0.55)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderRadius: 18,
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
        }}
      >
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={currentPlaceholder}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled || isStreaming}
          className={placeholderFade ? 'placeholder-visible' : 'placeholder-hidden'}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--tx)',
            fontSize: '12.5px',
            lineHeight: 1.55,
            padding: '10px 42px 8px 14px',
            resize: 'none',
            minHeight: 40,
            maxHeight: 140,
            outline: 'none',
            fontFamily: 'inherit',
            width: '100%',
          }}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
        />

        {/* Character count */}
        {charCount > 500 && (
          <span
            className="char-count"
            style={{
              position: 'absolute',
              right: 44,
              bottom: 12,
              fontSize: '9px',
              color: charCount > 4000 ? 'var(--rd)' : 'var(--mu)',
              opacity: 0.7,
              fontVariantNumeric: 'tabular-nums',
            }}
          >
            {charCount.toLocaleString()}
          </span>
        )}

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={!canSend}
          className={`send-btn ${canSend ? 'send-btn-ready' : ''}`}
          style={{
            position: 'absolute', right: 8, bottom: 8,
            width: 30, height: 30, borderRadius: 9,
            background: canSend ? '#fff' : 'rgba(255,255,255,0.08)',
            border: 'none',
            color: canSend ? '#000' : '#555',
            cursor: canSend ? 'pointer' : 'not-allowed',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
          }}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}
