import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Brain, Globe, FileText, X, Paperclip } from 'lucide-react';
import { useStore } from '../store';
import { getPageContentCached } from '../ext';
import ModelPill from './ModelPill';
import type { ThinkingLevel } from '../types';

const THINKING_LABELS: Record<ThinkingLevel, string> = {
  low: 'Low',
  medium: 'Medium',
  high: 'High',
};

const THINKING_TOOLTIPS: Record<ThinkingLevel, string> = {
  low: 'Quick reasoning check',
  medium: 'Step-by-step analysis',
  high: 'Deep multi-perspective reasoning',
};

const THINKING_CYCLE: ThinkingLevel[] = ['low', 'medium', 'high'];

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

const MAX_VISIBLE_LINES = 6;
const LINE_HEIGHT_PX = 20.25; // 13.5px * 1.5
const MAX_TEXTAREA_HEIGHT = Math.ceil(MAX_VISIBLE_LINES * LINE_HEIGHT_PX) + 20; // + padding

export default function InputBar({ onSend, featureKey = 'chat', placeholder, disabled }: Props) {
  const { thinkingMode, setThinkingMode, thinkingLevel, setThinkingLevel, deepResearch, setDeepResearch, activeStream, setPendingCtx, pendingCtx } = useStore();
  const [showThinkTooltip, setShowThinkTooltip] = useState(false);
  const thinkLongPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [isFocused, setIsFocused] = useState(false);
  const [hasText, setHasText] = useState(false);
  const [charCount, setCharCount] = useState(0);
  const [placeholderIdx, setPlaceholderIdx] = useState(0);
  const [placeholderFade, setPlaceholderFade] = useState(true);
  const [attachments, setAttachments] = useState<{ name: string; id: string }[]>([]);

  const isStreaming = !!activeStream;

  // Cycle placeholder suggestions with crossfade
  useEffect(() => {
    if (placeholder) return;
    const interval = setInterval(() => {
      setPlaceholderFade(false);
      setTimeout(() => {
        setPlaceholderIdx(prev => (prev + 1) % PLACEHOLDER_SUGGESTIONS.length);
        setPlaceholderFade(true);
      }, 250);
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

  const autoResize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const next = Math.min(el.scrollHeight, MAX_TEXTAREA_HEIGHT);
    el.style.height = next + 'px';
    el.style.overflowY = el.scrollHeight > MAX_TEXTAREA_HEIGHT ? 'auto' : 'hidden';
  }, []);

  const handleInput = useCallback(() => {
    autoResize();
    const val = textareaRef.current?.value || '';
    setHasText(val.trim().length > 0);
    setCharCount(val.length);
  }, [autoResize]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(prev => prev.filter(a => a.id !== id));
  };

  const canSend = hasText && !isStreaming && !disabled;

  // Derive favicon URL from pending context
  const faviconUrl = pendingCtx?.url
    ? (() => {
        try {
          const u = new URL(pendingCtx.url);
          return `https://www.google.com/s2/favicons?domain=${u.hostname}&sz=32`;
        } catch {
          return null;
        }
      })()
    : null;

  return (
    <div className="input-bar-root">
      {/* Attachments area */}
      {attachments.length > 0 && (
        <div className="input-attachments">
          {attachments.map(att => (
            <div key={att.id} className="input-attachment-pill">
              <Paperclip size={10} />
              <span className="input-attachment-name">{att.name}</span>
              <button
                className="input-attachment-remove"
                onClick={() => removeAttachment(att.id)}
                aria-label="Remove attachment"
              >
                <X size={10} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Context bar */}
      {pendingCtx && (
        <div className="input-ctx-bar">
          {faviconUrl && (
            <img
              src={faviconUrl}
              alt=""
              width={14}
              height={14}
              className="input-ctx-favicon"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          )}
          <FileText size={12} className="input-ctx-icon" />
          <span className="input-ctx-title">
            {pendingCtx.title || pendingCtx.url || 'Page context'}
          </span>
          <button
            onClick={() => setPendingCtx(null)}
            className="input-ctx-remove"
            aria-label="Remove context"
          >
            <X size={12} />
          </button>
        </div>
      )}

      {/* Glass input wrapper */}
      <div className={`input-glass-wrap ${isFocused ? 'input-focused' : ''} ${hasText ? 'has-text' : ''}`}>
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={currentPlaceholder}
          onInput={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled || isStreaming}
          className={`input-textarea ${placeholderFade ? 'placeholder-visible' : 'placeholder-hidden'}`}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
        />

        {/* Action row */}
        <div className="input-action-row">
          {/* Left side: pills */}
          <div className="input-action-left">
            <button
              onClick={() => {
                if (!thinkingMode) {
                  setThinkingMode(true);
                } else {
                  setThinkingMode(false);
                }
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                if (thinkingMode) {
                  // Cycle level on right-click
                  const idx = THINKING_CYCLE.indexOf(thinkingLevel);
                  setThinkingLevel(THINKING_CYCLE[(idx + 1) % THINKING_CYCLE.length]);
                } else {
                  setThinkingMode(true);
                }
              }}
              onMouseDown={() => {
                // Long-press to cycle level
                thinkLongPressTimer.current = setTimeout(() => {
                  thinkLongPressTimer.current = null;
                  if (thinkingMode) {
                    const idx = THINKING_CYCLE.indexOf(thinkingLevel);
                    setThinkingLevel(THINKING_CYCLE[(idx + 1) % THINKING_CYCLE.length]);
                  } else {
                    setThinkingMode(true);
                  }
                }, 500);
              }}
              onMouseUp={() => {
                if (thinkLongPressTimer.current) {
                  clearTimeout(thinkLongPressTimer.current);
                  thinkLongPressTimer.current = null;
                }
              }}
              onMouseLeave={() => {
                if (thinkLongPressTimer.current) {
                  clearTimeout(thinkLongPressTimer.current);
                  thinkLongPressTimer.current = null;
                }
                setShowThinkTooltip(false);
              }}
              onMouseEnter={() => setShowThinkTooltip(true)}
              className={`input-pill ${thinkingMode ? 'input-pill-think-active' : ''}`}
              title=""
            >
              <Brain size={11} />
              <span>{thinkingMode ? `Think: ${THINKING_LABELS[thinkingLevel]}` : 'Think'}</span>
              {showThinkTooltip && (
                <div className="think-tooltip">
                  {thinkingMode
                    ? `${THINKING_TOOLTIPS[thinkingLevel]} — right-click to cycle level`
                    : 'Enable chain-of-thought reasoning'}
                </div>
              )}
            </button>
            <button
              onClick={() => setDeepResearch(!deepResearch)}
              className={`input-pill ${deepResearch ? 'input-pill-active' : ''}`}
              title="Deep research mode"
            >
              <Globe size={11} />
              <span>Research</span>
            </button>
            <button
              onClick={addPageCtx}
              disabled={pageLoading}
              className={`input-pill ${pendingCtx ? 'input-pill-active' : ''}`}
              title="Add page content as context"
              style={{ opacity: pageLoading ? 0.5 : 1 }}
            >
              <FileText size={11} />
              <span>{pageLoading ? '...' : 'Page'}</span>
            </button>
          </div>

          {/* Right side: char count, kbd hint, model, send */}
          <div className="input-action-right">
            {charCount > 500 && (
              <span className={`input-char-count ${charCount > 4000 ? 'input-char-warn' : ''}`}>
                {charCount.toLocaleString()}
              </span>
            )}
            <ModelPill featureKey={featureKey} />
            {!hasText && (
              <kbd className="input-kbd-hint">
                <span style={{ fontSize: '10px' }}>&#8984;</span>K
              </kbd>
            )}
            <button
              onClick={handleSend}
              disabled={!canSend}
              className={`input-send-btn ${canSend ? 'input-send-ready' : ''}`}
              aria-label="Send message"
            >
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  d="M7 11.5V2.5M7 2.5L3 6.5M7 2.5L11 6.5"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
