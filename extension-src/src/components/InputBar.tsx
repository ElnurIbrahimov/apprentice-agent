import React, { useRef, useState } from 'react';
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

export default function InputBar({ onSend, featureKey = 'chat', placeholder = 'Message AURA…', disabled }: Props) {
  const { thinkingMode, setThinkingMode, deepResearch, setDeepResearch, activeStream, setPendingCtx } = useStore();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [pageLoading, setPageLoading] = useState(false);

  const isStreaming = !!activeStream;

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
      autoResize();
    }
  };

  const autoResize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div
      className="flex flex-col flex-shrink-0"
      style={{ padding: '6px 8px 10px', background: 'transparent' }}
    >
      {/* Mode pills row */}
      <div className="flex items-center gap-1.5 mb-2">
        <button
          onClick={() => setThinkingMode(!thinkingMode)}
          className="flex items-center gap-1 px-2 py-1 transition-all duration-150"
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
          className="flex items-center gap-1 px-2 py-1 transition-all duration-150"
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
          className="flex items-center gap-1 px-2 py-1 transition-all duration-150"
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-pill)',
            color: 'var(--mu)',
            fontSize: '11px',
            cursor: pageLoading ? 'not-allowed' : 'pointer',
            fontFamily: 'inherit',
            opacity: pageLoading ? 0.6 : 1,
          }}
          title="Add page content as context"
        >
          <FileText size={11} />
          <span>{pageLoading ? '…' : 'Page'}</span>
        </button>
        <div className="flex-1" />
        <ModelPill featureKey={featureKey} />
      </div>

      {/* Glass input wrapper */}
      <div style={{
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        background: 'rgba(20, 20, 24, 0.55)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: 18,
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)',
        transition: 'border-color 0.2s, box-shadow 0.2s',
      }}>
        <textarea
          ref={textareaRef}
          rows={1}
          placeholder={placeholder}
          onInput={autoResize}
          onKeyDown={handleKeyDown}
          disabled={disabled || isStreaming}
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
          onFocus={e => {
            const w = e.target.closest('div') as HTMLElement;
            if (w) { w.style.borderColor = 'rgba(255,255,255,0.18)'; w.style.boxShadow = '0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1), inset 0 1px 0 rgba(255,255,255,0.04)'; }
          }}
          onBlur={e => {
            const w = e.target.closest('div') as HTMLElement;
            if (w) { w.style.borderColor = 'rgba(255,255,255,0.07)'; w.style.boxShadow = '0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)'; }
          }}
        />
        <button
          onClick={handleSend}
          disabled={isStreaming || disabled}
          style={{
            position: 'absolute', right: 8, bottom: 8,
            width: 30, height: 30, borderRadius: 9,
            background: (isStreaming || disabled) ? 'rgba(255,255,255,0.08)' : '#fff',
            border: 'none',
            color: (isStreaming || disabled) ? '#555' : '#000',
            cursor: (isStreaming || disabled) ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            boxShadow: (isStreaming || disabled) ? 'none' : '0 2px 8px rgba(255,255,255,0.15)',
            transition: 'all 0.2s',
          }}
        >
          <Send size={14} />
        </button>
      </div>
    </div>
  );
}
