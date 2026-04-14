import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Plus, Sun, Moon, RefreshCw, Download, Clock, Server } from 'lucide-react';
import { useStore } from '../store';
import { fetchStatus } from '../ws';
import { exportChat } from '../exportChat';
import { getServerLabel } from '../api';
import ConversationHistory from './ConversationHistory';
import ext from '../ext';

const STATUS_DOT: Record<string, string> = {
  online: '#10b981',
  connecting: '#f59e0b',
  offline: '#ef4444',
};

// Map mood emoji to a glow color for the logo dot
const MOOD_GLOW: Record<string, string> = {
  '\uD83D\uDE0A': '#eab308', // happy - yellow
  '\uD83D\uDE04': '#eab308', // grinning - yellow
  '\uD83D\uDE03': '#f59e0b', // smiley - amber
  '\uD83D\uDE42': '#10b981', // slightly smiling - green
  '\uD83D\uDE14': '#3b82f6', // pensive - blue
  '\uD83D\uDE22': '#3b82f6', // crying - blue
  '\uD83D\uDE20': '#ef4444', // angry - red
  '\uD83E\uDD14': '#8b5cf6', // thinking - purple
  '\uD83D\uDE10': '#6b7280', // neutral - gray
  '\uD83D\uDE31': '#f97316', // scared - orange
  '\uD83D\uDE2E': '#a78bfa', // surprised - light purple
  '\uD83D\uDE0D': '#ec4899', // heart eyes - pink
  '\uD83D\uDE34': '#06b6d4', // sleeping - teal
};

function getMoodGlowColor(moodEmoji: string): string | null {
  if (!moodEmoji) return null;
  return MOOD_GLOW[moodEmoji] || null;
}

export default function Header() {
  const { wsReady, modelName, mood, theme, toggleTheme, backendStatus, messages, featureModels, newConversation, conversations, loadConversationList, historyLoaded, activeStream, setPanel } = useStore();
  const status = backendStatus === 'online' ? 'online' : backendStatus === 'offline' ? 'offline' : 'connecting';
  const isThinking = !!activeStream;
  const [scrolled, setScrolled] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [pageType, setPageType] = useState<string | null>(null);
  const headerRef = useRef<HTMLElement>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  // Load conversation list on mount
  useEffect(() => {
    if (!historyLoaded) {
      loadConversationList();
    }
  }, [historyLoaded, loadConversationList]);

  // Detect page type from messages + active tab URL
  useEffect(() => {
    const handler = (msg: any) => {
      if (msg.type === 'YT_TAB_DETECTED') setPageType('youtube');
      if (msg.type === 'PDF_TAB_DETECTED') setPageType('pdf');
    };
    ext?.runtime?.onMessage?.addListener(handler);

    // Also check current tab URL on mount
    try {
      ext?.tabs?.query?.({ active: true, currentWindow: true }, (tabs: any[]) => {
        const url = tabs?.[0]?.url || '';
        if (url.includes('youtube.com/watch')) setPageType('youtube');
        else if (url.endsWith('.pdf') || url.includes('pdf')) setPageType('pdf');
      });
    } catch { /* not available outside extension context */ }

    return () => ext?.runtime?.onMessage?.removeListener(handler);
  }, []);

  const handleHistoryToggle = useCallback(() => {
    setHistoryOpen(prev => !prev);
    setExportOpen(false);
  }, []);

  const handleNewConversation = useCallback(() => {
    newConversation();
  }, [newConversation]);

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

  const chatModel = featureModels['chat']?.replace(/:cloud$/, '').split('/').pop() || undefined;

  const handleExport = (format: 'markdown' | 'json' | 'text') => {
    exportChat(messages, format, chatModel);
    setExportOpen(false);
  };

  // Track scroll position to add shadow
  useEffect(() => {
    const panel = headerRef.current?.parentElement?.querySelector('[data-scroll-panel]')
      || headerRef.current?.nextElementSibling?.nextElementSibling // message panel after offline banner
      || headerRef.current?.nextElementSibling; // message panel directly

    if (!panel) return;

    const handleScroll = () => {
      setScrolled(panel.scrollTop > 4);
    };

    panel.addEventListener('scroll', handleScroll, { passive: true });
    return () => panel.removeEventListener('scroll', handleScroll);
  }, []);

  const handleRetry = () => {
    useStore.getState().setBackendStatus('connecting');
    fetchStatus();
  };

  const pillLabel = status === 'online'
    ? 'connected'
    : status === 'connecting'
      ? 'reconnecting'
      : 'offline';

  const moodGlow = getMoodGlowColor(mood);

  return (
    <>
      <header
        ref={headerRef}
        className={`header-bar flex items-center justify-between px-3 h-12 flex-shrink-0 relative z-10${scrolled ? ' header-scrolled' : ''}`}
      >
        {/* Gradient accent line at bottom */}
        <div className="header-gradient-line" />

        {/* Logo */}
        <div className="header-logo select-none">
          <span
            className="header-logo-dot"
            style={{
              background: status === 'offline'
                ? 'radial-gradient(circle, #f87171, #ef4444)'
                : isThinking
                  ? 'radial-gradient(circle, #c4b5fd, #7c3aed)'
                  : status === 'online'
                    ? 'radial-gradient(circle, #6ee7b7, #10b981)'
                    : 'var(--logo-dot-bg)',
              boxShadow: status === 'offline'
                ? '0 0 0 2px rgba(239,68,68,0.22), 0 0 10px rgba(239,68,68,0.5)'
                : isThinking
                  ? '0 0 0 2px rgba(124,58,237,0.3), 0 0 14px rgba(167,139,250,0.7)'
                  : status === 'online'
                    ? moodGlow
                      ? `0 0 0 2px rgba(16,185,129,0.22), 0 0 10px rgba(16,185,129,0.3), 0 0 16px ${moodGlow}66`
                      : '0 0 0 2px rgba(16,185,129,0.22), 0 0 10px rgba(16,185,129,0.5)'
                    : 'var(--logo-dot-shadow)',
              animation: status === 'connecting'
                ? 'pulse 1s ease-in-out infinite'
                : isThinking
                  ? 'auraDotThink 0.8s ease-in-out infinite'
                  : status === 'offline'
                    ? 'none'
                    : 'breatheGlow 2.5s ease-in-out infinite',
            }}
          />
          <span className="header-logo-text">AURA</span>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-1.5">
          {/* Mood */}
          {mood && (
            <span className="text-sm leading-none mr-0.5">{mood}</span>
          )}

          {/* Connection dot only — minimal */}
          <span
            className={`header-conn-dot${status === 'online' ? ' header-conn-dot-pulse' : ''}`}
            title={`${pillLabel} — ${getServerLabel()}`}
            style={{
              background: STATUS_DOT[status] || '#ef4444',
              animation: status === 'connecting' ? 'pulse 1s ease-in-out infinite' : undefined,
              width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
            }}
          />

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            <span className={`theme-icon-wrap${theme === 'dark' ? ' theme-show-sun' : ' theme-show-moon'}`}>
              <Sun size={13} className="theme-icon-sun" />
              <Moon size={13} className="theme-icon-moon" />
            </span>
          </button>

          {/* Export dropdown — icon only, no label */}
          {messages.length > 0 && (
            <div ref={exportRef} style={{ position: 'relative', display: 'inline-flex' }}>
              <button onClick={() => setExportOpen(!exportOpen)} title="Export" aria-label="Export conversation" aria-expanded={exportOpen} className="btn-ghost">
                <Download size={13} />
              </button>
              {exportOpen && (
                <div className="dropdown-menu" style={{ position: 'absolute', top: '100%', right: 0, marginTop: 4, minWidth: 155 }}>
                  {[
                    { label: 'Export as Markdown', fmt: 'markdown' as const },
                    { label: 'Export as JSON', fmt: 'json' as const },
                    { label: 'Export as Text', fmt: 'text' as const },
                  ].map((item) => (
                    <button key={item.label} onClick={() => handleExport(item.fmt)} className="dropdown-item">
                      {item.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* History button */}
          <div style={{ position: 'relative', display: 'inline-flex' }}>
            <button onClick={handleHistoryToggle} title="History" aria-label="Chat history" aria-expanded={historyOpen} className="btn-ghost" style={{ color: historyOpen ? 'var(--pl)' : undefined }}>
              <Clock size={13} />
            </button>
            <ConversationHistory open={historyOpen} onClose={() => setHistoryOpen(false)} />
          </div>

          {/* New button */}
          <button onClick={handleNewConversation} title="New chat" aria-label="New conversation" className="btn-ghost">
            <Plus size={14} />
          </button>
        </div>
      </header>

      {/* Page-aware context bar */}
      {pageType === 'youtube' && (
        <div className="page-context-bar">
          <span className="page-context-label">{'\u25B6'} YouTube detected</span>
          <div style={{ flex: 1 }} />
          <button className="page-context-btn" onClick={() => { setPanel('youtube'); window.dispatchEvent(new CustomEvent('yt-action', { detail: 'summarize' })); }}>Summarize</button>
          <button className="page-context-btn" onClick={() => { setPanel('youtube'); window.dispatchEvent(new CustomEvent('yt-action', { detail: 'transcript' })); }}>Transcript</button>
        </div>
      )}
      {pageType === 'pdf' && (
        <div className="page-context-bar">
          <span className="page-context-label">{'\uD83D\uDCC4'} PDF detected</span>
          <div style={{ flex: 1 }} />
          <button className="page-context-btn" onClick={() => setPanel('pdf')}>Chat with PDF</button>
        </div>
      )}

      {/* Offline banner */}
      {status === 'offline' && (
        <div
          className="flex items-center justify-between px-3 py-2 flex-shrink-0"
          style={{
            background: 'linear-gradient(90deg, rgba(245,158,11,0.15), rgba(239,68,68,0.10))',
            borderBottom: '1px solid rgba(245,158,11,0.3)',
            fontSize: '11.5px',
            color: '#fbbf24',
          }}
        >
          <span style={{ lineHeight: 1.4 }}>
            Backend offline — Start with: <code style={{ fontSize: '10.5px', opacity: 0.85 }}>python run_web.py</code>
          </span>
          <button
            onClick={handleRetry}
            className="flex items-center gap-1 px-2 py-0.5 transition-all duration-150"
            style={{
              background: 'rgba(245,158,11,0.18)',
              border: '1px solid rgba(245,158,11,0.35)',
              borderRadius: '4px',
              color: '#fbbf24',
              fontSize: '10.5px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              flexShrink: 0,
            }}
            title="Retry connection"
          >
            <RefreshCw size={10} />
            Retry
          </button>
        </div>
      )}
    </>
  );
}
