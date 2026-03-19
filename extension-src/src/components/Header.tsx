import React, { useEffect, useRef, useState, useCallback } from 'react';
import { Plus, Sun, Moon, RefreshCw, Download, Clock, Server } from 'lucide-react';
import { useStore } from '../store';
import { fetchStatus } from '../ws';
import { exportChat } from '../exportChat';
import { getServerLabel } from '../api';
import ConversationHistory from './ConversationHistory';

const STATUS_DOT: Record<string, string> = {
  online: '#10b981',
  connecting: '#f59e0b',
  offline: '#ef4444',
};

export default function Header() {
  const { wsReady, modelName, mood, theme, toggleTheme, backendStatus, messages, featureModels, newConversation, conversations, loadConversationList, historyLoaded } = useStore();
  const status = backendStatus === 'online' ? 'online' : backendStatus === 'offline' ? 'offline' : 'connecting';
  const [scrolled, setScrolled] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const headerRef = useRef<HTMLElement>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  // Load conversation list on mount
  useEffect(() => {
    if (!historyLoaded) {
      loadConversationList();
    }
  }, [historyLoaded, loadConversationList]);

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
              background: status === 'online'
                ? 'radial-gradient(circle, #6ee7b7, #10b981)'
                : 'var(--logo-dot-bg)',
              boxShadow: status === 'online'
                ? '0 0 0 2px rgba(16,185,129,0.22), 0 0 10px rgba(16,185,129,0.5)'
                : 'var(--logo-dot-shadow)',
              animation: status === 'connecting'
                ? 'pulse 1s ease-in-out infinite'
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
              <button
                onClick={() => setExportOpen(!exportOpen)}
                title="Export"
                style={{
                  background: 'none', border: 'none', color: 'var(--mu)',
                  cursor: 'pointer', padding: 2, display: 'flex',
                }}
              >
                <Download size={13} />
              </button>
              {exportOpen && (
                <div
                  style={{
                    position: 'absolute',
                    top: '100%',
                    right: 0,
                    marginTop: 4,
                    background: 'var(--s2)',
                    border: '1px solid var(--b3)',
                    borderRadius: 6,
                    boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
                    zIndex: 50,
                    minWidth: 155,
                    overflow: 'hidden',
                  }}
                >
                  {[
                    { label: 'Export as Markdown', fmt: 'markdown' as const },
                    { label: 'Export as JSON', fmt: 'json' as const },
                    { label: 'Export as Text', fmt: 'text' as const },
                  ].map((item) => (
                    <button
                      key={item.label}
                      onClick={() => handleExport(item.fmt)}
                      style={{
                        display: 'block',
                        width: '100%',
                        padding: '7px 12px',
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
          )}

          {/* History button — icon only */}
          <div style={{ position: 'relative', display: 'inline-flex' }}>
            <button
              onClick={handleHistoryToggle}
              title="History"
              style={{
                background: 'none', border: 'none',
                color: historyOpen ? 'var(--pl)' : 'var(--mu)',
                cursor: 'pointer', padding: 2, display: 'flex',
              }}
            >
              <Clock size={13} />
            </button>
            <ConversationHistory open={historyOpen} onClose={() => setHistoryOpen(false)} />
          </div>

          {/* New button */}
          <button
            onClick={handleNewConversation}
            title="New chat"
            style={{
              background: 'none', border: 'none',
              color: 'var(--mu)', cursor: 'pointer', padding: 2, display: 'flex',
            }}
          >
            <Plus size={14} />
          </button>
        </div>
      </header>

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
