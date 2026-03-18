import React, { useEffect, useRef, useState } from 'react';
import { Plus, Sun, Moon, RefreshCw } from 'lucide-react';
import { useStore } from '../store';
import { fetchStatus } from '../ws';

const STATUS_DOT: Record<string, string> = {
  online: '#10b981',
  connecting: '#f59e0b',
  offline: '#ef4444',
};

export default function Header() {
  const { wsReady, modelName, mood, clearAll, theme, toggleTheme, backendStatus } = useStore();
  const status = backendStatus || (wsReady ? 'online' : 'offline');
  const [scrolled, setScrolled] = useState(false);
  const headerRef = useRef<HTMLElement>(null);

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

          {/* Model name pill (when connected) */}
          {status === 'online' && modelName && (
            <div className="header-model-pill">
              <span className="truncate">{modelName}</span>
            </div>
          )}

          {/* Connection pill */}
          <div className="header-conn-pill">
            <span
              className={`header-conn-dot${status === 'online' ? ' header-conn-dot-pulse' : ''}`}
              style={{
                background: STATUS_DOT[status] || '#ef4444',
                animation: status === 'connecting' ? 'pulse 1s ease-in-out infinite' : undefined,
              }}
            />
            <span className="header-conn-label">
              {pillLabel}
            </span>
          </div>

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

          {/* New button */}
          <button
            onClick={clearAll}
            className="flex items-center gap-1 px-2 py-1 transition-all duration-150"
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-sm)',
              color: 'var(--mu)',
              fontSize: '11px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
            title="New conversation"
          >
            <Plus size={12} />
            <span>New</span>
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
