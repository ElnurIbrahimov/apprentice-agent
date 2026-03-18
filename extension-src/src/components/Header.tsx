import React from 'react';
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

  const handleRetry = () => {
    useStore.getState().setBackendStatus('connecting');
    fetchStatus();
  };

  const pillLabel = status === 'online'
    ? (modelName || 'connected')
    : status === 'connecting'
      ? 'reconnecting'
      : 'offline';

  return (
    <>
      <header
        className="flex items-center justify-between px-3 h-12 flex-shrink-0 relative z-10"
        style={{
          background: 'var(--glass)',
          borderBottom: status === 'offline' ? 'none' : '1px solid var(--b1)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
        }}
      >
        {/* Logo */}
        <div
          className="flex items-center gap-2 select-none"
          style={{ fontSize: '12.5px', fontWeight: 700, letterSpacing: '0.14em', color: 'var(--logo-text)' }}
        >
          <span
            className="w-[7px] h-[7px] rounded-full flex-shrink-0"
            style={{
              background: status === 'online'
                ? 'radial-gradient(circle, #6ee7b7, #10b981)'
                : 'var(--logo-dot-bg)',
              boxShadow: status === 'online'
                ? '0 0 0 2px rgba(16,185,129,0.22), 0 0 10px rgba(16,185,129,0.5)'
                : 'var(--logo-dot-shadow)',
              animation: status === 'connecting'
                ? 'pulse 1s ease-in-out infinite'
                : 'pulse 2.5s ease-in-out infinite',
            }}
          />
          AURA
        </div>

        {/* Right side */}
        <div className="flex items-center gap-1.5">
          {/* Mood */}
          {mood && (
            <span className="text-sm leading-none mr-0.5">{mood}</span>
          )}

          {/* Connection pill */}
          <div
            className="flex items-center gap-1.5 px-2.5 py-1 cursor-default overflow-hidden"
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-pill)',
              maxWidth: 140,
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full flex-shrink-0 transition-colors duration-300"
              style={{
                background: STATUS_DOT[status] || '#ef4444',
                animation: status === 'connecting' ? 'pulse 1s ease-in-out infinite' : undefined,
              }}
            />
            <span
              className="text-[11px] truncate"
              style={{ color: 'var(--mu)' }}
            >
              {pillLabel}
            </span>
          </div>

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? <Sun size={13} /> : <Moon size={13} />}
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
