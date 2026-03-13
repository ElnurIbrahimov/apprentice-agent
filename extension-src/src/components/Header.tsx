import React from 'react';
import { Plus } from 'lucide-react';
import { useStore } from '../store';

export default function Header() {
  const { wsReady, modelName, mood, clearAll } = useStore();

  return (
    <header
      className="flex items-center justify-between px-3 h-12 flex-shrink-0 relative z-10"
      style={{
        background: 'var(--glass)',
        borderBottom: '1px solid var(--b1)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-2 select-none"
        style={{ fontSize: '12.5px', fontWeight: 700, letterSpacing: '0.14em', color: 'rgba(224,214,255,0.92)' }}
      >
        <span
          className="w-[7px] h-[7px] rounded-full flex-shrink-0"
          style={{
            background: 'radial-gradient(circle, #e0d6ff, #7c3aed)',
            boxShadow: '0 0 0 2px rgba(124,58,237,0.22), 0 0 14px rgba(167,139,250,0.65)',
            animation: 'pulse 2.5s ease-in-out infinite',
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
            style={{ background: wsReady ? '#10b981' : '#ef4444' }}
          />
          <span
            className="text-[11px] truncate"
            style={{ color: 'var(--mu)' }}
          >
            {wsReady ? (modelName || 'connected') : 'offline'}
          </span>
        </div>

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
  );
}
