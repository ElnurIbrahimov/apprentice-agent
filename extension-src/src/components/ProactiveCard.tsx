import React, { useEffect, useRef, useState, useCallback } from 'react';
import { X, Check } from 'lucide-react';
import { useStore } from '../store';
import type { ProactiveMessage } from '../types';

const AUTO_DISMISS_MS = 30_000;

// ---- Single card ----

interface CardProps {
  message: ProactiveMessage;
  onDismiss: (id: string) => void;
  onAccept: (id: string) => void;
}

function ProactiveCard({ message, onDismiss, onAccept }: CardProps) {
  const [visible, setVisible] = useState(false);
  const [exiting, setExiting] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Slide in on mount (rAF delay so CSS transition fires)
  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 16);
    return () => clearTimeout(t);
  }, []);

  const triggerDismiss = useCallback(() => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    setExiting(true);
    setTimeout(() => onDismiss(message.id), 280);
  }, [message.id, onDismiss]);

  // Auto-dismiss after 30s
  useEffect(() => {
    timerRef.current = setTimeout(triggerDismiss, AUTO_DISMISS_MS);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [triggerDismiss]);

  const handleAccept = useCallback(() => {
    if (timerRef.current) { clearTimeout(timerRef.current); timerRef.current = null; }
    setExiting(true);
    setTimeout(() => onAccept(message.id), 280);
  }, [message.id, onAccept]);

  return (
    <div
      style={{
        transform: visible && !exiting ? 'translateY(0)' : 'translateY(-110%)',
        opacity: visible && !exiting ? 1 : 0,
        transition: 'transform 0.28s cubic-bezier(0.34,1.3,0.64,1), opacity 0.25s ease',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        padding: '7px 8px 7px 10px',
        borderRadius: 10,
        background: 'var(--glass)',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
        boxShadow: '0 2px 16px rgba(0,0,0,0.45), inset 0 0 0 1px rgba(124,58,237,0.28)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Gradient shimmer layer */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: 10,
          background: 'linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(59,130,246,0.08) 100%)',
          pointerEvents: 'none',
        }}
      />

      {/* Aura avatar dot */}
      <div
        style={{
          width: 22,
          height: 22,
          borderRadius: '50%',
          flexShrink: 0,
          background: 'radial-gradient(circle at 35% 35%, var(--pl), var(--p))',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 10,
          fontWeight: 700,
          color: 'white',
          boxShadow: '0 0 8px rgba(124,58,237,0.45)',
          zIndex: 1,
        }}
      >
        A
      </div>

      {/* Message text */}
      <span
        style={{
          flex: 1,
          fontSize: 11.5,
          lineHeight: 1.45,
          color: 'var(--tx)',
          zIndex: 1,
          minWidth: 0,
        }}
      >
        {message.text}
      </span>

      {/* Buttons */}
      <div style={{ display: 'flex', gap: 4, flexShrink: 0, zIndex: 1 }}>
        {/* Accept ("Tell me more") */}
        <button
          onClick={handleAccept}
          title="Tell me more"
          aria-label="Tell me more"
          style={{
            width: 24,
            height: 24,
            borderRadius: 6,
            border: '1px solid rgba(124,58,237,0.35)',
            background: 'rgba(124,58,237,0.18)',
            color: 'var(--pl)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 0,
            transition: 'background 0.15s, border-color 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'rgba(124,58,237,0.32)';
            e.currentTarget.style.borderColor = 'rgba(124,58,237,0.6)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'rgba(124,58,237,0.18)';
            e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)';
          }}
        >
          <Check size={12} strokeWidth={2.5} />
        </button>

        {/* Dismiss */}
        <button
          onClick={triggerDismiss}
          title="Dismiss"
          aria-label="Dismiss suggestion"
          style={{
            width: 24,
            height: 24,
            borderRadius: 6,
            border: '1px solid var(--b1)',
            background: 'transparent',
            color: 'var(--mu)',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 0,
            transition: 'background 0.15s, color 0.15s',
          }}
          onMouseEnter={e => {
            e.currentTarget.style.background = 'var(--b1)';
            e.currentTarget.style.color = 'var(--tx)';
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = 'var(--mu)';
          }}
        >
          <X size={11} strokeWidth={2.5} />
        </button>
      </div>
    </div>
  );
}

// ---- Container ----

interface ContainerProps {
  /** Called when user accepts a suggestion — injects the text as a chat message */
  onAccept: (text: string) => void;
}

export default function ProactiveNotifications({ onAccept }: ContainerProps) {
  const proactiveMessages = useStore(s => s.proactiveMessages);
  const dismissProactive = useStore(s => s.dismissProactive);
  const acceptProactive = useStore(s => s.acceptProactive);

  const handleAccept = useCallback((id: string) => {
    acceptProactive(id, onAccept);
  }, [acceptProactive, onAccept]);

  if (proactiveMessages.length === 0) return null;

  return (
    <div
      style={{
        position: 'relative',
        zIndex: 30,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        padding: '6px 8px 2px',
        overflow: 'hidden',
      }}
      aria-live="polite"
      aria-label="Proactive suggestions"
    >
      {proactiveMessages.map(msg => (
        <ProactiveCard
          key={msg.id}
          message={msg}
          onDismiss={dismissProactive}
          onAccept={handleAccept}
        />
      ))}
    </div>
  );
}
