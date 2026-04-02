import React from 'react';
import { X } from 'lucide-react';

interface OverlayModalProps {
  children: React.ReactNode;
  contentStyle?: React.CSSProperties;
  icon?: React.ReactNode;
  onClose: () => void;
  title: string;
  zIndex?: number;
}

export default function OverlayModal({
  children,
  contentStyle,
  icon,
  onClose,
  title,
  zIndex = 10010,
}: OverlayModalProps) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 'min(420px, 100%)',
          background: '#12131c',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 16,
          boxShadow: '0 20px 60px rgba(0,0,0,0.4)',
          padding: 18,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
          ...contentStyle,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {icon}
          <div style={{ fontSize: '13px', fontWeight: 700, color: 'var(--tx)' }}>{title}</div>
          <div style={{ flex: 1 }} />
          <button
            onClick={onClose}
            style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}
          >
            <X size={14} />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
