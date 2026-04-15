/**
 * ThinkingModeToggle — compact header control for system 1 / system 2 / auto.
 *
 * POSTs to /api/thinking-mode/set and polls /state on mount.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Zap, Brain, Repeat } from 'lucide-react';
import { thinkingMode } from '../api/client';
import type { ThinkingMode } from '../api/types';

const MODES: { id: ThinkingMode; label: string; icon: React.ReactNode; title: string }[] = [
  { id: 'auto', label: 'Auto', icon: <Repeat size={11} />, title: 'Auto — Aura picks system 1 or 2' },
  { id: 'system1', label: 'Fast', icon: <Zap size={11} />, title: 'System 1 — fast, intuitive' },
  { id: 'system2', label: 'Deep', icon: <Brain size={11} />, title: 'System 2 — deliberate, analytical' },
];

export default function ThinkingModeToggle() {
  const [mode, setMode] = useState<ThinkingMode>('auto');
  const [open, setOpen] = useState(false);
  const [load, setLoad] = useState<number | null>(null);

  useEffect(() => {
    thinkingMode
      .get()
      .then((s) => {
        if (s.mode) setMode(s.mode);
        if (typeof s.cognitive_load === 'number') setLoad(s.cognitive_load);
      })
      .catch(() => {});
  }, []);

  const change = useCallback(async (m: ThinkingMode) => {
    setMode(m);
    setOpen(false);
    try {
      await thinkingMode.set(m);
    } catch { /* silent */ }
  }, []);

  const current = MODES.find((m) => m.id === mode) || MODES[0];

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        title={`Thinking mode: ${current.label}${load != null ? ` · load ${(load * 100).toFixed(0)}%` : ''}`}
        aria-label="Thinking mode"
        style={{
          background: 'none',
          border: 'none',
          color: 'var(--di)',
          cursor: 'pointer',
          padding: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 3,
          borderRadius: 4,
          fontSize: 10,
        }}
      >
        {current.icon}
      </button>

      {open && (
        <div
          onMouseLeave={() => setOpen(false)}
          style={{
            position: 'absolute',
            top: '100%',
            right: 0,
            marginTop: 4,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 6,
            boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            zIndex: 50,
            minWidth: 130,
            overflow: 'hidden',
          }}
        >
          {MODES.map((m) => (
            <button
              key={m.id}
              onClick={() => change(m.id)}
              title={m.title}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                width: '100%',
                padding: '7px 10px',
                background: mode === m.id ? 'var(--b1)' : 'none',
                border: 'none',
                color: 'var(--tx)',
                fontSize: 11,
                textAlign: 'left',
                cursor: 'pointer',
              }}
            >
              {m.icon}
              {m.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
