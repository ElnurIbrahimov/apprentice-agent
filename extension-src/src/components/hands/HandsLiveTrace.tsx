import React from 'react';
import { Activity } from 'lucide-react';
import type { HandLiveTrace } from '../../types';

interface Props {
  traces: HandLiveTrace[];
}

export default function HandsLiveTrace({ traces }: Props) {
  if (traces.length === 0) return null;
  const recent = traces.slice(-5);
  return (
    <div
      style={{
        borderRadius: 10,
        background: 'rgba(59,130,246,0.06)',
        border: '1px solid rgba(59,130,246,0.3)',
        padding: '8px 10px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <Activity size={11} style={{ color: 'var(--bl, #3b82f6)' }} />
        <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--bl, #3b82f6)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          Live trace
        </span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {recent.map((t, i) => (
          <div key={i} style={{
            fontSize: 10.5,
            color: 'var(--mu)',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            fontFamily: 'monospace',
          }}>
            <span style={{ color: 'var(--pl)' }}>[{t.step}]</span>{' '}
            <span style={{ color: 'var(--tx)' }}>{t.hand}</span>{' '}
            <span>— {t.description}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
