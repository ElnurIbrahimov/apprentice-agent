import React from 'react';
import { FileText, Lightbulb, List, HelpCircle } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { useStore } from '../store';

interface Props {
  onChip: (q: string) => void;
}

interface ChipDef {
  label: string;
  desc: string;
  q: string;
  Icon: LucideIcon;
}

const CHIPS: ChipDef[] = [
  { label: 'Summarize', desc: 'Get a quick summary of this page', q: 'Summarize this page for me', Icon: FileText },
  { label: 'Explain', desc: 'Break it down in simple terms', q: 'Explain this page in simple terms', Icon: Lightbulb },
  { label: 'Key points', desc: 'Extract the main takeaways', q: 'Extract the key points from this page', Icon: List },
  { label: 'Ask anything', desc: 'I can help with any question', q: 'What can you help me with?', Icon: HelpCircle },
];

export default function HomeScreen({ onChip }: Props) {
  const { wsReady, modelName } = useStore();

  return (
    <div className="flex flex-col items-center justify-center flex-1 px-6 gap-6">
      {/* Logo with breathing animation */}
      <div className="flex flex-col items-center gap-3">
        <div
          className="aura-logo-home w-12 h-12 rounded-xl flex items-center justify-center text-xl"
          style={{
            background: 'linear-gradient(135deg, rgba(124,58,237,0.2) 0%, rgba(124,58,237,0.08) 100%)',
            border: '1px solid rgba(124,58,237,0.25)',
            color: 'var(--pl)',
          }}
        >
          <span style={{ filter: 'drop-shadow(0 0 6px rgba(167,139,250,0.5))' }}>
            &#10022;
          </span>
        </div>
        <div style={{ fontSize: '13px', fontWeight: 700, letterSpacing: '0.14em', color: 'rgba(224,214,255,0.9)' }}>
          AURA
        </div>
        <div style={{ fontSize: '11px', color: 'var(--mu)', textAlign: 'center', lineHeight: 1.5 }}>
          Your private AI &mdash; always present
        </div>

        {/* Connection status + model */}
        <div
          className="flex items-center gap-2 mt-1"
          style={{ fontSize: '10px' }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{
              background: wsReady ? 'var(--gr)' : 'var(--rd)',
              boxShadow: wsReady ? '0 0 6px rgba(16,185,129,0.5)' : '0 0 6px rgba(239,68,68,0.4)',
            }}
          />
          <span style={{ color: 'var(--mu)' }}>
            {wsReady
              ? (modelName || 'Connected')
              : 'Offline'
            }
          </span>
        </div>
      </div>

      {/* Action cards */}
      <div className="flex flex-col gap-2 w-full max-w-[240px]">
        {CHIPS.map(chip => (
          <button
            key={chip.q}
            onClick={() => onChip(chip.q)}
            className="home-card w-full text-left flex items-center gap-3 px-3.5 py-3"
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            <div
              className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
              style={{
                background: 'rgba(124, 58, 237, 0.1)',
                border: '1px solid rgba(124, 58, 237, 0.12)',
              }}
            >
              <chip.Icon size={15} style={{ color: 'var(--pl)' }} />
            </div>
            <div className="flex-1 min-w-0">
              <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)', marginBottom: 1 }}>
                {chip.label}
              </div>
              <div style={{ fontSize: '10px', color: 'var(--mu)', lineHeight: 1.3 }}>
                {chip.desc}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
