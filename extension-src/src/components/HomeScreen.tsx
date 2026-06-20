import React from 'react';
import { FileText, Lightbulb, List, HelpCircle, Zap, Globe } from 'lucide-react';
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
  { label: 'Summarize', desc: 'Quick summary of this page', q: 'Summarize this page for me', Icon: FileText },
  { label: 'Explain', desc: 'Break down in simple terms', q: 'Explain this page in simple terms', Icon: Lightbulb },
  { label: 'Key points', desc: 'Extract main takeaways', q: 'Extract the key points from this page', Icon: List },
  { label: 'Quick answer', desc: 'Ask me anything', q: 'What can you help me with?', Icon: Zap },
  { label: 'Translate', desc: 'Translate page content', q: 'Translate this page to English', Icon: Globe },
  { label: 'Deep dive', desc: 'Research this topic further', q: 'Do a deep research on this topic', Icon: HelpCircle },
];

export default function HomeScreen({ onChip }: Props) {
  const { wsReady, modelName } = useStore();

  const isMac = navigator.platform?.toLowerCase().includes('mac');
  const modKey = isMac ? '\u2318' : 'Ctrl';

  return (
    <div className="flex flex-col items-center justify-center flex-1 px-5 gap-5 home-screen-enter">
      {/* Hero section — logo + tagline */}
      <div className="flex flex-col items-center gap-2" style={{ marginTop: -12 }}>
        {/* Large breathing logo */}
        <div
          className="aura-logo-home"
          style={{
            width: 56,
            height: 56,
            borderRadius: 16,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 26,
            background: 'linear-gradient(135deg, rgba(124,58,237,0.22) 0%, rgba(124,58,237,0.06) 100%)',
            border: '1px solid rgba(124,58,237,0.25)',
            color: 'var(--pl)',
            position: 'relative',
          }}
        >
          <span style={{ filter: 'drop-shadow(0 0 10px rgba(167,139,250,0.6))' }}>
            &#10022;
          </span>
          {/* Outer glow ring */}
          <div
            className="aura-logo-ring"
            style={{
              position: 'absolute',
              inset: -3,
              borderRadius: 19,
              border: '1px solid rgba(124, 58, 237, 0.12)',
              pointerEvents: 'none',
            }}
          />
        </div>

        {/* Brand name */}
        <div style={{
          fontSize: 15,
          fontWeight: 700,
          letterSpacing: '0.16em',
          color: 'var(--logo-text)',
          marginTop: 6,
        }}>
          AURA
        </div>

        {/* Tagline — lighter weight */}
        <div style={{
          fontSize: 12,
          fontWeight: 300,
          color: 'var(--mu)',
          letterSpacing: '0.04em',
        }}>
          Your AI, everywhere.
        </div>
      </div>

      {/* Quick action cards — 2-column grid */}
      <div
        className="home-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 8,
          width: '100%',
          maxWidth: 280,
        }}
      >
        {CHIPS.map(chip => (
          <button
            key={chip.q}
            onClick={() => onChip(chip.q)}
            className="home-card"
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'flex-start',
              gap: 8,
              padding: '12px 12px 10px',
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              cursor: 'pointer',
              fontFamily: 'inherit',
              textAlign: 'left',
              width: '100%',
            }}
          >
            {/* Icon in purple-tinted circle */}
            <div
              style={{
                width: 30,
                height: 30,
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(124, 58, 237, 0.12)',
                border: '1px solid rgba(124, 58, 237, 0.1)',
                flexShrink: 0,
              }}
            >
              <chip.Icon size={14} style={{ color: 'var(--pl)' }} />
            </div>
            {/* Text */}
            <div>
              <div style={{ fontSize: '11.5px', fontWeight: 600, color: 'var(--tx)', lineHeight: 1.3 }}>
                {chip.label}
              </div>
              <div style={{ fontSize: '9.5px', color: 'var(--mu)', lineHeight: 1.35, marginTop: 2 }}>
                {chip.desc}
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* Keyboard shortcut hint */}
      <div style={{
        fontSize: 10,
        color: 'var(--di)',
        display: 'flex',
        alignItems: 'center',
        gap: 5,
        marginTop: 4,
      }}>
        <kbd style={{
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '1px 5px',
          fontSize: 9,
          fontFamily: 'inherit',
          fontWeight: 500,
          background: 'var(--s3)',
          border: '1px solid var(--b1)',
          borderRadius: 4,
          color: 'var(--mu)',
          lineHeight: 1.6,
        }}>
          {modKey}+K  </kbd>
        <span>to switch panel</span>
      </div>

      {/* Footer — model + connection status */}
      <div
        className="flex items-center gap-2"
        style={{
          fontSize: '9.5px',
          color: 'var(--di)',
          paddingTop: 2,
        }}
      >
        <span
          style={{
            display: 'inline-block',
            width: 5,
            height: 5,
            borderRadius: '50%',
            background: wsReady ? 'var(--gr)' : 'var(--rd)',
            boxShadow: wsReady
              ? '0 0 6px rgba(16,185,129,0.5)'
              : '0 0 6px rgba(239,68,68,0.4)',
          }}
        />
        <span style={{ color: 'var(--mu)' }}>
          {wsReady
            ? (modelName ? `${modelName} · Connected` : 'Connected')
            : 'Offline'
          }
        </span>
      </div>
    </div>
  );
}
