import React from 'react';

interface Props {
  onChip: (q: string) => void;
}

const CHIPS = [
  { label: '✨ Summarize this page', q: 'Summarize this page for me' },
  { label: '💡 Explain this', q: 'Explain this page in simple terms' },
  { label: '🔑 Key points', q: 'Extract the key points from this page' },
  { label: '❓ Ask anything', q: 'What can you help me with?' },
];

export default function HomeScreen({ onChip }: Props) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-6 gap-6">
      {/* Logo */}
      <div className="flex flex-col items-center gap-2">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-lg"
          style={{
            background: 'var(--pg)',
            border: '1px solid rgba(124,58,237,0.3)',
            boxShadow: '0 0 20px rgba(124,58,237,0.2)',
          }}
        >
          ✦
        </div>
        <div style={{ fontSize: '12px', fontWeight: 600, letterSpacing: '0.1em', color: 'rgba(224,214,255,0.7)' }}>
          AURA
        </div>
        <div style={{ fontSize: '11px', color: 'var(--mu)', textAlign: 'center', lineHeight: 1.5 }}>
          Your private AI — always present
        </div>
      </div>

      {/* Chips */}
      <div className="flex flex-col gap-2 w-full max-w-[220px]">
        {CHIPS.map(chip => (
          <button
            key={chip.q}
            onClick={() => onChip(chip.q)}
            className="w-full text-left px-3 py-2 transition-all duration-150"
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--mu)',
              fontSize: '12px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            {chip.label}
          </button>
        ))}
      </div>
    </div>
  );
}
