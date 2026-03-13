import React, { useRef } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';

const QUICK_ACTIONS = [
  { label: 'Summarize', prompt: 'Summarize this: ' },
  { label: 'Explain', prompt: 'Explain this in simple terms: ' },
  { label: 'Translate', prompt: 'Translate this to English: ' },
  { label: 'Key Points', prompt: 'Extract key points from: ' },
  { label: 'Critique', prompt: 'Critically analyze: ' },
  { label: 'Simplify', prompt: 'Simplify this: ' },
];

export default function AskPanel() {
  const { pendingCtx, setPanel, wsReady, setPendingCtx } = useStore();
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const goToChat = (msg: string) => {
    if (!wsReady) return;
    if (pendingCtx) setPendingCtx(pendingCtx); // keep it for chat
    setPanel('chat');
    // Trigger send via custom event (ChatPanel listens)
    window.dispatchEvent(new CustomEvent('aura-send', { detail: { text: msg } }));
  };

  const handleQuick = (prompt: string) => {
    if (!pendingCtx?.text) return;
    const msg = prompt + pendingCtx.text;
    setPendingCtx(null);
    setPanel('chat');
    window.dispatchEvent(new CustomEvent('aura-send', { detail: { text: msg } }));
  };

  const handleCustom = () => {
    const q = inputRef.current?.value.trim();
    if (!q) return;
    const msg = pendingCtx?.text ? q + '\n\n[Context]\n' + pendingCtx.text : q;
    setPendingCtx(null);
    setPanel('chat');
    window.dispatchEvent(new CustomEvent('aura-send', { detail: { text: msg } }));
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      {/* Context preview */}
      {pendingCtx?.text && (
        <div
          style={{
            background: 'var(--pg)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            padding: '10px',
            fontSize: '11.5px',
          }}
        >
          <div style={{ color: 'var(--pl)', fontWeight: 500, marginBottom: 4 }}>
            {pendingCtx.title || pendingCtx.url || 'Selected text'}
          </div>
          <div style={{ color: 'var(--mu)' }}>
            {pendingCtx.text.slice(0, 300)}{pendingCtx.text.length > 300 ? '…' : ''}
          </div>
        </div>
      )}

      {/* Quick action buttons */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 8 }}>
          Quick Actions
        </div>
        <div className="flex flex-wrap gap-2">
          {QUICK_ACTIONS.map(a => (
            <button
              key={a.label}
              onClick={() => handleQuick(a.prompt)}
              disabled={!pendingCtx?.text}
              style={{
                padding: '5px 12px',
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-pill)',
                color: pendingCtx?.text ? 'var(--tx)' : 'var(--di)',
                fontSize: '12px',
                cursor: pendingCtx?.text ? 'pointer' : 'not-allowed',
                fontFamily: 'inherit',
              }}
            >
              {a.label}
            </button>
          ))}
        </div>
      </div>

      {/* Custom question */}
      <div className="flex flex-col gap-2">
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)' }}>
          Ask a Question
        </div>
        <textarea
          ref={inputRef}
          placeholder="Ask anything about the selected text…"
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleCustom(); } }}
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12.5px',
            padding: '8px 10px',
            resize: 'none',
            height: 80,
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <div className="flex items-center justify-between">
          <ModelPill featureKey="ask" />
          <button
            onClick={handleCustom}
            style={{
              background: 'var(--p)',
              border: 'none',
              borderRadius: 'var(--r-md)',
              color: 'white',
              padding: '7px 18px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            Ask
          </button>
        </div>
      </div>

      {!pendingCtx?.text && (
        <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 8 }}>
          Select text on any page and right-click → Ask AURA
        </div>
      )}
    </div>
  );
}
