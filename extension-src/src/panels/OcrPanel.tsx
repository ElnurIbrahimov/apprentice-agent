import React, { useState, useEffect } from 'react';
import { Camera, MessageSquare, Languages, Copy } from 'lucide-react';
import { useStore } from '../store';
import ext from '../ext';

export default function OcrPanel() {
  const { setPendingCtx, setPanel } = useStore();
  const [text, setText] = useState('');
  const [status, setStatus] = useState('');

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail.error && detail.error !== '' && detail.error !== 'Cancelled') {
        setStatus('⚠ OCR error: ' + detail.error);
        setText('');
      } else if (detail.text) {
        setText(detail.text);
        setStatus('');
      }
    };
    window.addEventListener('ocr-result', handler);
    return () => window.removeEventListener('ocr-result', handler);
  }, []);

  const capture = () => {
    ext?.runtime?.sendMessage({ type: 'OCR_START' });
    setStatus('Draw a region on the page…');
    setText('');
  };

  const toChat = () => {
    if (!text) return;
    setPendingCtx({ text, title: 'OCR extract', url: '' });
    setPanel('chat');
  };

  const toTranslate = () => {
    if (!text) return;
    // Store in global for TranslatePanel to pick up
    (window as any).__ocrText = text;
    window.dispatchEvent(new CustomEvent('ocr-to-translate', { detail: { text } }));
    setPanel('translate');
  };

  const copy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('ocr-copy-btn');
      if (btn) { btn.textContent = '✓ Copied'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); }
    });
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      <button
        onClick={capture}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
          background: 'var(--p)',
          border: 'none',
          borderRadius: 'var(--r-md)',
          color: 'white',
          padding: '10px',
          cursor: 'pointer',
          fontSize: '13px',
          fontFamily: 'inherit',
          fontWeight: 500,
        }}
      >
        <Camera size={16} />
        Capture Screen Region
      </button>

      {status && (
        <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center' }}>{status}</div>
      )}

      {text && (
        <>
          <div
            style={{
              flex: 1,
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
              fontSize: '12.5px',
              color: 'var(--tx)',
              overflowY: 'auto',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.6,
              minHeight: 80,
            }}
          >
            {text}
          </div>

          <div className="flex gap-2">
            <button
              onClick={toChat}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--tx)',
                padding: '7px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              <MessageSquare size={13} /> Chat
            </button>
            <button
              onClick={toTranslate}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--tx)',
                padding: '7px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              <Languages size={13} /> Translate
            </button>
            <button
              id="ocr-copy-btn"
              onClick={copy}
              style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 6,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                color: 'var(--tx)',
                padding: '7px',
                cursor: 'pointer',
                fontSize: '12px',
                fontFamily: 'inherit',
              }}
            >
              <Copy size={13} /> Copy
            </button>
          </div>
        </>
      )}

      {!text && !status && (
        <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 16 }}>
          Click "Capture" then draw a region on any page to extract text
        </div>
      )}
    </div>
  );
}
