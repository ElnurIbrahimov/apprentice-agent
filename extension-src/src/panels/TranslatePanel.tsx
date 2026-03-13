import React, { useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';

const LANGS = [
  'auto', 'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
  'Russian', 'Chinese', 'Japanese', 'Korean', 'Arabic', 'Hindi', 'Turkish',
  'Dutch', 'Polish', 'Swedish', 'Azerbaijani',
];

export default function TranslatePanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [from, setFrom] = useState('auto');
  const [to, setTo] = useState('English');
  const [output, setOutput] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const doTranslate = () => {
    const text = inputRef.current?.value.trim();
    if (!text) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
      setOutput('AURA is offline.');
      return;
    }
    if (activeStream) return;

    setOutput('Translating…');
    const fromLang = from === 'auto' ? 'the detected language' : from;
    const prompt = `Translate from ${fromLang} to ${to}. Output only the translation, no explanation:\n\n${text}`;

    setActiveStream({
      type: 'translate',
      rawText: '',
      onFirstChunk: () => setOutput(''),
      onDone: (rawText) => setOutput(rawText),
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('translate'),
      conversation_id: null,
    }));
  };

  // Track streaming output
  const { activeStream: stream } = useStore();
  const streamText = (stream && stream !== true && stream.type === 'translate') ? stream.rawText : null;
  const displayText = streamText !== null ? streamText : output;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Lang selectors */}
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <select
          value={from}
          onChange={e => setFrom(e.target.value)}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '6px 8px',
            fontFamily: 'inherit',
          }}
        >
          {LANGS.map(l => <option key={l} value={l}>{l === 'auto' ? 'Auto-detect' : l}</option>)}
        </select>
        <span style={{ color: 'var(--mu)', fontSize: '14px' }}>→</span>
        <select
          value={to}
          onChange={e => setTo(e.target.value)}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '6px 8px',
            fontFamily: 'inherit',
          }}
        >
          {LANGS.filter(l => l !== 'auto').map(l => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      {/* Input */}
      <div className="flex-1 flex flex-col gap-3 p-3 overflow-hidden">
        <textarea
          ref={inputRef}
          placeholder="Enter text to translate…"
          autoFocus
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doTranslate(); } }}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12.5px',
            padding: '10px',
            resize: 'none',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />

        {/* Output */}
        <div
          style={{
            flex: 1,
            background: 'var(--s1)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            padding: '10px',
            fontSize: '12.5px',
            color: displayText ? 'var(--tx)' : 'var(--di)',
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {displayText || 'Translation will appear here…'}
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span style={{ fontSize: '11px', color: 'var(--mu)' }}>Model:</span>
            <ModelPill featureKey="translate" />
          </div>
          <button
            onClick={doTranslate}
            disabled={!!activeStream}
            style={{
              background: activeStream ? 'var(--s3)' : 'var(--p)',
              border: 'none',
              borderRadius: 'var(--r-md)',
              color: 'white',
              padding: '7px 18px',
              cursor: activeStream ? 'not-allowed' : 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            Translate
          </button>
        </div>
      </div>
    </div>
  );
}
