import React, { useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { md } from '../markdown';

const TYPES = ['Essay', 'Article', 'Email', 'Story', 'Report', 'Letter', 'Poem'];
const TONES = ['Formal', 'Casual', 'Persuasive', 'Informative', 'Creative'];
const LENS = ['Short', 'Medium', 'Long'];

export default function WritePanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [tab, setTab] = useState<'write' | 'improve'>('write');
  const [type, setType] = useState('Essay');
  const [tone, setTone] = useState('Formal');
  const [len, setLen] = useState('Medium');
  const [result, setResult] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const doWrite = () => {
    const text = inputRef.current?.value.trim();
    if (!text) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { alert('AURA is offline.'); return; }
    if (activeStream) return;

    setResult('');

    const prompt = tab === 'improve'
      ? `Improve the following text. Make it ${tone.toLowerCase()} in tone and ${len.toLowerCase()} in length. Output only the improved text:\n\n${text}`
      : `Write a ${tone.toLowerCase()}, ${len.toLowerCase()}-length ${type} about: ${text}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setResult(''),
      onDone: (rawText) => setResult(rawText),
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('write'), conversation_id: null }));
  };

  const stream = useStore(s => s.activeStream);
  const streamText = (stream && stream !== true && stream.type === 'write') ? stream.rawText : null;
  const displayText = streamText !== null ? streamText : result;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Tabs */}
      <div className="flex flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        {(['write', 'improve'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: '9px',
              background: 'none',
              border: 'none',
              borderBottom: tab === t ? '2px solid var(--p)' : '2px solid transparent',
              color: tab === t ? 'var(--pl)' : 'var(--mu)',
              fontSize: '12px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontWeight: tab === t ? 500 : 400,
            }}
          >
            {t === 'write' ? 'Write' : 'Improve'}
          </button>
        ))}
      </div>

      <div className="flex-1 flex flex-col gap-2 p-3 overflow-hidden">
        {/* Options */}
        {tab === 'write' && (
          <div className="flex flex-wrap gap-1 flex-shrink-0">
            {TYPES.map(t => (
              <button
                key={t}
                onClick={() => setType(t)}
                style={{
                  padding: '3px 9px',
                  background: type === t ? 'var(--pg2)' : 'var(--s2)',
                  border: `1px solid ${type === t ? 'var(--p)' : 'var(--b1)'}`,
                  borderRadius: 'var(--r-pill)',
                  color: type === t ? 'var(--pl)' : 'var(--mu)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                {t}
              </button>
            ))}
          </div>
        )}
        <div className="flex flex-wrap gap-1 flex-shrink-0">
          {TONES.map(t => (
            <button
              key={t}
              onClick={() => setTone(t)}
              style={{
                padding: '3px 9px',
                background: tone === t ? 'var(--pg2)' : 'var(--s2)',
                border: `1px solid ${tone === t ? 'var(--p)' : 'var(--b1)'}`,
                borderRadius: 'var(--r-pill)',
                color: tone === t ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1 flex-shrink-0">
          {LENS.map(t => (
            <button
              key={t}
              onClick={() => setLen(t)}
              style={{
                padding: '3px 9px',
                background: len === t ? 'var(--pg2)' : 'var(--s2)',
                border: `1px solid ${len === t ? 'var(--p)' : 'var(--b1)'}`,
                borderRadius: 'var(--r-pill)',
                color: len === t ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {t}
            </button>
          ))}
        </div>

        <textarea
          ref={inputRef}
          placeholder={tab === 'improve' ? 'Paste text to improve…' : 'Describe what to write…'}
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doWrite(); } }}
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
            flexShrink: 0,
          }}
        />

        <div className="flex items-center justify-between flex-shrink-0">
          <ModelPill featureKey="write" />
          <button
            onClick={doWrite}
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
            {activeStream ? '…' : 'Write'}
          </button>
        </div>

        {/* Result */}
        {(displayText || activeStream) && (
          <div
            className="flex-1 overflow-y-auto"
            style={{
              background: 'var(--s1)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
            }}
          >
            {displayText ? (
              <div
                className="md-body"
                style={{ fontSize: '12.5px', lineHeight: 1.65 }}
                dangerouslySetInnerHTML={{ __html: md(displayText) }}
              />
            ) : (
              <div className="dots"><span /><span /><span /></div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
