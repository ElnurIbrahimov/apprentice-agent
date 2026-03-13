import React, { useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { md } from '../markdown';
import { sendMsg } from '../ext';
import { HTTP } from '../api';

export default function SummaryPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [format, setFormat] = useState<'bullets' | 'paragraph' | 'tldr'>('bullets');
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');

  const summarize = async () => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setStatus('Loading page…');
    setResultHtml('');

    const resp = await sendMsg({ type: 'GET_PAGE_CONTENT' });
    if (!resp?.ok || !resp.text) {
      setStatus('Could not read page. Try on a regular website.');
      return;
    }

    const formatInstr = {
      bullets: 'Format as bullet points.',
      paragraph: 'Write as a concise paragraph.',
      tldr: 'Write a single sentence TL;DR.',
    }[format];

    const prompt = `Summarize this page. ${formatInstr}\n\nTitle: ${resp.title || 'Untitled'}\n\nContent:\n${resp.text.slice(0, 15000)}`;

    setStatus('Summarizing…');

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setStatus(''),
      onDone: (rawText) => setResultHtml(md(rawText)),
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('summary'),
      conversation_id: null,
    }));
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      {/* Format selector */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 8 }}>
          Format
        </div>
        <div className="flex gap-2">
          {(['bullets', 'paragraph', 'tldr'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFormat(f)}
              style={{
                flex: 1,
                padding: '6px',
                background: format === f ? 'var(--pg2)' : 'var(--s2)',
                border: `1px solid ${format === f ? 'var(--p)' : 'var(--b1)'}`,
                borderRadius: 'var(--r-md)',
                color: format === f ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              {f === 'bullets' ? 'Bullets' : f === 'paragraph' ? 'Paragraph' : 'TL;DR'}
            </button>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between">
        <ModelPill featureKey="summary" />
        <button
          onClick={summarize}
          disabled={!!activeStream}
          style={{
            background: activeStream ? 'var(--s3)' : 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '8px 20px',
            cursor: activeStream ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {activeStream ? '…' : 'Summarize This Page'}
        </button>
      </div>

      {status && (
        <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center' }}>{status}</div>
      )}

      {(isStreaming || resultHtml) && (
        <div
          className="flex-1 overflow-y-auto"
          style={{
            background: 'var(--s1)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            padding: '10px',
          }}
        >
          {isStreaming && !streamText ? (
            <div className="dots"><span /><span /><span /></div>
          ) : (
            <div
              className="md-body"
              style={{ fontSize: '12.5px', lineHeight: 1.65 }}
              dangerouslySetInnerHTML={{ __html: resultHtml || md(streamText || '') }}
            />
          )}
        </div>
      )}
    </div>
  );
}
