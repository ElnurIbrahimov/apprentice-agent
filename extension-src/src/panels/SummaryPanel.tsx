import React, { useState, useRef } from 'react';
import { Copy, Check, ExternalLink } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { md } from '../markdown';
import { sendMsg } from '../ext';

type SummaryFormat = 'bullets' | 'paragraph' | 'tldr';
type SummaryLength = 'brief' | 'standard' | 'detailed';

export default function SummaryPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [format, setFormat] = useState<SummaryFormat>('bullets');
  const [length, setLength] = useState<SummaryLength>('standard');
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');
  const [resultRaw, setResultRaw] = useState('');
  const [pageInfo, setPageInfo] = useState<{ title: string; url: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const summarize = async () => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setStatus('Loading page...');
    setResultHtml('');
    setResultRaw('');
    setPageInfo(null);

    const resp = await sendMsg({ type: 'GET_PAGE_CONTENT' });
    if (!resp?.ok || !resp.text) {
      setStatus('Could not read page. Try on a regular website.');
      return;
    }

    setPageInfo({ title: resp.title || 'Untitled', url: resp.url || '' });

    const formatInstr: Record<SummaryFormat, string> = {
      bullets: 'Format as bullet points with key takeaways.',
      paragraph: 'Write as prose paragraphs.',
      tldr: 'Write a single sentence TL;DR.',
    };

    const lengthInstr: Record<SummaryLength, string> = {
      brief: 'Keep it very brief - maximum 3 sentences or 3 bullet points.',
      standard: 'Write a standard-length summary - about one paragraph or 5-7 bullet points.',
      detailed: 'Write a detailed, comprehensive summary covering all important points - multiple paragraphs or 10+ bullet points with sub-points.',
    };

    const prompt = `Summarize this page. ${formatInstr[format]} ${lengthInstr[length]}\n\nTitle: ${resp.title || 'Untitled'}\nURL: ${resp.url || ''}\n\nContent:\n${resp.text.slice(0, 15000)}`;

    setStatus('Summarizing...');

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setStatus(''),
      onDone: (rawText) => {
        setResultRaw(rawText);
        setResultHtml(md(rawText));
      },
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('summary'),
      conversation_id: null,
    }));
  };

  const copyResult = async () => {
    if (!resultRaw) return;
    try {
      await navigator.clipboard.writeText(resultRaw);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  const btnStyle = (active: boolean): React.CSSProperties => ({
    flex: 1,
    padding: '6px',
    background: active ? 'var(--pg2)' : 'var(--s2)',
    border: `1px solid ${active ? 'var(--p)' : 'var(--b1)'}`,
    borderRadius: 'var(--r-md)',
    color: active ? 'var(--pl)' : 'var(--mu)',
    fontSize: '11px',
    cursor: 'pointer',
    fontFamily: 'inherit',
  });

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      {/* Format selector */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
          Format
        </div>
        <div className="flex gap-2">
          {([
            { key: 'bullets' as const, label: 'Key Points' },
            { key: 'paragraph' as const, label: 'Paragraph' },
            { key: 'tldr' as const, label: 'TL;DR' },
          ]).map(f => (
            <button key={f.key} onClick={() => setFormat(f.key)} style={btnStyle(format === f.key)}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Length selector */}
      <div>
        <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
          Length
        </div>
        <div className="flex gap-2">
          {([
            { key: 'brief' as const, label: 'Brief', desc: '~3 sentences' },
            { key: 'standard' as const, label: 'Standard', desc: '1 paragraph' },
            { key: 'detailed' as const, label: 'Detailed', desc: 'Full coverage' },
          ]).map(l => (
            <button
              key={l.key}
              onClick={() => setLength(l.key)}
              style={{
                ...btnStyle(length === l.key),
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 1,
                padding: '6px 4px',
              }}
            >
              <span>{l.label}</span>
              <span style={{ fontSize: '9px', opacity: 0.7 }}>{l.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Action row */}
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
          {activeStream ? '...' : 'Summarize This Page'}
        </button>
      </div>

      {status && (
        <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center' }}>{status}</div>
      )}

      {/* Source page info */}
      {pageInfo && (
        <div
          className="flex items-center gap-2 flex-shrink-0"
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            padding: '7px 10px',
            fontSize: '11px',
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ color: 'var(--tx)', fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {pageInfo.title}
            </div>
            {pageInfo.url && (
              <div style={{ color: 'var(--di)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '10px', marginTop: 1 }}>
                {pageInfo.url}
              </div>
            )}
          </div>
          {pageInfo.url && (
            <a
              href={pageInfo.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'var(--pl)', flexShrink: 0, display: 'flex' }}
              title="Open source page"
            >
              <ExternalLink size={13} />
            </a>
          )}
        </div>
      )}

      {/* Results */}
      {(isStreaming || resultHtml) && (
        <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
          <div
            className="flex-1 overflow-y-auto"
            style={{
              height: '100%',
              background: 'var(--s1)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
              paddingRight: '36px',
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

          {/* Copy button */}
          {resultRaw && !isStreaming && (
            <button
              onClick={copyResult}
              title="Copy summary"
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                color: copied ? 'var(--green, #22c55e)' : 'var(--mu)',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
