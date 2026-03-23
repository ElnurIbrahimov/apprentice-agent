import React, { useEffect, useRef, useState } from 'react';
import { Copy, Check } from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';

// LCS word diff — ported exactly from sidebar.js
function wordDiff(a: string, b: string) {
  const wa = a.split(/(\s+)/), wb = b.split(/(\s+)/);
  const m = wa.length, n = wb.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = wa[i - 1] === wb[j - 1] ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
  const ops: { t: '=' | '+' | '-'; w: string }[] = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && wa[i - 1] === wb[j - 1]) { ops.unshift({ t: '=', w: wa[i - 1] }); i--; j--; }
    else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) { ops.unshift({ t: '+', w: wb[j - 1] }); j--; }
    else { ops.unshift({ t: '-', w: wa[i - 1] }); i--; }
  }
  return ops;
}

function esc(s: string) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderWordDiff(a: string, b: string): string {
  return wordDiff(a, b).map(op => {
    if (op.t === '=') return esc(op.w);
    if (op.t === '+') return `<span class="gr-ins">${esc(op.w)}</span>`;
    return `<span class="gr-del">${esc(op.w)}</span>`;
  }).join('');
}

function renderGrammarResult(original: string, rawText: string): string {
  const sep = rawText.indexOf('---CHANGES---');
  if (sep === -1) return `<p>${esc(rawText)}</p>`;
  const corrected = rawText.slice(0, sep).trim();
  const changesRaw = rawText.slice(sep + 13).trim();
  const diffHtml = renderWordDiff(original.trim(), corrected);
  const changesHtml = changesRaw
    ? `<div style="margin-top:12px;padding-top:10px;border-top:1px solid var(--b1)">
        <div style="font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mu);margin-bottom:6px">Changes</div>
        ${changesRaw.split('\n').filter(l => l.trim()).map(l => `<div style="font-size:12px;color:var(--mu);margin-bottom:3px">${esc(l)}</div>`).join('')}
      </div>`
    : '';
  return `<div>${diffHtml}</div>${changesHtml}`;
}

const MODES = [
  { key: 'grammar', label: 'Grammar' },
  { key: 'style', label: 'Style' },
  { key: 'rewrite', label: 'Rewrite' },
];

export default function GrammarPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [mode, setMode] = useState('grammar');
  const [resultHtml, setResultHtml] = useState('');
  const [correctedText, setCorrectedText] = useState('');
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    };
  }, []);

  const doCheck = () => {
    const text = inputRef.current?.value.trim();
    if (!text) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setError('AURA is offline. Check your connection.'); setTimeout(() => setError(''), 5000); return; }
    if (activeStream) return;

    setResultHtml('');
    const origText = text;

    const prompts: Record<string, string> = {
      grammar: `Fix grammar and spelling. Return the corrected text, then the separator line "---CHANGES---", then each change as "original → corrected" on its own line.\n\nText:\n${text}`,
      style: `Fix grammar, spelling, and improve clarity and style. Return the corrected text, then "---CHANGES---", then each change as "original → corrected".\n\nText:\n${text}`,
      rewrite: `Completely rewrite for maximum clarity and flow. Return the rewritten text, then "---CHANGES---", then a brief summary of what changed.\n\nText:\n${text}`,
    };

    setActiveStream({
      type: 'grammar',
      rawText: '',
      onFirstChunk: () => setResultHtml(''),
      onDone: (rawText) => {
        setResultHtml(renderGrammarResult(origText, rawText));
        const sep = rawText.indexOf('---CHANGES---');
        setCorrectedText(sep === -1 ? rawText.trim() : rawText.slice(0, sep).trim());
      },
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompts[mode], model: getModel('grammar'), conversation_id: null }));
  };

  const handleCopy = () => {
    if (!correctedText) return;
    navigator.clipboard.writeText(correctedText).then(() => {
      setCopied(true);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setCopied(false), 2000);
    }).catch(() => {});
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true && stream.type === 'grammar';

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Mode toggle */}
      <div className="flex flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        {MODES.map(m => (
          <button
            key={m.key}
            onClick={() => setMode(m.key)}
            style={{
              flex: 1,
              padding: '9px',
              background: 'none',
              border: 'none',
              borderBottom: mode === m.key ? '2px solid var(--p)' : '2px solid transparent',
              color: mode === m.key ? 'var(--pl)' : 'var(--mu)',
              fontSize: '12px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontWeight: mode === m.key ? 500 : 400,
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="flex-1 flex flex-col gap-3 p-3 overflow-hidden">
        <textarea
          ref={inputRef}
          placeholder="Paste text to check…"
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doCheck(); } }}
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12.5px',
            padding: '8px 10px',
            resize: 'none',
            height: 120,
            outline: 'none',
            fontFamily: 'inherit',
            flexShrink: 0,
          }}
        />

        <div className="flex items-center justify-between flex-shrink-0">
          <ModelPill featureKey="grammar" />
          <button
            onClick={doCheck}
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
            {activeStream ? '…' : 'Check'}
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 'var(--r-md)',
            padding: '8px 12px',
            fontSize: '12px',
            color: '#ef4444',
            flexShrink: 0,
          }}>
            {error}
          </div>
        )}

        {/* Result */}
        {(isStreaming || resultHtml) && (
          <div
            className="flex-1 overflow-y-auto"
            style={{
              position: 'relative',
              background: 'var(--s1)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
              paddingRight: '36px',
              fontSize: '12.5px',
              lineHeight: 1.65,
            }}
          >
            {isStreaming && !resultHtml ? (
              <div className="dots"><span /><span /><span /></div>
            ) : (
              <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(resultHtml) }} />
            )}
            {resultHtml && !isStreaming && (
              <button
                onClick={handleCopy}
                title="Copy corrected text"
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
    </div>
  );
}
