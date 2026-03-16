import React, { useRef, useState } from 'react';
import { Copy, Download } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import { md } from '../markdown';

export default function ArtifactsPanel() {
  const { getModel } = useStore();
  const [lang, setLang] = useState<'html' | 'svg' | 'markdown'>('html');
  const [view, setView] = useState<'preview' | 'code'>('preview');
  const [status, setStatus] = useState('');
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const generate = async () => {
    const prompt = inputRef.current?.value.trim();
    if (!prompt) return;
    setLoading(true);
    setStatus('Generating…');
    setCode('');
    if (iframeRef.current) iframeRef.current.srcdoc = '';

    const systemNote =
      lang === 'svg'
        ? 'Respond with only the SVG code, no explanation.'
        : lang === 'markdown'
        ? 'Respond with only Markdown, no explanation.'
        : 'Respond with only a complete HTML file including CSS and JS. No explanation, no markdown fences.';

    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `${systemNote}\n\nTask: ${prompt}`,
          stream: false,
          model: getModel('artifacts') || undefined,
        }),
      });
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('⚠ ' + ((d as any).detail || resp.statusText));
        return;
      }
      const data = await resp.json();
      let result = (data.response || data.message || '').trim();
      // Strip markdown code fences
      result = result.replace(/^```[\w\-\.]*\r?\n?/, '').replace(/\r?\n?```[\w\-\.]*\s*$/, '').trim();
      setCode(result);

      if (lang === 'svg') {
        iframeRef.current!.srcdoc = `<html><body style="margin:0;display:flex;align-items:center;justify-content:center;min-height:100vh;background:#fff">${result}</body></html>`;
      } else if (lang === 'markdown') {
        iframeRef.current!.srcdoc = `<html><head><style>body{font-family:system-ui,sans-serif;padding:16px;line-height:1.6;max-width:700px;margin:0 auto}</style></head><body>${md(result)}</body></html>`;
      } else {
        iframeRef.current!.srcdoc = result;
      }

      setStatus('');
      setView('preview');
    } catch (err: any) {
      setStatus('⚠ ' + (err.message || 'Request failed'));
    } finally {
      setLoading(false);
    }
  };

  const copyCode = () => {
    if (!code) return;
    navigator.clipboard.writeText(code).then(() => {
      setStatus('✓ Copied!');
      setTimeout(() => setStatus(''), 1500);
    });
  };

  const download = () => {
    if (!code) return;
    const ext = lang === 'html' ? '.html' : lang === 'svg' ? '.svg' : '.md';
    const blob = new Blob([code], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `artifact${ext}`;
    a.click();
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Controls */}
      <div className="flex items-center gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <select
          value={lang}
          onChange={e => setLang(e.target.value as any)}
          style={{
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-sm)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '4px 8px',
            fontFamily: 'inherit',
          }}
        >
          <option value="html">HTML</option>
          <option value="svg">SVG</option>
          <option value="markdown">Markdown</option>
        </select>
        <div className="flex-1" />
        <ModelPill featureKey="artifacts" />
      </div>

      <div className="flex gap-2 px-3 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <textarea
          ref={inputRef}
          placeholder="Describe what to generate… (Ctrl+Enter)"
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); generate(); } }}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '7px 10px',
            resize: 'none',
            height: 60,
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={generate}
          disabled={loading}
          style={{
            background: loading ? 'var(--s3)' : 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '8px 14px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
            alignSelf: 'flex-end',
          }}
        >
          {loading ? '…' : 'Generate'}
        </button>
      </div>

      {status && (
        <div style={{ padding: '6px 12px', fontSize: '12px', color: status.startsWith('⚠') ? 'var(--rd)' : status.startsWith('✓') ? 'var(--gr)' : 'var(--mu)' }}>
          {status}
        </div>
      )}

      {/* View toggle */}
      {code && (
        <div className="flex flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
          {(['preview', 'code'] as const).map(v => (
            <button
              key={v}
              onClick={() => setView(v)}
              style={{
                flex: 1,
                padding: '7px',
                background: 'none',
                border: 'none',
                borderBottom: view === v ? '2px solid var(--p)' : '2px solid transparent',
                color: view === v ? 'var(--pl)' : 'var(--mu)',
                fontSize: '12px',
                cursor: 'pointer',
                fontFamily: 'inherit',
                textTransform: 'capitalize',
              }}
            >
              {v}
            </button>
          ))}
        </div>
      )}

      {/* Content area */}
      <div className="flex-1 relative overflow-hidden">
        {/* Preview iframe */}
        <iframe
          ref={iframeRef}
          sandbox="allow-scripts"
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            border: 'none',
            display: code && view === 'preview' ? 'block' : 'none',
          }}
        />
        {/* Code view */}
        {code && view === 'code' && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              overflow: 'auto',
              padding: '12px',
              fontFamily: 'monospace',
              fontSize: '11.5px',
              color: 'var(--tx)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-all',
              lineHeight: 1.5,
            }}
          >
            {code}
          </div>
        )}
        {!code && !loading && (
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--mu)',
              fontSize: '12px',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <span style={{ fontSize: '32px' }}>⌨</span>
            Describe what to generate above
          </div>
        )}
      </div>

      {/* Footer actions */}
      {code && (
        <div className="flex gap-2 p-2 flex-shrink-0" style={{ borderTop: '1px solid var(--b1)' }}>
          <button
            onClick={copyCode}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--mu)',
              padding: '6px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            <Copy size={13} /> Copy Code
          </button>
          <button
            onClick={download}
            style={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              color: 'var(--mu)',
              padding: '6px',
              cursor: 'pointer',
              fontSize: '12px',
              fontFamily: 'inherit',
            }}
          >
            <Download size={13} /> Download
          </button>
        </div>
      )}
    </div>
  );
}
