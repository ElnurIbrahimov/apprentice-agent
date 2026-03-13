import React, { useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import { md } from '../markdown';

export default function ResearchPanel() {
  const { getModel } = useStore();
  const [depth, setDepth] = useState<'quick' | 'standard' | 'deep'>('standard');
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');
  const [sources, setSources] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const doResearch = async () => {
    const query = inputRef.current?.value.trim();
    if (!query) return;
    setLoading(true);
    setStatus('Searching the web…');
    setResultHtml('');
    setSources([]);

    try {
      const resp = await fetch(`${HTTP}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, depth, model: getModel('research') }),
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('⚠ ' + ((d as any).detail || resp.statusText));
        return;
      }

      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop()!;
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.status && ev.status !== 'done') setStatus(ev.message || ev.status);
            if (ev.status === 'done') {
              setStatus(`Done — ${ev.sources?.length || 0} sources`);
              setResultHtml(md(ev.report || ''));
              setSources(ev.sources || []);
            }
          } catch {}
        }
      }
    } catch (err: any) {
      setStatus('⚠ ' + (err.message || 'Request failed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Input */}
      <div className="flex gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <input
          ref={inputRef}
          placeholder="Research topic…"
          autoFocus
          onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doResearch(); } }}
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '7px 10px',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
      </div>

      {/* Depth + model */}
      <div className="flex items-center gap-2 px-3 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <div className="flex gap-1">
          {(['quick', 'standard', 'deep'] as const).map(d => (
            <button
              key={d}
              onClick={() => setDepth(d)}
              style={{
                padding: '4px 10px',
                background: depth === d ? 'var(--pg2)' : 'var(--s2)',
                border: `1px solid ${depth === d ? 'var(--p)' : 'var(--b1)'}`,
                borderRadius: 'var(--r-pill)',
                color: depth === d ? 'var(--pl)' : 'var(--mu)',
                fontSize: '11px',
                cursor: 'pointer',
                fontFamily: 'inherit',
                textTransform: 'capitalize',
              }}
            >
              {d}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        <ModelPill featureKey="research" />
        <button
          onClick={doResearch}
          disabled={loading}
          style={{
            background: loading ? 'var(--s3)' : 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '5px 14px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {loading ? '…' : 'Research'}
        </button>
      </div>

      {/* Results */}
      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        {status && (
          <div style={{ color: status.startsWith('⚠') ? 'var(--rd)' : 'var(--mu)', fontSize: '12px' }}>
            {loading && !status.startsWith('⚠') && <span style={{ marginRight: 6 }}>🔍</span>}
            {status}
          </div>
        )}

        {resultHtml && (
          <div
            className="md-body"
            style={{ fontSize: '12.5px', lineHeight: 1.65 }}
            dangerouslySetInnerHTML={{ __html: resultHtml }}
          />
        )}

        {sources.length > 0 && (
          <div className="flex flex-col gap-2">
            <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)' }}>
              Sources
            </div>
            {sources.slice(0, 6).map((s: any) => (
              <div
                key={s.index}
                style={{
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  padding: '8px 10px',
                }}
              >
                <a
                  href={s.url}
                  target="_blank"
                  rel="noopener"
                  style={{ color: 'var(--pl)', fontSize: '12px', fontWeight: 500, textDecoration: 'none' }}
                >
                  [{s.index}] {s.title || s.domain}
                </a>
                {s.snippet && (
                  <div style={{ fontSize: '11px', color: 'var(--mu)', marginTop: 3 }}>{s.snippet}</div>
                )}
              </div>
            ))}
          </div>
        )}

        {!loading && !status && !resultHtml && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            Enter a topic to start multi-source research
          </div>
        )}
      </div>
    </div>
  );
}
