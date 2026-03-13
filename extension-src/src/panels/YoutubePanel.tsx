import React, { useState, useEffect, useRef } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP } from '../api';
import { md } from '../markdown';

export default function YoutubePanel() {
  const { getModel, activePanel } = useStore();
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [snippetOpen, setSnippetOpen] = useState(false);
  const [autoUrl, setAutoUrl] = useState('');
  const [autoTitle, setAutoTitle] = useState('');

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      setAutoUrl(detail.url || '');
      setAutoTitle(detail.title || detail.url || '');
    };
    window.addEventListener('yt-detected', handler);
    // Check if already detected
    if ((window as any).__ytAutoUrl) {
      setAutoUrl((window as any).__ytAutoUrl);
      setAutoTitle((window as any).__ytAutoTitle || (window as any).__ytAutoUrl);
    }
    return () => window.removeEventListener('yt-detected', handler);
  }, []);

  useEffect(() => {
    if (activePanel === 'youtube' && autoUrl) setUrl(autoUrl);
  }, [activePanel, autoUrl]);

  const summarize = async (urlToUse?: string) => {
    const target = (urlToUse || url).trim();
    if (!target) return;
    setLoading(true);
    setStatus('Fetching transcript…');
    setResult(null);
    try {
      const resp = await fetch(`${HTTP}/api/youtube/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: target }),
        signal: AbortSignal.timeout(90000),
      });
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('⚠ ' + ((d as any).detail || `Error ${resp.status}`));
        return;
      }
      const data = await resp.json();
      setResult(data);
      setStatus('');
    } catch (err: any) {
      setStatus('⚠ ' + (err.name === 'TimeoutError' ? 'Request timed out.' : err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Auto-detected banner */}
      {autoUrl && (
        <div
          className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
          style={{ background: 'rgba(239,68,68,0.1)', borderBottom: '1px solid rgba(239,68,68,0.2)' }}
        >
          <span style={{ fontSize: '12px', color: 'var(--tx)', flex: 1 }}>
            ▶ {autoTitle}
          </span>
          <button
            onClick={() => summarize(autoUrl)}
            disabled={loading}
            style={{
              background: '#ef4444',
              border: 'none',
              borderRadius: 'var(--r-sm)',
              color: 'white',
              fontSize: '11px',
              padding: '4px 10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Summarize
          </button>
        </div>
      )}

      {/* URL input */}
      <div className="flex gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <input
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') summarize(); }}
          placeholder="YouTube URL…"
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
        <button
          onClick={() => summarize()}
          disabled={loading}
          style={{
            background: '#ef4444',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '7px 14px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {loading ? '…' : 'Summarize'}
        </button>
      </div>

      <div className="flex items-center gap-2 px-3 py-1.5 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <ModelPill featureKey="youtube" />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {status && (
          <div style={{ color: status.startsWith('⚠') ? 'var(--rd)' : 'var(--mu)', fontSize: '12px', marginBottom: 12 }}>
            {loading && !status.startsWith('⚠') && <span className="inline-flex gap-1 mr-2"><span className="dots"><span /><span /><span /></span></span>}
            {status}
          </div>
        )}

        {result && (
          <div className="flex flex-col gap-3">
            <div style={{ fontWeight: 600, fontSize: '13.5px', color: 'var(--tx)' }}>{result.title || 'Untitled'}</div>
            <div className="flex gap-3" style={{ fontSize: '11px', color: 'var(--mu)' }}>
              {result.channel && <span>▶ {result.channel}</span>}
              {result.duration && <span>⏱ {result.duration}</span>}
            </div>

            <div
              className="md-body"
              style={{ fontSize: '12.5px', lineHeight: 1.65 }}
              dangerouslySetInnerHTML={{ __html: md(result.summary || 'No summary available.') }}
            />

            {result.key_points?.length > 0 && (
              <div>
                <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
                  Key Points
                </div>
                <ul style={{ paddingLeft: 16, fontSize: '12px', color: 'var(--tx)' }}>
                  {result.key_points.map((pt: string, i: number) => (
                    <li key={i} style={{ marginBottom: 4 }}>{pt}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.transcript_snippet && (
              <div>
                <button
                  onClick={() => setSnippetOpen(!snippetOpen)}
                  style={{
                    background: 'none',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: 'var(--mu)',
                    fontSize: '11px',
                    padding: '4px 10px',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  {snippetOpen ? '▼' : '▶'} Transcript snippet
                </button>
                {snippetOpen && (
                  <div style={{ marginTop: 8, fontSize: '11.5px', color: 'var(--mu)', fontStyle: 'italic', lineHeight: 1.6 }}>
                    {result.transcript_snippet}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {!loading && !status && !result && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            Paste a YouTube URL to summarize
          </div>
        )}
      </div>
    </div>
  );
}
