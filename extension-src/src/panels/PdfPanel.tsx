import React, { useState, useRef, useEffect } from 'react';
import { Upload, File } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, apiFetch } from '../api';
import { md } from '../markdown';

interface PdfCtx {
  text: string;
  page_count: number;
  word_count: number;
  filename?: string;
}

export default function PdfPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [pdfCtx, setPdfCtx] = useState<PdfCtx | null>(null);
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');
  const [autoUrl, setAutoUrl] = useState('');
  const [autoTitle, setAutoTitle] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      setAutoUrl(detail.url);
      setAutoTitle(detail.title || detail.url);
    };
    window.addEventListener('pdf-detected', handler);
    return () => window.removeEventListener('pdf-detected', handler);
  }, []);

  const uploadPdf = async (file: File) => {
    setStatus('Extracting text…');
    setPdfCtx(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const data = await apiFetch(`${HTTP}/api/pdf/extract`, { method: 'POST', body: form });
      setPdfCtx({ text: data.text, page_count: data.page_count, word_count: data.word_count, filename: file.name });
      setStatus(`✓ ${data.page_count} pages · ${data.word_count} words`);
    } catch (err: any) {
      setStatus('⚠ ' + err.message);
    }
  };

  const loadPdfUrl = async (url: string) => {
    setStatus('Loading PDF…');
    setPdfCtx(null);
    try {
      const data = await apiFetch(`${HTTP}/api/pdf/extract_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      setPdfCtx({ text: data.text, page_count: data.page_count, word_count: data.word_count, filename: url });
      setStatus(`✓ ${data.page_count} pages · ${data.word_count} words`);
    } catch (err: any) {
      setStatus('⚠ ' + err.message);
    }
  };

  const ask = () => {
    const question = inputRef.current?.value.trim();
    if (!question || !pdfCtx) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setResultHtml('');
    const prompt = `Based on this PDF content, answer: ${question}\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setResultHtml(''),
      onDone: (rawText) => setResultHtml(md(rawText)),
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('pdf'), conversation_id: null }));
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Auto-detected */}
      {autoUrl && !pdfCtx && (
        <div
          className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
          style={{ background: 'rgba(124,58,237,0.1)', borderBottom: '1px solid var(--b1)' }}
        >
          <span style={{ fontSize: '12px', color: 'var(--tx)', flex: 1 }}>
            📄 {autoTitle}
          </span>
          <button
            onClick={() => loadPdfUrl(autoUrl)}
            style={{
              background: 'var(--p)',
              border: 'none',
              borderRadius: 'var(--r-sm)',
              color: 'white',
              fontSize: '11px',
              padding: '4px 10px',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            Load
          </button>
        </div>
      )}

      <div className="flex-1 flex flex-col gap-3 p-3 overflow-hidden">
        {/* Upload */}
        {!pdfCtx && (
          <div>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) uploadPdf(f); }}
            />
            <button
              onClick={() => fileRef.current?.click()}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                background: 'var(--s2)',
                border: '2px dashed var(--b2)',
                borderRadius: 'var(--r-lg)',
                color: 'var(--mu)',
                padding: '24px',
                cursor: 'pointer',
                fontSize: '13px',
                fontFamily: 'inherit',
                flexDirection: 'column',
              }}
            >
              <Upload size={22} style={{ color: 'var(--pl)' }} />
              <span>Click to upload PDF</span>
            </button>
            {status && (
              <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 8 }}>{status}</div>
            )}
          </div>
        )}

        {/* PDF loaded */}
        {pdfCtx && (
          <>
            <div
              className="flex items-center gap-2 flex-shrink-0"
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: '10px',
              }}
            >
              <File size={16} style={{ color: 'var(--pl)', flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {pdfCtx.filename || 'PDF document'}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--mu)' }}>
                  {pdfCtx.page_count} pages · {pdfCtx.word_count?.toLocaleString()} words
                </div>
              </div>
              <button
                onClick={() => { setPdfCtx(null); setStatus(''); setResultHtml(''); }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', fontSize: '11px', fontFamily: 'inherit' }}
              >
                ✕
              </button>
            </div>

            <div className="flex gap-2 flex-shrink-0">
              <textarea
                ref={inputRef}
                placeholder="Ask a question about the PDF…"
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } }}
                style={{
                  flex: 1,
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--tx)',
                  fontSize: '12.5px',
                  padding: '8px 10px',
                  resize: 'none',
                  height: 70,
                  outline: 'none',
                  fontFamily: 'inherit',
                }}
              />
            </div>

            <div className="flex items-center justify-between flex-shrink-0">
              <ModelPill featureKey="pdf" />
              <button
                onClick={ask}
                disabled={!!activeStream || !pdfCtx}
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
                {activeStream ? '…' : 'Ask'}
              </button>
            </div>

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
          </>
        )}
      </div>
    </div>
  );
}
