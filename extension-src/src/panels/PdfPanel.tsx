import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, File, ChevronLeft, ChevronRight, Languages, FileText, Copy, Check } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, apiFetch } from '../api';
import { md } from '../markdown';

const TRANSLATE_LANGS = [
  'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
  'Russian', 'Chinese (Simplified)', 'Japanese', 'Korean', 'Arabic', 'Hindi', 'Turkish',
  'Dutch', 'Polish', 'Swedish', 'Azerbaijani',
];

interface PdfCtx {
  text: string;
  page_count: number;
  word_count: number;
  filename?: string;
  file_size?: number;
  title?: string;
  pages?: string[]; // text per page if available
}

const PAGE_SIZE = 3000; // chars per "page" if backend doesn't split

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function PdfPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();
  const [pdfCtx, setPdfCtx] = useState<PdfCtx | null>(null);
  const [status, setStatus] = useState('');
  const [resultHtml, setResultHtml] = useState('');
  const [resultRaw, setResultRaw] = useState('');
  const [autoUrl, setAutoUrl] = useState('');
  const [autoTitle, setAutoTitle] = useState('');
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);
  const [translateLang, setTranslateLang] = useState('');
  const [copied, setCopied] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      setAutoUrl(detail.url);
      setAutoTitle(detail.title || detail.url);
    };
    window.addEventListener('pdf-detected', handler);
    return () => window.removeEventListener('pdf-detected', handler);
  }, []);

  // Split full text into virtual pages
  const getPages = useCallback((): string[] => {
    if (!pdfCtx) return [];
    if (pdfCtx.pages && pdfCtx.pages.length > 0) return pdfCtx.pages;
    // Virtual pagination by character count
    const text = pdfCtx.text;
    const pages: string[] = [];
    for (let i = 0; i < text.length; i += PAGE_SIZE) {
      pages.push(text.slice(i, i + PAGE_SIZE));
    }
    return pages.length > 0 ? pages : [text];
  }, [pdfCtx]);

  const totalPages = pdfCtx ? getPages().length : 0;

  const uploadPdf = async (file: globalThis.File) => {
    setStatus('Extracting text...');
    setPdfCtx(null);
    setResultHtml('');
    setResultRaw('');
    setCurrentPage(0);
    setUploading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const data = await apiFetch(`${HTTP}/api/pdf/extract`, { method: 'POST', body: form });
      setPdfCtx({
        text: data.text,
        page_count: data.page_count,
        word_count: data.word_count,
        filename: file.name,
        file_size: file.size,
        title: data.title || file.name,
        pages: data.pages || undefined,
      });
      setStatus('');
    } catch (err: any) {
      setStatus('Error: ' + err.message);
    }
    setUploading(false);
  };

  const loadPdfUrl = async (url: string) => {
    setStatus('Loading PDF...');
    setPdfCtx(null);
    setResultHtml('');
    setResultRaw('');
    setCurrentPage(0);
    setUploading(true);
    try {
      const data = await apiFetch(`${HTTP}/api/pdf/extract_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      });
      setPdfCtx({
        text: data.text,
        page_count: data.page_count,
        word_count: data.word_count,
        filename: url,
        title: data.title || url,
        pages: data.pages || undefined,
      });
      setStatus('');
    } catch (err: any) {
      setStatus('Error: ' + err.message);
    }
    setUploading(false);
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        uploadPdf(file);
      } else {
        setStatus('Please drop a PDF file.');
      }
    }
  };

  const sendPrompt = (prompt: string) => {
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
      setStatus('AURA is offline.');
      return;
    }
    if (activeStream) return;

    setResultHtml('');
    setResultRaw('');

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setResultHtml(''),
      onDone: (rawText) => {
        setResultRaw(rawText);
        setResultHtml(md(rawText));
      },
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('pdf'), conversation_id: null }));
  };

  const ask = () => {
    const question = inputRef.current?.value.trim();
    if (!question || !pdfCtx) return;
    const prompt = `Based on this PDF content, answer: ${question}\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;
    sendPrompt(prompt);
  };

  const summarizePdf = () => {
    if (!pdfCtx) return;
    const prompt = `Provide a comprehensive summary of this PDF document. Include the main topics, key findings, and important details. Format with clear headings and bullet points.\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;
    sendPrompt(prompt);
  };

  const translatePdf = () => {
    if (!pdfCtx || !translateLang) return;
    const prompt = `Translate the following PDF content to ${translateLang}. Maintain the original formatting, paragraph breaks, and structure as much as possible. Output only the translation.\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;
    sendPrompt(prompt);
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

  const pages = pdfCtx ? getPages() : [];

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Auto-detected */}
      {autoUrl && !pdfCtx && (
        <div
          className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
          style={{ background: 'rgba(124,58,237,0.1)', borderBottom: '1px solid var(--b1)' }}
        >
          <span style={{ fontSize: '12px', color: 'var(--tx)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            PDF detected: {autoTitle}
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
        {/* Upload zone */}
        {!pdfCtx && (
          <div>
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) uploadPdf(f); }}
            />
            <div
              ref={dropRef}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                background: dragging ? 'rgba(124,58,237,0.08)' : 'var(--s2)',
                border: `2px dashed ${dragging ? 'var(--p)' : 'var(--b2)'}`,
                borderRadius: 'var(--r-lg)',
                color: 'var(--mu)',
                padding: '28px 24px',
                cursor: 'pointer',
                fontSize: '13px',
                fontFamily: 'inherit',
                flexDirection: 'column',
                transition: 'border-color 0.15s, background 0.15s',
              }}
            >
              {uploading ? (
                <>
                  <div className="dots" style={{ marginBottom: 4 }}><span /><span /><span /></div>
                  <span>Extracting text...</span>
                </>
              ) : (
                <>
                  <Upload size={22} style={{ color: 'var(--pl)' }} />
                  <span>Click or drag & drop PDF here</span>
                  <span style={{ fontSize: '11px', color: 'var(--di)' }}>Supports .pdf files</span>
                </>
              )}
            </div>
            {status && (
              <div style={{ color: status.startsWith('Error') ? 'var(--red, #ef4444)' : 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 8 }}>
                {status}
              </div>
            )}
          </div>
        )}

        {/* PDF loaded */}
        {pdfCtx && (
          <>
            {/* Metadata card */}
            <div
              className="flex items-center gap-2 flex-shrink-0"
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: '10px',
              }}
            >
              <File size={18} style={{ color: 'var(--pl)', flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {pdfCtx.title || pdfCtx.filename || 'PDF document'}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--mu)', display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 2 }}>
                  <span>{pdfCtx.page_count} page{pdfCtx.page_count !== 1 ? 's' : ''}</span>
                  <span>{pdfCtx.word_count?.toLocaleString()} words</span>
                  {pdfCtx.file_size != null && <span>{formatFileSize(pdfCtx.file_size)}</span>}
                </div>
              </div>
              <button
                onClick={() => { setPdfCtx(null); setStatus(''); setResultHtml(''); setResultRaw(''); setCurrentPage(0); setTranslateLang(''); }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', fontSize: '14px', fontFamily: 'inherit', padding: '2px 6px' }}
                title="Close PDF"
              >
                ✕
              </button>
            </div>

            {/* Page navigation */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between flex-shrink-0" style={{ fontSize: '11px' }}>
                <button
                  onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                  disabled={currentPage === 0}
                  style={{
                    background: 'none',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: currentPage === 0 ? 'var(--di)' : 'var(--mu)',
                    cursor: currentPage === 0 ? 'not-allowed' : 'pointer',
                    padding: '3px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    fontFamily: 'inherit',
                  }}
                >
                  <ChevronLeft size={12} /> Prev
                </button>
                <span style={{ color: 'var(--mu)' }}>
                  Page {currentPage + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages - 1, p + 1))}
                  disabled={currentPage >= totalPages - 1}
                  style={{
                    background: 'none',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: currentPage >= totalPages - 1 ? 'var(--di)' : 'var(--mu)',
                    cursor: currentPage >= totalPages - 1 ? 'not-allowed' : 'pointer',
                    padding: '3px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    fontFamily: 'inherit',
                  }}
                >
                  Next <ChevronRight size={12} />
                </button>
              </div>
            )}

            {/* Page content preview (collapsible) */}
            {pages[currentPage] && (
              <div
                style={{
                  background: 'var(--s1)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  padding: '8px 10px',
                  fontSize: '11px',
                  color: 'var(--mu)',
                  maxHeight: 80,
                  overflowY: 'auto',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  flexShrink: 0,
                }}
              >
                {pages[currentPage]}
              </div>
            )}

            {/* Quick action buttons */}
            <div className="flex gap-2 flex-shrink-0">
              <button
                onClick={summarizePdf}
                disabled={!!activeStream}
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 5,
                  background: activeStream ? 'var(--s3)' : 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: activeStream ? 'var(--di)' : 'var(--pl)',
                  padding: '7px 8px',
                  cursor: activeStream ? 'not-allowed' : 'pointer',
                  fontSize: '11px',
                  fontFamily: 'inherit',
                }}
              >
                <FileText size={13} />
                Summarize
              </button>

              {/* Translate PDF dropdown */}
              <div style={{ flex: 1, display: 'flex', gap: 0 }}>
                <select
                  value={translateLang}
                  onChange={e => setTranslateLang(e.target.value)}
                  style={{
                    flex: 1,
                    background: 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderRight: 'none',
                    borderRadius: 'var(--r-md) 0 0 var(--r-md)',
                    color: 'var(--tx)',
                    fontSize: '11px',
                    padding: '5px 6px',
                    fontFamily: 'inherit',
                  }}
                >
                  <option value="">Translate to...</option>
                  {TRANSLATE_LANGS.map(l => <option key={l} value={l}>{l}</option>)}
                </select>
                <button
                  onClick={translatePdf}
                  disabled={!!activeStream || !translateLang}
                  style={{
                    background: (!translateLang || activeStream) ? 'var(--s3)' : 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderRadius: '0 var(--r-md) var(--r-md) 0',
                    color: (!translateLang || activeStream) ? 'var(--di)' : 'var(--pl)',
                    cursor: (!translateLang || activeStream) ? 'not-allowed' : 'pointer',
                    padding: '5px 8px',
                    display: 'flex',
                    alignItems: 'center',
                    fontFamily: 'inherit',
                  }}
                >
                  <Languages size={13} />
                </button>
              </div>
            </div>

            {/* Ask question */}
            <div className="flex gap-2 flex-shrink-0">
              <textarea
                ref={inputRef}
                placeholder="Ask a question about the PDF..."
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
                  height: 60,
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
                {activeStream ? '...' : 'Ask'}
              </button>
            </div>

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

                {/* Copy result */}
                {resultRaw && !isStreaming && (
                  <button
                    onClick={copyResult}
                    title="Copy result"
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
          </>
        )}
      </div>
    </div>
  );
}
