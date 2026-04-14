import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Upload, File, ChevronLeft, ChevronRight, Languages, FileText, Copy, Check, Download, X } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, apiFetch } from '../api';
import { md } from '../markdown';

const TRANSLATE_LANGS = [
  'English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese',
  'Russian', 'Chinese (Simplified)', 'Chinese (Traditional)', 'Japanese', 'Korean',
  'Arabic', 'Hindi', 'Turkish', 'Vietnamese', 'Thai', 'Indonesian',
  'Polish', 'Dutch', 'Swedish', 'Norwegian', 'Danish', 'Finnish',
  'Greek', 'Czech', 'Romanian', 'Hungarian', 'Ukrainian', 'Hebrew', 'Bengali',
  'Azerbaijani', 'Persian', 'Malay', 'Filipino', 'Swahili',
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

interface TranslationParagraph {
  original: string;
  translated: string;
}

const PAGE_SIZE = 3000; // chars per "page" if backend doesn't split
const BATCH_CHAR_LIMIT = 3000; // chars per translation batch

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

/** Split text into paragraphs, preserving structure */
function splitParagraphs(text: string): string[] {
  // Split on double newlines or single newlines with enough gap
  return text
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p.length > 0);
}

/** Group paragraphs into batches under the char limit */
function batchParagraphs(paragraphs: string[], limit: number): string[][] {
  const batches: string[][] = [];
  let current: string[] = [];
  let currentLen = 0;

  for (const p of paragraphs) {
    if (currentLen + p.length > limit && current.length > 0) {
      batches.push(current);
      current = [];
      currentLen = 0;
    }
    current.push(p);
    currentLen += p.length;
  }
  if (current.length > 0) batches.push(current);
  return batches;
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
  const [truncatedLen, setTruncatedLen] = useState(0); // original char count when truncated

  // Translation state
  const [translateMode, setTranslateMode] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [translateProgress, setTranslateProgress] = useState('');
  const [translatedParagraphs, setTranslatedParagraphs] = useState<TranslationParagraph[]>([]);
  const [translateError, setTranslateError] = useState('');
  const translateAbortRef = useRef(false);

  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      translateAbortRef.current = true;
    };
  }, []);

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
    resetTranslation();
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
    resetTranslation();
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
      type: 'chat',
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
    setTruncatedLen(pdfCtx.text.length > 20000 ? pdfCtx.text.length : 0);
    const prompt = `Based on this PDF content, answer: ${question}\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;
    sendPrompt(prompt);
  };

  const summarizePdf = () => {
    if (!pdfCtx) return;
    setTruncatedLen(pdfCtx.text.length > 20000 ? pdfCtx.text.length : 0);
    const prompt = `Provide a comprehensive summary of this PDF document. Include the main topics, key findings, and important details. Format with clear headings and bullet points.\n\nPDF Content:\n${pdfCtx.text.slice(0, 20000)}`;
    sendPrompt(prompt);
  };

  /* ─── Enhanced PDF Translation ─── */

  const resetTranslation = () => {
    setTranslateMode(false);
    setTranslating(false);
    setTranslateProgress('');
    setTranslatedParagraphs([]);
    setTranslateError('');
    translateAbortRef.current = false;
  };

  /** Translate a single batch of paragraphs via /api/chat (REST, not WS) */
  const translateBatch = async (paragraphs: string[], lang: string): Promise<string> => {
    const text = paragraphs.join('\n\n');
    const prompt = `Translate the following text to ${lang}. Maintain paragraph breaks exactly as they are. Output ONLY the translation, nothing else.\n\n${text}`;

    const resp = await apiFetch(`${HTTP}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: prompt,
        model: getModel('pdf') || null,
        conversation_id: null,
      }),
    });
    return resp.response || resp.text || resp.message || '';
  };

  /** Start the full PDF translation flow */
  const startTranslation = async () => {
    if (!pdfCtx || !translateLang) return;

    setTranslateMode(true);
    setTranslating(true);
    setTranslateError('');
    setTranslatedParagraphs([]);
    translateAbortRef.current = false;

    const paragraphs = splitParagraphs(pdfCtx.text);
    if (paragraphs.length === 0) {
      setTranslateError('No text to translate.');
      setTranslating(false);
      return;
    }

    const batches = batchParagraphs(paragraphs, BATCH_CHAR_LIMIT);
    const results: TranslationParagraph[] = [];
    let batchIdx = 0;

    for (const batch of batches) {
      if (translateAbortRef.current) break;

      batchIdx++;
      setTranslateProgress(`Translating batch ${batchIdx}/${batches.length}...`);

      try {
        const translated = await translateBatch(batch, translateLang);
        // Split translated text back into paragraphs to align with originals
        const translatedParts = translated.split(/\n{2,}/).map(p => p.trim()).filter(p => p.length > 0);

        // Align: best effort — match 1:1, or merge remainder
        for (let i = 0; i < batch.length; i++) {
          results.push({
            original: batch[i],
            translated: translatedParts[i] || (i === batch.length - 1 ? translatedParts.slice(i).join('\n\n') : ''),
          });
        }
        // If translated has more paragraphs than original batch, append extras to last
        if (translatedParts.length > batch.length) {
          const lastIdx = results.length - 1;
          results[lastIdx].translated += '\n\n' + translatedParts.slice(batch.length).join('\n\n');
        }

        setTranslatedParagraphs([...results]);
      } catch (err: any) {
        if (translateAbortRef.current) break;
        setTranslateError(`Error on batch ${batchIdx}: ${err.message}`);
        break;
      }
    }

    setTranslating(false);
    if (!translateAbortRef.current && !results.length) {
      setTranslateError('Translation returned no results.');
    } else if (!translateAbortRef.current) {
      setTranslateProgress(`Done — ${results.length} paragraph${results.length !== 1 ? 's' : ''} translated`);
    }
  };

  const cancelTranslation = () => {
    translateAbortRef.current = true;
    setTranslating(false);
    setTranslateProgress('Cancelled');
  };

  const getFullTranslatedText = (): string => {
    return translatedParagraphs.map(p => p.translated).join('\n\n');
  };

  const copyTranslated = async () => {
    const text = getFullTranslatedText();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  const downloadTranslated = () => {
    const text = getFullTranslatedText();
    if (!text) return;
    const filename = (pdfCtx?.filename || 'document').replace(/\.pdf$/i, '') + `_${translateLang}.txt`;
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyResult = async () => {
    if (!resultRaw) return;
    try {
      await navigator.clipboard.writeText(resultRaw);
      setCopied(true);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setCopied(false), 1500);
    } catch { /* ignore */ }
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  const pages = pdfCtx ? getPages() : [];

  /* ─── Render ─── */

  // Translation view
  if (translateMode && pdfCtx) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div
          className="flex items-center gap-2 px-3 py-2 flex-shrink-0"
          style={{ borderBottom: '1px solid var(--b1)', background: 'rgba(124,58,237,0.06)' }}
        >
          <Languages size={14} style={{ color: 'var(--pl)' }} />
          <span style={{ flex: 1, fontSize: '12px', color: 'var(--tx)', fontWeight: 500 }}>
            PDF Translation — {translateLang}
          </span>
          <button
            onClick={resetTranslation}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: '2px', display: 'flex', alignItems: 'center' }}
            title="Close translation"
          >
            <X size={14} />
          </button>
        </div>

        {/* Progress bar */}
        {translating && (
          <div className="flex items-center gap-2 px-3 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
            <div className="dots" style={{ flexShrink: 0 }}><span /><span /><span /></div>
            <span style={{ fontSize: '11px', color: 'var(--mu)', flex: 1 }}>{translateProgress}</span>
            <button
              onClick={cancelTranslation}
              style={{
                background: 'none',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                color: 'var(--rd, #ef4444)',
                fontSize: '10px',
                padding: '2px 8px',
                cursor: 'pointer',
                fontFamily: 'inherit',
              }}
            >
              Cancel
            </button>
          </div>
        )}

        {/* Error */}
        {translateError && (
          <div style={{ padding: '8px 12px', fontSize: '12px', color: 'var(--rd, #ef4444)', background: 'rgba(239, 68, 68, 0.08)', borderBottom: '1px solid var(--b1)' }}>
            {translateError}
          </div>
        )}

        {/* Done bar */}
        {!translating && translatedParagraphs.length > 0 && (
          <div className="flex items-center gap-2 px-3 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
            <span style={{ fontSize: '11px', color: 'var(--gr, #10b981)', flex: 1 }}>{translateProgress}</span>
            <button
              onClick={copyTranslated}
              title="Copy translated text"
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                color: copied ? 'var(--gr, #22c55e)' : 'var(--mu)',
                cursor: 'pointer',
                padding: '3px 8px',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                fontSize: '11px',
                fontFamily: 'inherit',
              }}
            >
              {copied ? <Check size={11} /> : <Copy size={11} />}
              Copy
            </button>
            <button
              onClick={downloadTranslated}
              title="Download as .txt"
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                color: 'var(--mu)',
                cursor: 'pointer',
                padding: '3px 8px',
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                fontSize: '11px',
                fontFamily: 'inherit',
              }}
            >
              <Download size={11} />
              .txt
            </button>
          </div>
        )}

        {/* Side-by-side paragraphs */}
        <div className="flex-1 overflow-y-auto" style={{ padding: '8px' }}>
          {translatedParagraphs.length === 0 && !translating && !translateError && (
            <div style={{ fontSize: '12px', color: 'var(--di)', textAlign: 'center', padding: '24px' }}>
              Starting translation...
            </div>
          )}

          {translatedParagraphs.map((p, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                gap: 8,
                marginBottom: 8,
                borderRadius: 'var(--r-md)',
                border: '1px solid var(--b1)',
                overflow: 'hidden',
              }}
            >
              {/* Original */}
              <div style={{
                flex: 1,
                padding: '8px 10px',
                fontSize: '11.5px',
                lineHeight: 1.6,
                color: 'var(--mu)',
                background: 'var(--s1)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                borderRight: '1px solid var(--b1)',
              }}>
                {p.original}
              </div>
              {/* Translated */}
              <div style={{
                flex: 1,
                padding: '8px 10px',
                fontSize: '11.5px',
                lineHeight: 1.6,
                color: 'var(--tx)',
                background: 'rgba(124, 58, 237, 0.03)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}>
                {p.translated || (translating ? '...' : '')}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Normal PDF panel view
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
                onClick={() => { setPdfCtx(null); setStatus(''); setResultHtml(''); setResultRaw(''); setCurrentPage(0); setTranslateLang(''); resetTranslation(); }}
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

            {/* Page content preview */}
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

              {/* Translate PDF dropdown + button */}
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
                  onClick={startTranslation}
                  disabled={!!activeStream || !translateLang || translating}
                  title={translateLang ? `Translate PDF to ${translateLang}` : 'Select a language first'}
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

            {/* Truncation warning */}
            {truncatedLen > 0 && (isStreaming || resultHtml) && (
              <div
                style={{
                  background: 'rgba(234, 179, 8, 0.12)',
                  border: '1px solid rgba(234, 179, 8, 0.3)',
                  borderRadius: 'var(--r-md)',
                  padding: '6px 10px',
                  fontSize: '11px',
                  color: 'rgb(202, 138, 4)',
                  flexShrink: 0,
                }}
              >
                Note: This PDF has {truncatedLen.toLocaleString()} characters. Analysis is limited to the first 20,000 characters.
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
