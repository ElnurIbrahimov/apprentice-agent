import { useState, useRef, useEffect, useCallback } from 'react';
import { StopIcon, ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';

/* ── Types ── */
type InputMode = 'paste' | 'upload';
type Action = 'summarize' | 'keypoints' | 'qa' | 'explain';

interface ActionConfig {
  label: string;
  description: string;
  systemPrompt: (content: string, question?: string) => string;
}

const ACTIONS: Record<Action, ActionConfig> = {
  summarize: {
    label: 'Summarize',
    description: 'Concise summary of the document',
    systemPrompt: (content) =>
      `Summarize the following document content concisely. Highlight the main points.\n\n${content}`,
  },
  keypoints: {
    label: 'Key Points',
    description: 'Numbered list of key findings',
    systemPrompt: (content) =>
      `Extract the key points from this document as a numbered list. Include the most important facts, findings, and conclusions.\n\n${content}`,
  },
  qa: {
    label: 'Q&A',
    description: 'Ask a question about the document',
    systemPrompt: (content, question) =>
      `Based on the following document content, answer the user's question accurately. Quote relevant passages when possible.\n\nDocument:\n${content}\n\nQuestion: ${question}`,
  },
  explain: {
    label: 'Explain Simply',
    description: 'Plain-language explanation',
    systemPrompt: (content) =>
      `Explain the following document content in simple, easy-to-understand language. Avoid jargon.\n\n${content}`,
  },
};

/* ── Component ── */
export function PdfPanel() {
  const [inputMode, setInputMode] = useState<InputMode>('paste');
  const [pastedText, setPastedText] = useState('');
  const [uploadedText, setUploadedText] = useState('');
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [uploadNote, setUploadNote] = useState('');
  const [action, setAction] = useState<Action>('summarize');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  /* Fetch models */
  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  /* Close model menu on outside click */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* Abort on unmount */
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  /* Auto-scroll result */
  useEffect(() => {
    if (isGenerating && resultRef.current) {
      resultRef.current.scrollTop = resultRef.current.scrollHeight;
    }
  }, [result, isGenerating]);

  const readFileAsText = useCallback((file: File) => {
    setUploadedFileName(file.name);
    setUploadNote('');
    setUploadedText('');

    const ext = file.name.split('.').pop()?.toLowerCase();

    if (ext === 'txt' || ext === 'md' || ext === 'csv') {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedText(e.target?.result as string || '');
      };
      reader.onerror = () => setError('Failed to read file.');
      reader.readAsText(file);
      return;
    }

    if (ext === 'pdf') {
      const reader = new FileReader();
      reader.onload = (e) => {
        const raw = e.target?.result as string || '';
        // Attempt naive text extraction from PDF bytes
        // Works for text-based PDFs; scanned PDFs will yield garbage
        const text = raw
          .replace(/[^\x20-\x7E\n\r\t]/g, ' ')
          .replace(/ {3,}/g, ' ')
          .replace(/\n{3,}/g, '\n\n')
          .trim();

        if (text.length < 50) {
          setUploadNote(
            'This looks like a scanned or image-based PDF — text extraction failed. ' +
            'Please copy the text manually and use Paste mode instead.'
          );
          setUploadedText('');
        } else {
          setUploadedText(text);
          setUploadNote('PDF text extracted (may contain artifacts for complex layouts).');
        }
      };
      reader.onerror = () => setError('Failed to read file.');
      reader.readAsBinaryString(file);
      return;
    }

    setUploadNote(`Unsupported file type ".${ext}". Please use a .pdf or .txt file, or paste the text directly.`);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) readFileAsText(file);
  }, [readFileAsText]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFileAsText(file);
    // Reset so same file can be re-selected
    e.target.value = '';
  }, [readFileAsText]);

  const activeContent = inputMode === 'paste' ? pastedText : uploadedText;

  const handleAnalyze = useCallback(async () => {
    const content = activeContent.trim();
    if (!content) { setError('No content to analyze. Paste text or upload a file first.'); return; }
    if (action === 'qa' && !question.trim()) { setError('Enter a question for Q&A mode.'); return; }

    setError('');
    setResult('');
    setIsGenerating(true);

    const cfg = ACTIONS[action];
    const systemPrompt = cfg.systemPrompt(content, question.trim() || undefined);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: 'Analyze the document as instructed.',
          system_prompt: systemPrompt,
          history: [],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let full = '';

      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });

          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') continue;
              try {
                const parsed = JSON.parse(data);
                const text = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (text) { full += text; setResult(full); }
              } catch {
                full += data;
                setResult(full);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              full += line;
              setResult(full);
            }
          }
        }
      } else {
        full = await res.text();
        setResult(full);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(`Error: ${e.message}. Make sure the backend is running.`);
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [activeContent, action, question, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleCopy = useCallback(() => {
    if (!result) return;
    navigator.clipboard.writeText(result).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [result]);

  const hasContent = activeContent.trim().length > 0;

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">PDF Analyzer</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">
          Summarize, extract key points, or ask questions about any document
        </p>
      </div>

      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">
        {/* Left: Input panel */}
        <div className="flex flex-col md:w-[420px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 overflow-y-auto bg-surface-0">

          {/* Mode toggle */}
          <div className="px-4 pt-4 pb-2 flex-shrink-0">
            <div
              className="flex rounded-lg border border-chat-border overflow-hidden"
              style={{ background: 'var(--surface-1)' }}
            >
              {(['paste', 'upload'] as InputMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setInputMode(m)}
                  className="flex-1 py-1.5 text-xs font-medium capitalize transition-colors"
                  style={{
                    background: inputMode === m ? 'var(--chat-accent)' : 'transparent',
                    color: inputMode === m ? '#fff' : 'var(--text-secondary)',
                  }}
                >
                  {m === 'paste' ? 'Paste Text' : 'Upload File'}
                </button>
              ))}
            </div>
          </div>

          {/* Input area */}
          <div className="px-4 pb-3 flex-shrink-0">
            {inputMode === 'paste' ? (
              <textarea
                value={pastedText}
                onChange={(e) => setPastedText(e.target.value)}
                placeholder="Paste your document text here..."
                className="w-full h-40 p-3 rounded-lg text-xs font-mono resize-none outline-none"
                style={{
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-default)',
                  color: 'var(--text-primary)',
                }}
              />
            ) : (
              <div>
                {/* Drop zone */}
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className="cursor-pointer rounded-lg flex flex-col items-center justify-center h-28 transition-colors"
                  style={{
                    border: `2px dashed ${isDragging ? 'var(--chat-accent)' : 'var(--border-default)'}`,
                    background: isDragging ? 'var(--surface-2)' : 'var(--surface-1)',
                  }}
                >
                  <svg className="w-8 h-8 mb-2 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-xs text-chat-text-secondary">
                    {uploadedFileName ? uploadedFileName : 'Drop PDF or click to upload'}
                  </p>
                  <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
                    .pdf · .txt · .md
                  </p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md,.csv"
                  onChange={handleFileInput}
                  className="hidden"
                />

                {/* Upload note */}
                {uploadNote && (
                  <p className="mt-2 text-[10px] leading-relaxed rounded-md px-2 py-1.5"
                    style={{
                      background: 'var(--surface-2)',
                      color: uploadNote.includes('failed') ? '#f87171' : 'var(--text-secondary)',
                    }}
                  >
                    {uploadNote}
                  </p>
                )}

                {/* Extracted text preview */}
                {uploadedText && (
                  <div
                    className="mt-2 p-2 rounded-md text-[10px] font-mono max-h-24 overflow-y-auto"
                    style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
                  >
                    {uploadedText.slice(0, 400)}{uploadedText.length > 400 ? '…' : ''}
                  </div>
                )}
              </div>
            )}

            {hasContent && (
              <p className="mt-1 text-[10px]" style={{ color: 'var(--text-secondary)', opacity: 0.6 }}>
                {activeContent.trim().length.toLocaleString()} characters
              </p>
            )}
          </div>

          {/* Action selector */}
          <div className="px-4 pb-3 flex-shrink-0">
            <p className="text-[10px] font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Action</p>
            <div className="grid grid-cols-2 gap-1.5">
              {(Object.entries(ACTIONS) as [Action, ActionConfig][]).map(([key, cfg]) => (
                <button
                  key={key}
                  onClick={() => setAction(key)}
                  className="flex flex-col items-start p-2.5 rounded-lg text-left transition-all"
                  style={{
                    background: action === key ? 'var(--chat-accent)' : 'var(--surface-1)',
                    border: `1px solid ${action === key ? 'var(--chat-accent)' : 'var(--border-default)'}`,
                    color: action === key ? '#fff' : 'var(--text-primary)',
                  }}
                >
                  <span className="text-xs font-medium">{cfg.label}</span>
                  <span
                    className="text-[10px] mt-0.5 leading-tight"
                    style={{ opacity: action === key ? 0.85 : 0.55 }}
                  >
                    {cfg.description}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Q&A question input */}
          {action === 'qa' && (
            <div className="px-4 pb-3 flex-shrink-0">
              <p className="text-[10px] font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Your Question</p>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g. What are the main conclusions of this study?"
                rows={3}
                className="w-full p-2.5 rounded-lg text-xs resize-none outline-none"
                style={{
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-default)',
                  color: 'var(--text-primary)',
                }}
              />
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mx-4 mb-3 px-3 py-2 rounded-lg text-xs" style={{ background: '#7f1d1d33', color: '#f87171', border: '1px solid #991b1b55' }}>
              {error}
            </div>
          )}

          {/* Analyze button + model selector */}
          <div className="px-4 pb-4 flex-shrink-0 space-y-2">
            <button
              onClick={isGenerating ? handleStop : handleAnalyze}
              disabled={!isGenerating && !hasContent}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-all"
              style={{
                background: isGenerating ? '#991b1b' : 'var(--chat-accent)',
                color: '#fff',
                opacity: (!isGenerating && !hasContent) ? 0.4 : 1,
                cursor: (!isGenerating && !hasContent) ? 'not-allowed' : 'pointer',
              }}
            >
              {isGenerating ? (
                <>
                  <StopIcon className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  Analyze
                </>
              )}
            </button>

            {/* Model selector */}
            <div ref={modelMenuRef} className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu(p => !p)}
                className="flex items-center gap-1 text-[10px] transition-colors px-2 py-1 rounded-md w-full"
                style={{ background: 'var(--border-subtle)', color: 'var(--text-secondary)' }}
              >
                <span className="flex-1 truncate text-left">
                  Model: {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                </span>
                <svg className="w-2.5 h-2.5 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelMenu && availableModels.length > 0 && (
                <div
                  className="absolute bottom-8 left-0 w-full rounded-xl overflow-hidden z-50"
                  style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)', maxHeight: 220 }}
                >
                  <div style={{ maxHeight: 220, overflowY: 'auto', padding: 4 }}>
                    <button
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                      style={{
                        color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
                        background: !selectedModel ? 'var(--surface-3)' : 'transparent',
                      }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map((m) => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left truncate transition-colors"
                        style={{
                          color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                          background: selectedModel === m ? 'var(--surface-3)' : 'transparent',
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Result panel */}
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {/* Result header */}
          <div className="px-4 py-2.5 border-b border-chat-border flex items-center justify-between flex-shrink-0">
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-chat-text">Result</span>
              {isGenerating && (
                <div className="flex items-center gap-1.5">
                  <div className="shimmer-bar h-1.5 w-12" />
                  <span className="text-[10px]" style={{ color: 'var(--chat-accent)' }}>Analyzing…</span>
                </div>
              )}
            </div>
            {result && (
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] transition-colors"
                style={{
                  background: copied ? '#14532d55' : 'var(--surface-1)',
                  color: copied ? '#4ade80' : 'var(--text-secondary)',
                  border: '1px solid var(--border-default)',
                }}
              >
                {copied ? (
                  <><CheckIcon className="w-3 h-3" /> Copied</>
                ) : (
                  <><ClipboardDocumentIcon className="w-3 h-3" /> Copy</>
                )}
              </button>
            )}
          </div>

          {/* Result content */}
          <div
            ref={resultRef}
            className="flex-1 overflow-y-auto p-4"
          >
            {result ? (
              <pre className="text-sm whitespace-pre-wrap leading-relaxed font-sans" style={{ color: 'var(--text-primary)' }}>
                {result}
                {isGenerating && (
                  <span className="inline-block w-1.5 h-4 ml-0.5 align-middle animate-pulse"
                    style={{ background: 'var(--chat-accent)' }} />
                )}
              </pre>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center">
                <svg className="w-12 h-12 mb-3 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                  {hasContent ? 'Press Analyze to get results' : 'Paste or upload a document to get started'}
                </p>
                <p className="text-[10px] mt-1 max-w-xs" style={{ color: 'var(--text-secondary)', opacity: 0.5 }}>
                  Works best with text-based PDFs. For scanned PDFs, copy the text manually and use Paste mode.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
