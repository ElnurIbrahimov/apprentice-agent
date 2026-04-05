import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ClipboardDocumentIcon,
  ArrowUpTrayIcon,
  StopIcon,
  TrashIcon,
  DocumentDuplicateIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
type Action = 'analyze' | 'extract_text' | 'explain_code' | 'describe_ui' | 'find_issues';

interface ActionOption {
  value: Action;
  label: string;
  icon: string;
  system: string;
}

/* ── Constants ── */
const ACTIONS: ActionOption[] = [
  {
    value: 'analyze',
    label: 'Analyze',
    icon: '🔍',
    system: 'Analyze this screenshot in detail. Describe what you see, identify key elements, and note anything significant.',
  },
  {
    value: 'extract_text',
    label: 'Extract Text',
    icon: '📄',
    system: 'Extract all visible text from this screenshot. Preserve formatting and structure.',
  },
  {
    value: 'explain_code',
    label: 'Explain Code',
    icon: '💻',
    system: 'This screenshot shows code. Explain what this code does, identify the language, and note any issues or improvements.',
  },
  {
    value: 'describe_ui',
    label: 'Describe UI',
    icon: '🎨',
    system: 'Describe the UI shown in this screenshot. Note the layout, components, color scheme, typography, and UX patterns used.',
  },
  {
    value: 'find_issues',
    label: 'Find Issues',
    icon: '⚠️',
    system: 'Analyze this screenshot for potential issues: UI bugs, accessibility problems, design inconsistencies, errors, or areas for improvement.',
  },
];

/* ── Component ── */
export function CapturePanel() {
  const [imageData, setImageData] = useState<string | null>(null);
  const [selectedAction, setSelectedAction] = useState<Action>('analyze');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const resultEndRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Fetch models
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

  // Close model menu on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Auto-scroll result during streaming
  useEffect(() => {
    if (isGenerating && resultEndRef.current) {
      resultEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [result, isGenerating]);

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const loadFile = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Only image files are supported.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      setImageData(ev.target?.result as string);
      setResult('');
      setError(null);
    };
    reader.readAsDataURL(file);
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (items) {
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) {
            loadFile(file);
            return;
          }
        }
      }
    }
  }, [loadFile]);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
    // Reset input so same file can be re-selected
    e.target.value = '';
  }, [loadFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    if (!dropZoneRef.current?.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) loadFile(file);
  }, [loadFile]);

  const handleAnalyze = useCallback(async () => {
    if (!imageData || isGenerating) return;

    const action = ACTIONS.find(a => a.value === selectedAction)!;
    const systemPrompt = action.system;
    const userMessage = question.trim() || action.label;

    setResult('');
    setError(null);
    setIsGenerating(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Extract base64 data (strip data URL prefix)
    const base64 = imageData.split(',')[1];
    const mimeType = imageData.split(';')[0].split(':')[1];

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          system_prompt: systemPrompt,
          images: [{ data: base64, media_type: mimeType }],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullResponse = '';

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
                if (text) {
                  fullResponse += text;
                  setResult(fullResponse);
                }
              } catch {
                fullResponse += data;
                setResult(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setResult(fullResponse);
            }
          }
        }
      } else {
        fullResponse = await res.text();
        setResult(fullResponse);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Something went wrong. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [imageData, isGenerating, selectedAction, question, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleClear = useCallback(() => {
    abortRef.current?.abort();
    setImageData(null);
    setResult('');
    setError(null);
    setIsGenerating(false);
    setQuestion('');
  }, []);

  const handleCopy = useCallback(async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(result);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback — ignore
    }
  }, [result]);

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Controls */}
      <div
        className="flex flex-col md:w-[360px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[55vh] bg-surface-0"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Capture Panel</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Paste or upload a screenshot for AI analysis</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Drop / paste zone */}
          {!imageData ? (
            <div
              ref={dropZoneRef}
              tabIndex={0}
              onPaste={handlePaste}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
              className="flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-all cursor-pointer select-none outline-none focus-visible:ring-2 focus-visible:ring-purple-500"
              style={{
                minHeight: 160,
                borderColor: isDragging ? 'var(--chat-accent, #7c3aed)' : 'var(--border-default)',
                background: isDragging ? 'color-mix(in srgb, var(--chat-accent, #7c3aed) 8%, transparent)' : 'var(--surface-1)',
              }}
            >
              <ClipboardDocumentIcon className="w-8 h-8" style={{ color: 'var(--text-secondary)' }} />
              <div className="text-center">
                <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                  Press Ctrl+V to paste a screenshot
                </p>
                <p className="text-[11px] mt-1" style={{ color: 'var(--text-secondary)' }}>
                  or drag & drop / click to upload
                </p>
              </div>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors"
                style={{ background: 'var(--surface-3)', color: 'var(--text-secondary)' }}
              >
                <ArrowUpTrayIcon className="w-3.5 h-3.5" />
                Upload image
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            /* Image preview */
            <div className="relative rounded-xl overflow-hidden border border-chat-border group">
              <img
                src={imageData}
                alt="Captured screenshot"
                className="w-full object-contain max-h-48"
                style={{ background: 'var(--surface-1)' }}
              />
              <button
                onClick={handleClear}
                className="absolute top-2 right-2 p-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ background: 'rgba(0,0,0,0.6)', color: '#fff' }}
                title="Remove image"
              >
                <TrashIcon className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Action selector */}
          <div>
            <p className="text-[11px] font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Action</p>
            <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
              {ACTIONS.map((a) => (
                <button
                  key={a.value}
                  onClick={() => setSelectedAction(a.value)}
                  className="flex items-center gap-1.5 px-2.5 py-2 rounded-lg border text-xs font-medium transition-all text-left"
                  style={{
                    borderColor: selectedAction === a.value ? 'var(--chat-accent, #7c3aed)' : 'var(--border-default)',
                    background: selectedAction === a.value
                      ? 'color-mix(in srgb, var(--chat-accent, #7c3aed) 15%, transparent)'
                      : 'var(--surface-1)',
                    color: selectedAction === a.value ? 'var(--text-primary)' : 'var(--text-secondary)',
                  }}
                >
                  <span>{a.icon}</span>
                  <span>{a.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Optional question */}
          <div>
            <p className="text-[11px] font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>
              Additional instruction <span style={{ color: 'var(--text-tertiary, var(--text-secondary))' }}>(optional)</span>
            </p>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey && imageData) {
                  e.preventDefault();
                  handleAnalyze();
                }
              }}
              placeholder="e.g. Focus on the navigation bar..."
              rows={2}
              className="w-full p-2.5 rounded-lg text-sm resize-none outline-none transition-colors"
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border-default)',
                color: 'var(--text-primary)',
              }}
              disabled={isGenerating}
            />
          </div>

          {/* Model selector */}
          <div ref={modelMenuRef} className="relative">
            <p className="text-[11px] font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Model</p>
            <button
              type="button"
              onClick={() => setShowModelMenu(p => !p)}
              className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg w-full"
              style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
            >
              <span className="flex-1 truncate text-left">
                {selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}
              </span>
              <svg className="w-3 h-3 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showModelMenu && availableModels.length > 0 && (
              <div
                className="absolute left-0 right-0 rounded-xl overflow-hidden z-50"
                style={{
                  top: 'calc(100% + 4px)',
                  maxHeight: 240,
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-default)',
                }}
              >
                <div style={{ maxHeight: 240, overflowY: 'auto', padding: 4 }}>
                  <button
                    onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                    className="w-full flex items-center px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
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
                      className="w-full flex items-center px-2.5 py-1.5 rounded-lg text-xs text-left truncate transition-colors"
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

        {/* Footer: Analyze button */}
        <div className="p-3 border-t border-chat-border flex-shrink-0 space-y-2">
          {error && (
            <p className="text-xs px-2 py-1.5 rounded-lg" style={{ background: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#f87171' }}>
              {error}
            </p>
          )}
          <div className="flex gap-2">
            {imageData && (
              <button
                onClick={handleClear}
                disabled={isGenerating}
                className="px-3 py-2 rounded-lg text-xs font-medium transition-colors disabled:opacity-40"
                style={{ background: 'var(--surface-1)', border: '1px solid var(--border-default)', color: 'var(--text-secondary)' }}
                title="Clear / retake"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={isGenerating ? handleStop : handleAnalyze}
              disabled={!imageData && !isGenerating}
              className="flex-1 flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-40"
              style={{ background: 'var(--chat-accent, #7c3aed)', color: '#fff' }}
            >
              {isGenerating ? (
                <>
                  <StopIcon className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  <span>{ACTIONS.find(a => a.value === selectedAction)?.icon}</span>
                  {ACTIONS.find(a => a.value === selectedAction)?.label}
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Right: Result */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Result toolbar */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-chat-border flex-shrink-0">
          <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
            {isGenerating ? (
              <span className="flex items-center gap-2">
                <span className="shimmer-bar h-2 w-16 rounded" />
                Analyzing...
              </span>
            ) : result ? (
              `Result · ${result.length} chars`
            ) : (
              'Analysis will appear here'
            )}
          </span>
          {result && (
            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs transition-colors"
              style={{
                background: copied ? 'color-mix(in srgb, #22c55e 15%, transparent)' : 'var(--surface-1)',
                color: copied ? '#4ade80' : 'var(--text-secondary)',
                border: '1px solid var(--border-default)',
              }}
            >
              {copied ? <CheckIcon className="w-3.5 h-3.5" /> : <DocumentDuplicateIcon className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          )}
        </div>

        {/* Result content */}
        <div className="flex-1 overflow-y-auto p-4">
          {result ? (
            <div
              className="text-sm leading-relaxed whitespace-pre-wrap"
              style={{ color: 'var(--text-primary)' }}
            >
              {result}
              {isGenerating && (
                <span
                  className="inline-block w-1.5 h-4 ml-0.5 align-middle animate-pulse"
                  style={{ background: 'var(--chat-accent, #7c3aed)' }}
                />
              )}
              <div ref={resultEndRef} />
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full gap-3" style={{ color: 'var(--text-secondary)' }}>
              {!imageData ? (
                <>
                  <ClipboardDocumentIcon className="w-10 h-10 opacity-30" />
                  <p className="text-sm">Paste a screenshot on the left to get started</p>
                </>
              ) : (
                <>
                  <span className="text-3xl">{ACTIONS.find(a => a.value === selectedAction)?.icon}</span>
                  <p className="text-sm">Press <strong>{ACTIONS.find(a => a.value === selectedAction)?.label}</strong> to analyze</p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
