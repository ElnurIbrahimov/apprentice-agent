import { useState, useRef, useEffect, useCallback, DragEvent } from 'react';
import {
  DocumentMagnifyingGlassIcon,
  PhotoIcon,
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  StopIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type Action = 'extract' | 'describe' | 'data';

interface ActionOption {
  id: Action;
  label: string;
  prompt: string;
  systemPrompt: string;
}

/* ── Constants ── */
const ACTIONS: ActionOption[] = [
  {
    id: 'extract',
    label: 'Extract Text',
    prompt: 'Extract all text from this image',
    systemPrompt:
      'Extract ALL text visible in this image. Preserve the original formatting, line breaks, and structure as much as possible. Output only the extracted text.',
  },
  {
    id: 'describe',
    label: 'Describe Image',
    prompt: 'Describe this image in detail',
    systemPrompt:
      'Describe this image in detail. Include: what\'s shown, text visible, colors, layout, and any notable elements.',
  },
  {
    id: 'data',
    label: 'Extract Data',
    prompt: 'Extract structured data from this image',
    systemPrompt:
      'Extract any structured data from this image (tables, numbers, lists, forms). Format the data clearly, using markdown tables where appropriate.',
  },
];

/* ── Helpers ── */
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the data URL prefix — keep only the base64 payload
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/* ── Component ── */
export function OcrPanel() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [base64Data, setBase64Data] = useState<string | null>(null);
  const [action, setAction] = useState<Action>('extract');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const outputRef = useRef<HTMLTextAreaElement>(null);

  // Fetch available models
  useEffect(() => {
    apiFetch('/api/models')
      .then((res) => res.json())
      .then((data) => {
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

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Revoke object URL on image change
  useEffect(() => {
    return () => {
      if (imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl);
    };
  }, [imagePreviewUrl]);

  // Clipboard paste (Ctrl+V anywhere in window)
  useEffect(() => {
    const handler = (e: globalThis.ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of Array.from(items)) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) loadImage(file);
          break;
        }
      }
    };
    window.addEventListener('paste', handler);
    return () => window.removeEventListener('paste', handler);
  }, []);

  const loadImage = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Only image files are supported.');
      return;
    }
    setError(null);
    setOutput('');
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImagePreviewUrl(url);
    try {
      const b64 = await fileToBase64(file);
      setBase64Data(b64);
    } catch {
      setError('Failed to read image file.');
    }
  }, []);

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadImage(file);
    // Reset so same file can be re-selected
    e.target.value = '';
  }, [loadImage]);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) loadImage(file);
  }, [loadImage]);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleClearImage = useCallback(() => {
    setImageFile(null);
    setImagePreviewUrl(null);
    setBase64Data(null);
    setOutput('');
    setError(null);
  }, []);

  const handleExtract = useCallback(async () => {
    if (!base64Data || isGenerating) return;
    setError(null);
    setOutput('');
    setIsGenerating(true);

    const selectedAction = ACTIONS.find((a) => a.id === action)!;
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: selectedAction.prompt,
          system_prompt: selectedAction.systemPrompt,
          images: [base64Data],
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
                const text =
                  parsed.choices?.[0]?.delta?.content ||
                  parsed.content ||
                  parsed.chunk ||
                  '';
                if (text) {
                  fullResponse += text;
                  setOutput(fullResponse);
                }
              } catch {
                fullResponse += data;
                setOutput(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setOutput(fullResponse);
            }
          }
        }
      } else {
        const text = await res.text();
        fullResponse = text;
        setOutput(fullResponse);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Something went wrong.');
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [base64Data, isGenerating, action, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleCopy = useCallback(async () => {
    if (!output) return;
    try {
      await navigator.clipboard.writeText(output);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback: select all in textarea
      outputRef.current?.select();
    }
  }, [output]);

  const selectedActionObj = ACTIONS.find((a) => a.id === action)!;

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">

      {/* Left: Upload + controls */}
      <div
        className="flex flex-col md:w-[400px] md:min-w-[300px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[50vh] bg-surface-0"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">OCR / Image Reader</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">
            Upload an image — extract text, describe, or pull out data
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">

          {/* Drop zone / preview */}
          {!imageFile ? (
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className="relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed cursor-pointer transition-colors select-none"
              style={{
                minHeight: 180,
                borderColor: isDragOver ? 'var(--chat-accent)' : 'var(--border-default)',
                background: isDragOver ? 'var(--surface-2)' : 'var(--surface-1)',
              }}
            >
              <PhotoIcon className="w-10 h-10 text-chat-text-secondary opacity-40" />
              <div className="text-center">
                <p className="text-sm text-chat-text">Drop image here</p>
                <p className="text-[10px] text-chat-text-secondary mt-0.5">
                  or click to browse &nbsp;·&nbsp; Ctrl+V to paste
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            <div className="relative rounded-xl overflow-hidden border border-chat-border" style={{ background: 'var(--surface-1)' }}>
              <img
                src={imagePreviewUrl!}
                alt="Preview"
                className="w-full object-contain max-h-64"
                style={{ display: 'block' }}
              />
              <button
                onClick={handleClearImage}
                title="Remove image"
                className="absolute top-2 right-2 p-1 rounded-full text-white transition-opacity hover:opacity-80"
                style={{ background: 'rgba(0,0,0,0.55)' }}
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
              {imageFile && (
                <div
                  className="px-3 py-1.5 text-[10px] text-chat-text-secondary border-t border-chat-border truncate"
                  style={{ background: 'var(--surface-1)' }}
                >
                  {imageFile.name} &nbsp;·&nbsp; {(imageFile.size / 1024).toFixed(0)} KB
                </div>
              )}
            </div>
          )}

          {/* Action selector */}
          <div>
            <p className="text-[10px] text-chat-text-secondary mb-1.5 font-medium uppercase tracking-wide">Action</p>
            <div className="flex rounded-lg border border-chat-border overflow-hidden">
              {ACTIONS.map((a) => (
                <button
                  key={a.id}
                  onClick={() => setAction(a.id)}
                  className="flex-1 py-1.5 text-[11px] font-medium transition-colors"
                  style={{
                    background: action === a.id ? 'var(--chat-accent)' : 'var(--surface-1)',
                    color: action === a.id ? '#fff' : 'var(--text-secondary)',
                    borderRight: a.id !== 'data' ? '1px solid var(--border-default)' : 'none',
                  }}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          {/* Model selector */}
          <div ref={modelMenuRef}>
            <p className="text-[10px] text-chat-text-secondary mb-1.5 font-medium uppercase tracking-wide">Model</p>
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowModelMenu((p) => !p)}
                className="w-full flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors border border-chat-border"
                style={{ background: 'var(--surface-1)', color: 'var(--text-secondary)' }}
              >
                <span className="truncate">
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}
                </span>
                <svg className="w-3 h-3 opacity-50 flex-shrink-0 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showModelMenu && (
                <div
                  style={{
                    position: 'absolute',
                    top: 'calc(100% + 4px)',
                    left: 0,
                    right: 0,
                    maxHeight: 240,
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    borderRadius: 10,
                    overflow: 'hidden',
                    zIndex: 50,
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
                        className="w-full flex items-center px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
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

          {/* Error */}
          {error && (
            <div
              className="px-3 py-2 rounded-lg text-xs"
              style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171', border: '1px solid rgba(239,68,68,0.25)' }}
            >
              {error}
            </div>
          )}
        </div>

        {/* Extract button */}
        <div className="p-3 border-t border-chat-border flex-shrink-0">
          <button
            onClick={isGenerating ? handleStop : handleExtract}
            disabled={!base64Data && !isGenerating}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg text-sm font-medium transition-opacity disabled:opacity-40"
            style={{ background: 'var(--chat-accent)', color: '#fff' }}
          >
            {isGenerating ? (
              <>
                <StopIcon className="w-4 h-4" />
                Stop
              </>
            ) : (
              <>
                <DocumentMagnifyingGlassIcon className="w-4 h-4" />
                {selectedActionObj.label}
              </>
            )}
          </button>
        </div>
      </div>

      {/* Right: Output */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Output header */}
        <div
          className="flex items-center justify-between px-4 py-2.5 border-b border-chat-border flex-shrink-0"
        >
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-chat-text">Output</span>
            {isGenerating && (
              <div className="flex items-center gap-1.5 text-[10px] text-purple-400">
                <div className="shimmer-bar h-1.5 w-14 rounded-full" />
                Generating...
              </div>
            )}
          </div>
          <button
            onClick={handleCopy}
            disabled={!output}
            title="Copy to clipboard"
            className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] transition-colors disabled:opacity-30"
            style={{
              background: copied ? 'rgba(34,197,94,0.15)' : 'var(--surface-2)',
              color: copied ? '#4ade80' : 'var(--text-secondary)',
            }}
          >
            {copied ? (
              <><ClipboardDocumentCheckIcon className="w-3.5 h-3.5" /> Copied</>
            ) : (
              <><ClipboardDocumentIcon className="w-3.5 h-3.5" /> Copy</>
            )}
          </button>
        </div>

        {/* Textarea */}
        <div className="flex-1 overflow-hidden p-3">
          {output || isGenerating ? (
            <textarea
              ref={outputRef}
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              className="w-full h-full resize-none outline-none text-sm font-mono leading-relaxed p-2 rounded-lg"
              style={{
                background: 'var(--surface-1)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-subtle)',
                caretColor: 'var(--chat-accent)',
              }}
              spellCheck={false}
              placeholder={isGenerating ? 'Extracting…' : ''}
            />
          ) : (
            <div
              className="flex flex-col items-center justify-center h-full gap-3 text-center"
              style={{ color: 'var(--text-secondary)' }}
            >
              <DocumentMagnifyingGlassIcon className="w-12 h-12 opacity-20" />
              <div>
                <p className="text-sm">No output yet</p>
                <p className="text-[10px] mt-1 opacity-60">
                  Upload an image and click Extract
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
