import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  StopIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline';
import { useChatStore } from '../store/chatStore';
import { SendToMenu } from './SendToMenu';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type InputMode = 'text' | 'url';
type SummaryLength = 'brief' | 'standard' | 'detailed';
type SummaryFormat = 'paragraph' | 'bullets' | 'takeaways';

interface LengthOption {
  value: SummaryLength;
  label: string;
  description: string;
}

interface FormatOption {
  value: SummaryFormat;
  label: string;
}

/* ── Constants ── */
const LENGTH_OPTIONS: LengthOption[] = [
  { value: 'brief', label: 'Brief', description: '1–2 sentences' },
  { value: 'standard', label: 'Standard', description: '1 paragraph' },
  { value: 'detailed', label: 'Detailed', description: 'Full + key points' },
];

const FORMAT_OPTIONS: FormatOption[] = [
  { value: 'paragraph', label: 'Paragraph' },
  { value: 'bullets', label: 'Bullet Points' },
  { value: 'takeaways', label: 'Key Takeaways' },
];

const LENGTH_INSTRUCTIONS: Record<SummaryLength, string> = {
  brief: 'Summarize in 1-2 sentences only.',
  standard: 'Summarize in one concise paragraph.',
  detailed: 'Write a full summary with key points listed at the end.',
};

const FORMAT_INSTRUCTIONS: Record<SummaryFormat, string> = {
  paragraph: 'Write in flowing prose paragraphs.',
  bullets: 'Use bullet points (markdown - style).',
  takeaways: 'Structure as "Key Takeaways:" followed by a numbered list of the most important insights.',
};

function buildSystemPrompt(mode: InputMode, length: SummaryLength, format: SummaryFormat): string {
  if (mode === 'url') {
    return `The user wants to summarize content from a URL. Since you cannot access URLs directly, summarize based on what you know about the topic or ask the user to paste the content instead. Output ONLY the summary or your response — no preamble.`;
  }
  return `Summarize the following text. ${LENGTH_INSTRUCTIONS[length]} ${FORMAT_INSTRUCTIONS[format]} Output ONLY the summary, no preamble, no explanation.`;
}

function countWords(text: string): number {
  return text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
}

const DRAFT_KEY = 'aura-draft-summary';

/* ── Main Component ── */
export function SummaryPanel() {
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [inputText, setInputText] = useState('');
  const [urlInput, _setUrlInput] = useState('');
  const [length, setLength] = useState<SummaryLength>('standard');
  const [format, setFormat] = useState<SummaryFormat>('paragraph');
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  /* ── Prefill from tool suggestion ── */
  useEffect(() => {
    const prefill = useChatStore.getState().toolPrefill;
    if (prefill?.toolId === 'summary') {
      setInputText(prefill.query);
      useChatStore.getState().clearToolPrefill();
    }
  }, []);

  /* ── Load draft on mount ── */
  useEffect(() => {
    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) {
        const parsed = JSON.parse(draft);
        if (parsed.input) setInputText(parsed.input);
        if (parsed.output) setOutput(parsed.output);
        if (parsed.inputMode) setInputMode(parsed.inputMode);
        if (parsed.length) setLength(parsed.length);
        if (parsed.format) setFormat(parsed.format);
        setDraftLoaded(true);
      }
    } catch {}
  }, []);

  /* ── Auto-save on change ── */
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({ input: inputText, output, inputMode, length, format, timestamp: Date.now() }));
      } catch {}
    }, 5000);
    return () => clearTimeout(timer);
  }, [inputText, output, inputMode, length, format]);

  /* ── Fetch models ── */
  useEffect(() => {
    apiFetch('/api/models')
      .then(res => res.json())
      .then(data => {
        const all: string[] = [
          ...(data.chatgpt_models || []),
          ...(data.direct_api_models || []),
          ...(data.cloud_models || []),
          ...(data.local_models || []),
        ];
        if (all.length > 0) setAvailableModels(all);
      })
      .catch(() => {});
  }, []);

  /* ── Close model menu on outside click ── */
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modelMenuRef.current && !modelMenuRef.current.contains(e.target as Node)) {
        setShowModelMenu(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  /* ── Abort on unmount ── */
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  /* ── Auto-scroll output during streaming ── */
  useEffect(() => {
    if (isGenerating && outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [output, isGenerating]);

  const activeInput = inputMode === 'text' ? inputText : urlInput;
  const inputWordCount = countWords(inputMode === 'text' ? inputText : urlInput);
  const outputWordCount = countWords(output);
  const canSummarize = activeInput.trim().length > 0 && !isGenerating;

  const handleSummarize = useCallback(async () => {
    if (!canSummarize) return;
    setError(null);
    setOutput('');
    setIsGenerating(true);

    const systemPrompt = buildSystemPrompt(inputMode, length, format);
    const userMessage = inputMode === 'url'
      ? `Please summarize or respond to the content at this URL: ${urlInput.trim()}`
      : inputText.trim();

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          system_prompt: systemPrompt,
          history: [],
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
        setError(e.message || 'Something went wrong. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [canSummarize, inputMode, inputText, urlInput, length, format, selectedModel]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
  }, []);

  const handleCopy = useCallback(() => {
    if (!output) return;
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [output]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-5 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Summarizer</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">Paste text or enter a URL to get a summary</p>
      </div>

      {/* Draft recovered banner */}
      {draftLoaded && (
        <div className="flex items-center gap-2 px-3 py-1.5 mx-5 mt-2 bg-blue-500/10 text-blue-400 text-[10px] rounded-lg flex-shrink-0">
          <span className="flex-1">Draft recovered</span>
          <button onClick={() => { setDraftLoaded(false); try { localStorage.removeItem(DRAFT_KEY); } catch {} }} className="hover:text-blue-300 transition-colors">Dismiss</button>
        </div>
      )}

      {/* Body: two-column on md+ */}
      <div className="flex flex-col md:flex-row flex-1 overflow-hidden">

        {/* ── Left: Input panel ── */}
        <div className="flex flex-col md:w-[420px] md:min-w-[320px] border-b md:border-b-0 md:border-r border-chat-border flex-shrink-0 max-md:max-h-[55vh]">

          {/* Header */}
          <div className="flex items-center gap-2 p-3 border-b border-chat-border flex-shrink-0">
            <DocumentTextIcon className="w-4 h-4 text-chat-text-secondary" />
            <span className="text-xs font-medium text-chat-text-secondary">Paste text to summarize</span>
          </div>

          {/* Input area */}
          <div className="flex-1 p-3 overflow-hidden flex flex-col gap-2">
            <textarea
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="Paste your text here..."
              className="flex-1 w-full p-3 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-sm resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50 leading-relaxed"
              disabled={isGenerating}
            />
            <div className="text-[10px] text-chat-text-secondary text-right">
              {inputWordCount} {inputWordCount === 1 ? 'word' : 'words'}
            </div>
          </div>

          {/* Options */}
          <div className="p-3 border-t border-chat-border flex-shrink-0 space-y-3">
            {/* Length selector */}
            <div>
              <p className="text-[10px] font-medium text-chat-text-secondary mb-1.5 uppercase tracking-wide">Length</p>
              <div className="flex gap-1.5">
                {LENGTH_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setLength(opt.value)}
                    className={`flex-1 py-1.5 rounded-lg text-[11px] font-medium transition-colors border ${
                      length === opt.value
                        ? 'border-chat-accent text-chat-accent'
                        : 'border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-text-secondary/40'
                    }`}
                    style={{ background: length === opt.value ? 'var(--surface-2, var(--surface-1))' : 'var(--surface-1)' }}
                    title={opt.description}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Format selector */}
            <div>
              <p className="text-[10px] font-medium text-chat-text-secondary mb-1.5 uppercase tracking-wide">Format</p>
              <div className="flex gap-1.5">
                {FORMAT_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setFormat(opt.value)}
                    className={`flex-1 py-1.5 rounded-lg text-[11px] font-medium transition-colors border ${
                      format === opt.value
                        ? 'border-chat-accent text-chat-accent'
                        : 'border-chat-border text-chat-text-secondary hover:text-chat-text hover:border-chat-text-secondary/40'
                    }`}
                    style={{ background: format === opt.value ? 'var(--surface-2, var(--surface-1))' : 'var(--surface-1)' }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Model selector + Summarize button */}
            <div className="flex items-center gap-2">
              {/* Model dropdown */}
              <div className="relative" ref={modelMenuRef}>
                <button
                  type="button"
                  onClick={() => setShowModelMenu(p => !p)}
                  className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1.5 rounded-md"
                  style={{ background: 'var(--border-subtle, var(--surface-1))' }}
                >
                  <span className="max-w-[110px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
                  <svg className="w-2.5 h-2.5 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {showModelMenu && (
                  <div
                    style={{
                      position: 'absolute', bottom: 32, left: 0, width: 220, maxHeight: 260,
                      background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                      borderRadius: 10, overflow: 'hidden', zIndex: 50,
                    }}
                  >
                    <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
                      <button
                        onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                        style={{
                          color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
                          background: !selectedModel ? 'var(--surface-3, var(--surface-2, var(--surface-1)))' : 'transparent',
                        }}
                      >
                        Auto (recommended)
                      </button>
                      {availableModels.map(m => (
                        <button
                          key={m}
                          onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                          className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                          style={{
                            color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                            background: selectedModel === m ? 'var(--surface-3, var(--surface-2, var(--surface-1)))' : 'transparent',
                          }}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Summarize / Stop */}
              <button
                onClick={isGenerating ? handleStop : handleSummarize}
                disabled={!isGenerating && !canSummarize}
                className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-medium text-white bg-chat-accent hover:opacity-90 disabled:opacity-40 transition-opacity"
              >
                {isGenerating ? (
                  <>
                    <StopIcon className="w-3.5 h-3.5" />
                    Stop
                  </>
                ) : (
                  'Summarize'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* ── Right: Output panel ── */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Output header */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-chat-border flex-shrink-0">
            <span className="text-xs font-medium text-chat-text">Summary</span>
            <div className="flex items-center gap-3">
              {output && (
                <span className="text-[10px] text-chat-text-secondary">
                  {outputWordCount} {outputWordCount === 1 ? 'word' : 'words'}
                </span>
              )}
              <button
                onClick={handleCopy}
                disabled={!output}
                className="flex items-center gap-1 text-[11px] text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
                title="Copy summary"
              >
                {copied ? (
                  <>
                    <ClipboardDocumentCheckIcon className="w-3.5 h-3.5 text-green-400" />
                    <span className="text-green-400">Copied</span>
                  </>
                ) : (
                  <>
                    <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                    Copy
                  </>
                )}
              </button>
              <SendToMenu content={output} sourceToolId="summary" />
            </div>
          </div>

          {/* Output content */}
          <div ref={outputRef} className="flex-1 overflow-y-auto p-4">
            {error ? (
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
                {error}
              </div>
            ) : output ? (
              <div className="text-sm text-chat-text leading-relaxed whitespace-pre-wrap">
                {output}
                {isGenerating && (
                  <span className="inline-block w-1.5 h-3.5 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center text-chat-text-secondary">
                {isGenerating ? (
                  <div className="flex items-center gap-2 text-xs text-purple-400">
                    <div className="shimmer-bar h-2 w-20" />
                    Summarizing...
                  </div>
                ) : (
                  <>
                    <p className="text-2xl mb-2">📄</p>
                    <p className="text-sm">Your summary will appear here</p>
                    <p className="text-[11px] mt-1">
                      {inputMode === 'text'
                        ? 'Paste some text on the left and click Summarize'
                        : 'Enter a URL on the left and click Summarize'}
                    </p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
