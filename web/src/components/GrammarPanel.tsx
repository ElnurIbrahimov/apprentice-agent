import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ClipboardDocumentIcon,
  CheckIcon,
  StopIcon,
  ArrowsRightLeftIcon,
  DocumentCheckIcon,
} from '@heroicons/react/24/outline';

/* ── Types ── */
type CheckMode = 'grammar' | 'style' | 'rewrite';
type ViewMode = 'output' | 'split';

interface ModeConfig {
  label: string;
  description: string;
  systemPrompt: string;
}

/* ── Mode configs ── */
const MODES: Record<CheckMode, ModeConfig> = {
  grammar: {
    label: 'Grammar Only',
    description: 'Fix grammar, spelling & punctuation',
    systemPrompt:
      "Check the following text for grammar, spelling, and punctuation errors. Output the corrected text. After the corrected text, list each change you made on a new line starting with '→'.",
  },
  style: {
    label: 'Grammar + Style',
    description: 'Fix issues and improve clarity',
    systemPrompt:
      "Check the following text for grammar, spelling, punctuation, and style issues. Improve clarity and flow while preserving meaning. Output the corrected text. After the corrected text, list each change on a new line starting with '→'.",
  },
  rewrite: {
    label: 'Full Rewrite',
    description: 'Rewrite for clarity and professionalism',
    systemPrompt:
      "Rewrite the following text to be clearer, more concise, and more professional while preserving the original meaning. Output the rewritten text. After the rewritten text, list major changes on a new line starting with '→'.",
  },
};

/* ── Helpers ── */
function splitCorrectedAndChanges(raw: string): { corrected: string; changes: string[] } {
  const lines = raw.split('\n');
  const changeLines: string[] = [];
  const correctedLines: string[] = [];
  let inChanges = false;

  for (const line of lines) {
    if (line.trimStart().startsWith('→')) {
      inChanges = true;
      changeLines.push(line.trim());
    } else if (inChanges) {
      // Once we hit change lines, remaining non-arrow lines are still part of changes block
      if (line.trim()) changeLines.push(line.trim());
    } else {
      correctedLines.push(line);
    }
  }

  return {
    corrected: correctedLines.join('\n').trim(),
    changes: changeLines,
  };
}

const DRAFT_KEY = 'aura-draft-grammar';

/* ── Main Component ── */
export function GrammarPanel() {
  const [inputText, setInputText] = useState('');
  const [mode, setMode] = useState<CheckMode>('grammar');
  const [viewMode, setViewMode] = useState<ViewMode>('output');
  const [rawOutput, setRawOutput] = useState('');
  const [streamingOutput, setStreamingOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedOutput, setCopiedOutput] = useState(false);
  const [appliedCorrections, setAppliedCorrections] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const outputScrollRef = useRef<HTMLDivElement>(null);

  const { corrected, changes } = rawOutput
    ? splitCorrectedAndChanges(rawOutput)
    : { corrected: '', changes: [] };

  // Load draft on mount
  useEffect(() => {
    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) {
        const parsed = JSON.parse(draft);
        if (parsed.inputText) setInputText(parsed.inputText);
        if (parsed.correctedText) setRawOutput(parsed.correctedText);
        if (parsed.checkMode) setMode(parsed.checkMode);
        setDraftLoaded(true);
      }
    } catch {}
  }, []);

  // Auto-save on change
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({ inputText, correctedText: rawOutput, checkMode: mode, timestamp: Date.now() }));
      } catch {}
    }, 5000);
    return () => clearTimeout(timer);
  }, [inputText, rawOutput, mode]);

  // Fetch models
  useEffect(() => {
    fetch('/api/models')
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

  // Auto-scroll output while streaming
  useEffect(() => {
    if (isGenerating && outputScrollRef.current) {
      outputScrollRef.current.scrollTop = outputScrollRef.current.scrollHeight;
    }
  }, [streamingOutput, isGenerating]);

  const handleCheck = useCallback(async () => {
    if (!inputText.trim() || isGenerating) return;

    setIsGenerating(true);
    setError(null);
    setRawOutput('');
    setStreamingOutput('');
    setAppliedCorrections(false);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: inputText,
          system_prompt: MODES[mode].systemPrompt,
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
                const text =
                  parsed.choices?.[0]?.delta?.content ||
                  parsed.content ||
                  parsed.chunk ||
                  '';
                if (text) {
                  fullResponse += text;
                  setStreamingOutput(fullResponse);
                }
              } catch {
                fullResponse += data;
                setStreamingOutput(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setStreamingOutput(fullResponse);
            }
          }
        }
      } else {
        fullResponse = await res.text();
      }

      setRawOutput(fullResponse);
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Something went wrong. Make sure the backend is running.');
      }
    } finally {
      setIsGenerating(false);
      setStreamingOutput('');
      abortRef.current = null;
    }
  }, [inputText, mode, selectedModel, isGenerating]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsGenerating(false);
    if (streamingOutput) {
      setRawOutput(streamingOutput);
      setStreamingOutput('');
    }
  }, [streamingOutput]);

  const handleApplyCorrections = useCallback(() => {
    if (!corrected) return;
    setInputText(corrected);
    setAppliedCorrections(true);
    setTimeout(() => setAppliedCorrections(false), 2000);
  }, [corrected]);

  const handleCopyOutput = useCallback(async () => {
    if (!corrected) return;
    try {
      await navigator.clipboard.writeText(corrected);
      setCopiedOutput(true);
      setTimeout(() => setCopiedOutput(false), 2000);
    } catch {
      // fallback
      const ta = document.createElement('textarea');
      ta.value = corrected;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      setCopiedOutput(true);
      setTimeout(() => setCopiedOutput(false), 2000);
    }
  }, [corrected]);

  const displayOutput = isGenerating ? streamingOutput : rawOutput;
  const { corrected: displayCorrected, changes: displayChanges } = displayOutput
    ? splitCorrectedAndChanges(displayOutput)
    : { corrected: '', changes: [] };

  const hasOutput = Boolean(displayOutput);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-chat-text">Grammar Checker</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">
            Paste text to check for grammar, style, or a full rewrite
          </p>
        </div>
        {/* View toggle — only relevant when output exists */}
        {hasOutput && (
          <div className="flex rounded-md border border-chat-border overflow-hidden">
            <button
              onClick={() => setViewMode('output')}
              className={`px-2.5 py-1 text-[10px] transition-colors ${
                viewMode === 'output'
                  ? 'bg-chat-accent text-white'
                  : 'text-chat-text-secondary hover:text-chat-text'
              }`}
            >
              Output
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`flex items-center gap-1 px-2.5 py-1 text-[10px] transition-colors ${
                viewMode === 'split'
                  ? 'bg-chat-accent text-white'
                  : 'text-chat-text-secondary hover:text-chat-text'
              }`}
            >
              <ArrowsRightLeftIcon className="w-3 h-3" />
              Side by side
            </button>
          </div>
        )}
      </div>

      {/* Draft recovered banner */}
      {draftLoaded && (
        <div className="flex items-center gap-2 px-3 py-1.5 mx-4 mt-2 bg-blue-500/10 text-blue-400 text-[10px] rounded-lg flex-shrink-0">
          <span className="flex-1">Draft recovered</span>
          <button onClick={() => { setDraftLoaded(false); try { localStorage.removeItem(DRAFT_KEY); } catch {} }} className="hover:text-blue-300 transition-colors">Dismiss</button>
        </div>
      )}

      {/* Mode selector */}
      <div className="px-4 py-2.5 border-b border-chat-border flex-shrink-0 flex gap-2 flex-wrap">
        {(Object.entries(MODES) as [CheckMode, ModeConfig][]).map(([key, cfg]) => (
          <button
            key={key}
            onClick={() => setMode(key)}
            className={`flex flex-col items-start px-3 py-2 rounded-lg border text-left transition-all text-xs ${
              mode === key
                ? 'border-chat-accent bg-chat-accent/10 text-chat-text'
                : 'border-chat-border text-chat-text-secondary hover:border-chat-accent/40 hover:text-chat-text bg-surface-1'
            }`}
          >
            <span className="font-medium">{cfg.label}</span>
            <span className="text-[10px] opacity-70 mt-0.5">{cfg.description}</span>
          </button>
        ))}
      </div>

      {/* Main area */}
      <div className={`flex-1 overflow-hidden flex ${viewMode === 'split' && hasOutput ? 'flex-row' : 'flex-col'}`}>
        {/* Input pane */}
        <div
          className={`flex flex-col ${
            viewMode === 'split' && hasOutput
              ? 'w-1/2 border-r border-chat-border'
              : 'flex-1'
          } ${viewMode === 'output' && hasOutput ? 'hidden' : ''} min-h-0`}
        >
          {viewMode === 'split' && hasOutput && (
            <div className="px-3 py-1.5 border-b border-chat-border flex-shrink-0">
              <span className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">
                Original
              </span>
            </div>
          )}
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Paste your text here..."
            className="flex-1 w-full p-4 bg-surface-0 text-chat-text text-sm resize-none outline-none placeholder-chat-text-secondary/40 leading-relaxed"
            disabled={isGenerating}
          />
          <div className="text-[10px] text-chat-text-secondary/50 px-4 pb-1.5 flex-shrink-0 text-right">
            {inputText.length > 0 ? `${inputText.split(/\s+/).filter(Boolean).length} words` : ''}
          </div>
        </div>

        {/* Output pane */}
        {hasOutput && (
          <div
            className={`flex flex-col ${
              viewMode === 'split' ? 'w-1/2' : 'flex-1'
            } min-h-0 bg-surface-1`}
          >
            {viewMode === 'split' && (
              <div className="px-3 py-1.5 border-b border-chat-border flex-shrink-0">
                <span className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">
                  Corrected
                </span>
              </div>
            )}
            <div
              ref={outputScrollRef}
              className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0"
            >
              {/* Corrected text */}
              {displayCorrected && (
                <div>
                  {viewMode === 'output' && (
                    <p className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide mb-2">
                      Corrected Text
                    </p>
                  )}
                  <p className="text-sm text-chat-text leading-relaxed whitespace-pre-wrap">
                    {displayCorrected}
                    {isGenerating && !displayChanges.length && (
                      <span className="inline-block w-1.5 h-3.5 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
                    )}
                  </p>
                </div>
              )}

              {/* Changes list */}
              {displayChanges.length > 0 && (
                <div>
                  <p className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide mb-2">
                    Changes ({displayChanges.length})
                  </p>
                  <ul className="space-y-1.5">
                    {displayChanges.map((change, i) => (
                      <li
                        key={i}
                        className="text-xs text-chat-text-secondary leading-snug pl-2 border-l-2 border-chat-accent/40"
                      >
                        {change}
                        {isGenerating && i === displayChanges.length - 1 && (
                          <span className="inline-block w-1 h-3 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Streaming indicator with no content yet */}
              {isGenerating && !displayCorrected && (
                <div className="flex items-center gap-2 text-xs text-chat-text-secondary">
                  <div className="flex gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="w-1.5 h-1.5 rounded-full bg-chat-accent animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                  Checking...
                </div>
              )}
            </div>

            {/* Output action bar */}
            {!isGenerating && corrected && (
              <div className="flex items-center gap-2 px-4 py-2.5 border-t border-chat-border flex-shrink-0">
                <button
                  onClick={handleApplyCorrections}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-chat-accent hover:opacity-90 text-white text-xs font-medium transition-opacity"
                >
                  {appliedCorrections ? (
                    <><CheckIcon className="w-3.5 h-3.5" /> Applied</>
                  ) : (
                    <><DocumentCheckIcon className="w-3.5 h-3.5" /> Apply corrections</>
                  )}
                </button>
                <button
                  onClick={handleCopyOutput}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-chat-border hover:border-chat-accent/40 text-chat-text-secondary hover:text-chat-text text-xs transition-colors"
                >
                  {copiedOutput ? (
                    <><CheckIcon className="w-3.5 h-3.5 text-green-400" /> Copied</>
                  ) : (
                    <><ClipboardDocumentIcon className="w-3.5 h-3.5" /> Copy</>
                  )}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex-shrink-0">
          {error}
        </div>
      )}

      {/* Bottom toolbar */}
      <div className="px-4 py-2.5 border-t border-chat-border flex-shrink-0 flex items-center gap-3">
        <button
          onClick={isGenerating ? handleStop : handleCheck}
          disabled={!isGenerating && !inputText.trim()}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white text-xs font-medium transition-opacity flex-shrink-0"
        >
          {isGenerating ? (
            <><StopIcon className="w-3.5 h-3.5" /> Stop</>
          ) : (
            'Check text'
          )}
        </button>

        {/* Model selector */}
        <div className="relative" ref={modelMenuRef}>
          <button
            type="button"
            onClick={() => setShowModelMenu((p) => !p)}
            className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
            style={{ background: 'var(--border-subtle)' }}
          >
            <span className="max-w-[140px] truncate">
              {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
            </span>
            <svg
              className="w-2.5 h-2.5 opacity-50"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showModelMenu && availableModels.length > 0 && (
            <div
              style={{
                position: 'absolute',
                bottom: 32,
                left: 0,
                width: 220,
                maxHeight: 280,
                background: 'var(--surface-1)',
                border: '1px solid var(--border-default)',
                borderRadius: 10,
                overflow: 'hidden',
                zIndex: 50,
              }}
            >
              <div style={{ maxHeight: 280, overflowY: 'auto', padding: 4 }}>
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
                    className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
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

        {hasOutput && !isGenerating && (
          <span className="ml-auto text-[10px] text-chat-text-secondary/60">
            {changes.length} change{changes.length !== 1 ? 's' : ''} found
          </span>
        )}
      </div>
    </div>
  );
}
