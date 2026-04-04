import { useState, useRef, useEffect, useCallback } from 'react';
import {
  ArrowsRightLeftIcon,
  ClipboardDocumentIcon,
  CheckIcon,
  StopIcon,
  LanguageIcon,
} from '@heroicons/react/24/outline';

/* ── Constants ── */
const LANGUAGES = [
  { code: 'English', label: 'English' },
  { code: 'Spanish', label: 'Spanish' },
  { code: 'French', label: 'French' },
  { code: 'German', label: 'German' },
  { code: 'Chinese', label: 'Chinese' },
  { code: 'Japanese', label: 'Japanese' },
  { code: 'Korean', label: 'Korean' },
  { code: 'Arabic', label: 'Arabic' },
  { code: 'Russian', label: 'Russian' },
  { code: 'Portuguese', label: 'Portuguese' },
  { code: 'Italian', label: 'Italian' },
  { code: 'Turkish', label: 'Turkish' },
  { code: 'Azerbaijani', label: 'Azerbaijani' },
  { code: 'Hindi', label: 'Hindi' },
];

const AUTO_DETECT = 'Auto-detect';

function buildSystemPrompt(sourceLang: string, targetLang: string): string {
  if (sourceLang === AUTO_DETECT) {
    return `Detect the language of the following text and translate it to ${targetLang}. Output ONLY the translation.`;
  }
  return `You are a professional translator. Translate the following text from ${sourceLang} to ${targetLang}. Output ONLY the translation, no explanations or notes. Preserve the original formatting, tone, and meaning as closely as possible.`;
}

const DRAFT_KEY = 'aura-draft-translate';

/* ── Component ── */
export function TranslatePanel() {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState(AUTO_DETECT);
  const [targetLang, setTargetLang] = useState('English');
  const [isTranslating, setIsTranslating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);
  const [draftLoaded, setDraftLoaded] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);

  // Load draft on mount
  useEffect(() => {
    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) {
        const parsed = JSON.parse(draft);
        if (parsed.sourceText) setSourceText(parsed.sourceText);
        if (parsed.translatedText) setTranslatedText(parsed.translatedText);
        if (parsed.sourceLang) setSourceLang(parsed.sourceLang);
        if (parsed.targetLang) setTargetLang(parsed.targetLang);
        setDraftLoaded(true);
      }
    } catch {}
  }, []);

  // Auto-save on change
  useEffect(() => {
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(DRAFT_KEY, JSON.stringify({ sourceText, translatedText, sourceLang, targetLang, timestamp: Date.now() }));
      } catch {}
    }, 5000);
    return () => clearTimeout(timer);
  }, [sourceText, translatedText, sourceLang, targetLang]);

  // Fetch available models
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

  // Abort in-flight request on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  const handleTranslate = useCallback(async () => {
    if (!sourceText.trim() || isTranslating) return;

    setIsTranslating(true);
    setError(null);
    setTranslatedText('');

    const controller = new AbortController();
    abortRef.current = controller;

    const systemPrompt = buildSystemPrompt(sourceLang, targetLang);

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: sourceText,
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
                  setTranslatedText(fullResponse);
                }
              } catch {
                fullResponse += data;
                setTranslatedText(fullResponse);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullResponse += line;
              setTranslatedText(fullResponse);
            }
          }
        }
      } else {
        const text = await res.text();
        setTranslatedText(text);
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message || 'Translation failed. Make sure the backend is running.');
      }
    } finally {
      setIsTranslating(false);
      abortRef.current = null;
    }
  }, [sourceText, sourceLang, targetLang, selectedModel, isTranslating]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    setIsTranslating(false);
  }, []);

  const handleSwap = useCallback(() => {
    // Can't swap when source is Auto-detect
    if (sourceLang === AUTO_DETECT) return;
    const prevSource = sourceLang;
    const prevTarget = targetLang;
    setSourceLang(prevTarget);
    setTargetLang(prevSource);
    setSourceText(translatedText);
    setTranslatedText(sourceText);
  }, [sourceLang, targetLang, sourceText, translatedText]);

  const handleCopy = useCallback(() => {
    if (!translatedText) return;
    navigator.clipboard.writeText(translatedText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [translatedText]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleTranslate();
    }
  }, [handleTranslate]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-chat-border flex-shrink-0">
        <LanguageIcon className="w-4 h-4 text-chat-accent flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-chat-text">Translate</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Ctrl+Enter to translate</p>
        </div>

        {/* Model selector */}
        <div className="relative flex-shrink-0" ref={modelMenuRef}>
          <button
            type="button"
            onClick={() => setShowModelMenu(p => !p)}
            className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md"
            style={{ background: 'var(--border-subtle)' }}
          >
            <span className="max-w-[120px] truncate">{selectedModel ? selectedModel.split('/').pop() : 'Auto'}</span>
            <svg className="w-2.5 h-2.5 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {showModelMenu && availableModels.length > 0 && (
            <div
              style={{
                position: 'absolute', top: 28, right: 0, width: 220, maxHeight: 280,
                background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                borderRadius: 10, overflow: 'hidden', zIndex: 50,
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
      </div>

      {/* Draft recovered banner */}
      {draftLoaded && (
        <div className="flex items-center gap-2 px-3 py-1.5 mx-4 mt-2 bg-blue-500/10 text-blue-400 text-[10px] rounded-lg flex-shrink-0">
          <span className="flex-1">Draft recovered</span>
          <button onClick={() => { setDraftLoaded(false); try { localStorage.removeItem(DRAFT_KEY); } catch {} }} className="hover:text-blue-300 transition-colors">Dismiss</button>
        </div>
      )}

      {/* Language bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-chat-border flex-shrink-0 bg-surface-1">
        {/* Source language */}
        <select
          value={sourceLang}
          onChange={(e) => setSourceLang(e.target.value)}
          className="flex-1 min-w-0 text-xs rounded-lg px-2 py-1.5 bg-surface-2 border border-chat-border text-chat-text outline-none focus:border-chat-accent transition-colors cursor-pointer"
        >
          <option value={AUTO_DETECT}>Auto-detect</option>
          {LANGUAGES.map(l => (
            <option key={l.code} value={l.code}>{l.label}</option>
          ))}
        </select>

        {/* Swap button */}
        <button
          onClick={handleSwap}
          disabled={sourceLang === AUTO_DETECT}
          title={sourceLang === AUTO_DETECT ? 'Cannot swap while Auto-detect is active' : 'Swap languages'}
          className="flex-shrink-0 p-1.5 rounded-lg text-chat-text-secondary hover:text-chat-text hover:bg-surface-3 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
        >
          <ArrowsRightLeftIcon className="w-3.5 h-3.5" />
        </button>

        {/* Target language */}
        <select
          value={targetLang}
          onChange={(e) => setTargetLang(e.target.value)}
          className="flex-1 min-w-0 text-xs rounded-lg px-2 py-1.5 bg-surface-2 border border-chat-border text-chat-text outline-none focus:border-chat-accent transition-colors cursor-pointer"
        >
          {LANGUAGES.map(l => (
            <option key={l.code} value={l.code}>{l.label}</option>
          ))}
        </select>
      </div>

      {/* Main panels — side by side on desktop, stacked on mobile */}
      <div className="flex-1 overflow-hidden flex flex-col md:flex-row min-h-0">
        {/* Source panel */}
        <div className="flex-1 flex flex-col min-h-0 border-b md:border-b-0 md:border-r border-chat-border">
          <textarea
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type or paste text to translate..."
            className="flex-1 w-full p-4 bg-transparent text-chat-text text-sm resize-none outline-none placeholder-chat-text-secondary/40 leading-relaxed"
            spellCheck={false}
          />
          {/* Source footer */}
          <div className="flex items-center justify-between px-4 py-2 border-t border-chat-border flex-shrink-0">
            <span className="text-[10px] text-chat-text-secondary tabular-nums">
              {sourceText.length.toLocaleString()} chars
            </span>
            <button
              onClick={isTranslating ? handleStop : handleTranslate}
              disabled={!isTranslating && !sourceText.trim()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all disabled:opacity-40"
              style={{
                background: isTranslating ? 'var(--surface-3)' : 'var(--chat-accent)',
                color: 'white',
              }}
            >
              {isTranslating ? (
                <>
                  <StopIcon className="w-3 h-3" />
                  Stop
                </>
              ) : (
                <>
                  <LanguageIcon className="w-3 h-3" />
                  Translate
                </>
              )}
            </button>
          </div>
        </div>

        {/* Translation panel */}
        <div className="flex-1 flex flex-col min-h-0 bg-surface-1">
          <div className="flex-1 relative overflow-y-auto">
            {error ? (
              <p className="p-4 text-xs text-red-400 leading-relaxed">{error}</p>
            ) : translatedText ? (
              <p className="p-4 text-sm text-chat-text leading-relaxed whitespace-pre-wrap">
                {translatedText}
                {isTranslating && (
                  <span className="inline-block w-1.5 h-3.5 bg-chat-accent animate-pulse ml-0.5 align-middle rounded-sm" />
                )}
              </p>
            ) : (
              <div className="flex items-center justify-center h-full min-h-[80px]">
                {isTranslating ? (
                  <div className="flex items-center gap-2 text-xs text-chat-accent">
                    <div className="shimmer-bar h-2 w-20" />
                    Translating...
                  </div>
                ) : (
                  <p className="text-xs text-chat-text-secondary/50 px-4 text-center">
                    Translation will appear here
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Translation footer */}
          <div className="flex items-center justify-between px-4 py-2 border-t border-chat-border flex-shrink-0">
            <span className="text-[10px] text-chat-text-secondary tabular-nums">
              {translatedText.length.toLocaleString()} chars
            </span>
            <button
              onClick={handleCopy}
              disabled={!translatedText}
              title="Copy translation"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] text-chat-text-secondary hover:text-chat-text hover:bg-surface-3 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              {copied ? (
                <>
                  <CheckIcon className="w-3 h-3 text-green-400" />
                  <span className="text-green-400">Copied</span>
                </>
              ) : (
                <>
                  <ClipboardDocumentIcon className="w-3 h-3" />
                  Copy
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
