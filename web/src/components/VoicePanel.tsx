import { useState, useRef, useEffect, useCallback } from 'react';
import { StopIcon, ClipboardDocumentIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type SpeakState = 'idle' | 'speaking' | 'paused';

/* ── Main Component ── */
export function VoicePanel() {
  const [text, setText] = useState('');
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoiceURI, setSelectedVoiceURI] = useState<string>('');
  const [speed, setSpeed] = useState(1);
  const [pitch, setPitch] = useState(1);
  const [speakState, setSpeakState] = useState<SpeakState>('idle');
  const [currentWordIndex, setCurrentWordIndex] = useState(-1);
  const [supported, setSupported] = useState(true);
  const [error, setError] = useState('');

  // Generate feature
  const [generatePrompt, setGeneratePrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const wordsRef = useRef<string[]>([]);

  // Check support
  useEffect(() => {
    if (!('speechSynthesis' in window)) {
      setSupported(false);
    }
  }, []);

  // Load voices — they load async in most browsers
  useEffect(() => {
    if (!supported) return;
    const load = () => {
      const v = window.speechSynthesis.getVoices();
      if (v.length > 0) {
        setVoices(v);
        setSelectedVoiceURI(v[0].voiceURI);
      }
    };
    load();
    window.speechSynthesis.onvoiceschanged = load;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, [supported]);

  // Fetch models
  useEffect(() => {
    apiFetch('/api/models')
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

  // Cancel utterance on unmount
  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
      abortRef.current?.abort();
    };
  }, []);

  const handleSpeak = useCallback(() => {
    if (!supported || !text.trim()) return;
    setError('');

    window.speechSynthesis.cancel();
    setCurrentWordIndex(-1);

    const words = text.trim().split(/\s+/);
    wordsRef.current = words;

    const utterance = new SpeechSynthesisUtterance(text);
    const voice = voices.find(v => v.voiceURI === selectedVoiceURI);
    if (voice) utterance.voice = voice;
    utterance.rate = speed;
    utterance.pitch = pitch;

    utterance.onboundary = (e) => {
      if (e.name === 'word') {
        // Count words up to charIndex
        const spoken = text.slice(0, e.charIndex);
        const idx = spoken.trim().split(/\s+/).filter(Boolean).length;
        setCurrentWordIndex(idx);
      }
    };

    utterance.onstart = () => setSpeakState('speaking');
    utterance.onpause = () => setSpeakState('paused');
    utterance.onresume = () => setSpeakState('speaking');
    utterance.onend = () => { setSpeakState('idle'); setCurrentWordIndex(-1); };
    utterance.onerror = (e) => {
      // 'interrupted' fires when we cancel programmatically — not a real error
      if (e.error !== 'interrupted' && e.error !== 'canceled') {
        setError(`Speech error: ${e.error}`);
      }
      setSpeakState('idle');
      setCurrentWordIndex(-1);
    };

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    setSpeakState('speaking');
  }, [supported, text, voices, selectedVoiceURI, speed, pitch]);

  const handlePause = useCallback(() => {
    if (speakState === 'speaking') {
      window.speechSynthesis.pause();
      setSpeakState('paused');
    } else if (speakState === 'paused') {
      window.speechSynthesis.resume();
      setSpeakState('speaking');
    }
  }, [speakState]);

  const handleStop = useCallback(() => {
    window.speechSynthesis.cancel();
    setSpeakState('idle');
    setCurrentWordIndex(-1);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!generatePrompt.trim() || isGenerating) return;
    setIsGenerating(true);
    setError('');

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: `Write a clear, natural-sounding passage to be read aloud. Topic: ${generatePrompt.trim()}. Output only the text, no headings, no markdown.`,
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let fullText = '';

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
                const t = parsed.choices?.[0]?.delta?.content || parsed.content || parsed.chunk || '';
                if (t) { fullText += t; setText(fullText); }
              } catch {
                fullText += data;
                setText(fullText);
              }
            } else if (line.trim() && !line.startsWith(':')) {
              fullText += line;
              setText(fullText);
            }
          }
        }
      } else {
        const t = await res.text();
        setText(t.trim());
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(`Generation failed: ${e.message}`);
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [generatePrompt, isGenerating, selectedModel]);

  // Render text with highlighted current word
  const renderHighlightedText = () => {
    if (currentWordIndex < 0 || speakState === 'idle') return null;
    const words = wordsRef.current;
    return (
      <div
        className="absolute inset-0 p-3 text-sm font-mono leading-relaxed pointer-events-none overflow-auto"
        style={{ color: 'transparent', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
      >
        {words.map((word, i) => (
          <span key={i}>
            <span
              style={{
                background: i === currentWordIndex ? 'var(--chat-accent, #7c3aed)' : 'transparent',
                color: i === currentWordIndex ? 'white' : 'transparent',
                borderRadius: 3,
                padding: i === currentWordIndex ? '0 2px' : undefined,
              }}
            >
              {word}
            </span>
            {i < words.length - 1 ? ' ' : ''}
          </span>
        ))}
      </div>
    );
  };

  if (!supported) {
    return (
      <div className="flex items-center justify-center h-full text-chat-text-secondary text-sm">
        <div className="text-center">
          <p className="text-2xl mb-2">🔇</p>
          <p className="font-medium text-chat-text">Speech synthesis not supported</p>
          <p className="text-xs mt-1">Try Chrome, Edge, or Safari</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Left: Controls */}
      <div
        className="flex flex-col md:w-[320px] md:min-w-[260px] md:border-r border-b md:border-b-0 border-chat-border flex-shrink-0 max-md:max-h-[45vh] bg-surface-0"
      >
        {/* Header */}
        <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
          <h2 className="text-sm font-semibold text-chat-text">Voice Panel</h2>
          <p className="text-[10px] text-chat-text-secondary mt-0.5">Text-to-speech with browser voices</p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Voice selector */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">Voice</label>
            <select
              value={selectedVoiceURI}
              onChange={e => setSelectedVoiceURI(e.target.value)}
              className="w-full p-2 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-xs outline-none focus:border-chat-accent"
              disabled={speakState !== 'idle'}
            >
              {voices.length === 0 && (
                <option value="">Loading voices...</option>
              )}
              {voices.map(v => (
                <option key={v.voiceURI} value={v.voiceURI}>
                  {v.name} {v.lang ? `(${v.lang})` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Speed */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">Speed</label>
              <span className="text-[10px] text-chat-text font-mono">{speed.toFixed(1)}x</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.1}
              value={speed}
              onChange={e => setSpeed(parseFloat(e.target.value))}
              className="w-full accent-chat-accent"
              disabled={speakState !== 'idle'}
            />
            <div className="flex justify-between text-[9px] text-chat-text-secondary">
              <span>0.5x</span><span>1x</span><span>2x</span>
            </div>
          </div>

          {/* Pitch */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">Pitch</label>
              <span className="text-[10px] text-chat-text font-mono">{pitch.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.1}
              value={pitch}
              onChange={e => setPitch(parseFloat(e.target.value))}
              className="w-full accent-chat-accent"
              disabled={speakState !== 'idle'}
            />
            <div className="flex justify-between text-[9px] text-chat-text-secondary">
              <span>Low</span><span>Normal</span><span>High</span>
            </div>
          </div>

          {/* Divider */}
          <div className="border-t border-chat-border" />

          {/* Generate text */}
          <div className="space-y-2">
            <label className="text-[10px] font-medium text-chat-text-secondary uppercase tracking-wide">
              Generate text first
            </label>
            <textarea
              value={generatePrompt}
              onChange={e => setGeneratePrompt(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleGenerate(); }
              }}
              placeholder="Describe what to generate, e.g. 'a short poem about the ocean'"
              className="w-full p-2.5 rounded-lg bg-surface-1 border border-chat-border text-chat-text text-xs resize-none outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
              rows={2}
              disabled={isGenerating}
            />

            {/* Model selector + Generate button row */}
            <div className="flex items-center gap-2">
              {/* Model picker */}
              <div className="relative flex-1" ref={modelMenuRef}>
                <button
                  type="button"
                  onClick={() => setShowModelMenu(p => !p)}
                  className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md w-full truncate"
                  style={{ background: 'var(--border-subtle)' }}
                >
                  <span className="flex-1 text-left truncate">
                    {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                  </span>
                  <svg className="w-2.5 h-2.5 opacity-50 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {showModelMenu && availableModels.length > 0 && (
                  <div
                    style={{
                      position: 'absolute', bottom: 28, left: 0, width: 220, maxHeight: 260,
                      background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                      borderRadius: 10, overflow: 'hidden', zIndex: 50,
                    }}
                  >
                    <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
                      <button
                        onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                        className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors"
                        style={{ color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)', background: !selectedModel ? 'var(--surface-3)' : 'transparent' }}
                      >
                        Auto (recommended)
                      </button>
                      {availableModels.map(m => (
                        <button
                          key={m}
                          onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                          className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs text-left transition-colors truncate"
                          style={{ color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)', background: selectedModel === m ? 'var(--surface-3)' : 'transparent' }}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Generate button */}
              <button
                onClick={isGenerating ? () => abortRef.current?.abort() : handleGenerate}
                disabled={!isGenerating && !generatePrompt.trim()}
                className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white text-[10px] font-medium transition-opacity flex-shrink-0"
              >
                {isGenerating
                  ? <><StopIcon className="w-3 h-3" /> Stop</>
                  : 'Generate'
                }
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right: Text area + playback */}
      <div className="flex-1 flex flex-col min-w-0 bg-surface-0">
        {/* Playback toolbar */}
        <div className="flex items-center gap-2 px-4 py-2.5 border-b border-chat-border flex-shrink-0">
          {/* Play/Speak */}
          <button
            onClick={speakState === 'idle' ? handleSpeak : handlePause}
            disabled={!text.trim() || voices.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-chat-accent hover:opacity-90 disabled:opacity-40 text-white text-xs font-medium transition-opacity"
          >
            {speakState === 'speaking' ? (
              <>
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" rx="1" />
                  <rect x="14" y="4" width="4" height="16" rx="1" />
                </svg>
                Pause
              </>
            ) : speakState === 'paused' ? (
              <>
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
                Resume
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
                Speak
              </>
            )}
          </button>

          {/* Stop */}
          <button
            onClick={handleStop}
            disabled={speakState === 'idle'}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-chat-border hover:border-red-500/40 hover:text-red-400 disabled:opacity-30 text-chat-text-secondary text-xs font-medium transition-colors"
          >
            <StopIcon className="w-3.5 h-3.5" />
            Stop
          </button>

          {/* Status indicator */}
          <div className="flex items-center gap-2 ml-2">
            {speakState === 'speaking' && (
              <span className="flex items-center gap-1.5 text-xs text-green-400">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
                </span>
                Speaking
              </span>
            )}
            {speakState === 'paused' && (
              <span className="flex items-center gap-1.5 text-xs text-yellow-400">
                <span className="h-2 w-2 rounded-full bg-yellow-400" />
                Paused
              </span>
            )}
            {isGenerating && (
              <span className="flex items-center gap-1.5 text-xs text-purple-400">
                <div className="shimmer-bar h-2 w-12" />
                Generating...
              </span>
            )}
          </div>

          <div className="flex-1" />

          {/* Char count + copy */}
          <span className="text-[10px] text-chat-text-secondary">
            {text.length} chars · {text.trim() ? text.trim().split(/\s+/).length : 0} words
          </span>
          {text.trim() && (
            <button
              onClick={() => { navigator.clipboard.writeText(text); }}
              className="text-chat-text-secondary hover:text-chat-text p-1 rounded transition-colors"
              title="Copy text"
            >
              <ClipboardDocumentIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Text area */}
        <div className="flex-1 relative overflow-hidden">
          <textarea
            value={text}
            onChange={e => {
              setText(e.target.value);
              if (speakState !== 'idle') handleStop();
            }}
            placeholder="Type or paste text here, or use Generate above to create something to read..."
            className="absolute inset-0 w-full h-full p-4 bg-transparent text-chat-text text-sm font-mono leading-relaxed resize-none outline-none placeholder-chat-text-secondary/40"
            style={{ zIndex: 1 }}
          />
          {/* Word highlight overlay — rendered behind textarea interaction but visible */}
          {speakState !== 'idle' && currentWordIndex >= 0 && (
            <div className="absolute inset-0 p-4 text-sm font-mono leading-relaxed pointer-events-none overflow-hidden" style={{ zIndex: 0 }}>
              {renderHighlightedText()}
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="px-4 py-2 bg-red-500/10 border-t border-red-500/20 text-xs text-red-400">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
