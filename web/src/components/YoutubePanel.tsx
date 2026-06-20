import { useState, useRef, useEffect, useCallback } from 'react';
import { StopIcon, ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type InputMode = 'url' | 'transcript';
type Action = 'summarize' | 'key_moments' | 'qa' | 'study_notes';

interface TranscriptEntry {
  text: string;
  start: number;
  duration: number;
}

const ACTIONS: { value: Action; label: string }[] = [
  { value: 'summarize', label: 'Summarize' },
  { value: 'key_moments', label: 'Key Moments' },
  { value: 'qa', label: 'Q&A' },
  { value: 'study_notes', label: 'Study Notes' },
];

function extractVideoId(url: string): string | null {
  const match = url.match(
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})/
  );
  return match ? match[1] : null;
}

function formatTimestamp(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function entriesToText(entries: TranscriptEntry[]): string {
  return entries.map((e) => `[${formatTimestamp(e.start)}] ${e.text}`).join('\n');
}

function buildSystemPrompt(action: Action): string {
  switch (action) {
    case 'summarize':
      return 'Summarize this video transcript concisely. Cover the main topics, key arguments, and conclusions.';
    case 'key_moments':
      return 'Extract the key moments and timestamps from this transcript as a numbered list. Include the most important quotes and turning points.';
    case 'study_notes':
      return 'Create comprehensive study notes from this video transcript. Include: main topics, key concepts, definitions, examples, and action items. Format with headers and bullet points.';
    case 'qa':
      return ''; // handled inline
  }
}

function buildUserMessage(action: Action, content: string, question: string): string {
  if (action === 'qa') {
    return `Based on this video transcript, answer the question. Quote relevant parts.\n\nTranscript:\n${content}\n\nQuestion: ${question}`;
  }
  return content;
}

/* ── Main Component ── */
export function YoutubePanel() {
  const [inputMode, setInputMode] = useState<InputMode>('url');
  const [urlInput, setUrlInput] = useState('');
  const [transcript, setTranscript] = useState('');
  const [transcriptEntries, setTranscriptEntries] = useState<TranscriptEntry[]>([]);
  const [action, setAction] = useState<Action>('summarize');
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState('');
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [showModelMenu, setShowModelMenu] = useState(false);

  // Track current seek time from timestamp clicks
  const [seekTime, setSeekTime] = useState<number | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const resultEndRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const videoId = extractVideoId(urlInput);

  // Fetch models
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

  // Auto-scroll output during streaming
  useEffect(() => {
    if (isGenerating && resultEndRef.current) {
      resultEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [result, isGenerating]);

  // Abort on unmount
  useEffect(() => {
    return () => { abortRef.current?.abort(); };
  }, []);

  // Reset fetch error when video ID changes
  useEffect(() => {
    setFetchError('');
  }, [videoId]);

  // Handle timestamp seek via iframe postMessage
  useEffect(() => {
    if (seekTime === null || !iframeRef.current) return;
    // YouTube's iframe API doesn't support postMessage seeking without JS API init.
    // Reload the iframe src with start parameter as the simplest reliable approach.
    const src = `https://www.youtube.com/embed/${videoId}?start=${Math.floor(seekTime)}&autoplay=1`;
    iframeRef.current.src = src;
    setSeekTime(null);
  }, [seekTime, videoId]);

  const handleFetchTranscript = useCallback(async () => {
    if (!videoId) return;
    setIsFetching(true);
    setFetchError('');
    try {
      const res = await fetch('/api/youtube/transcript', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, language: 'en' }),
      });
      const data = await res.json();
      if (!res.ok) {
        setFetchError(data.detail || `Error ${res.status}`);
        return;
      }
      const entries: TranscriptEntry[] = data.transcript || [];
      setTranscriptEntries(entries);
      setTranscript(entriesToText(entries));
    } catch (e: any) {
      setFetchError(`Network error: ${e.message}`);
    } finally {
      setIsFetching(false);
    }
  }, [videoId]);

  const handleAnalyze = useCallback(async () => {
    const content = transcript.trim();
    if (!content) {
      setError('Please paste a transcript before analyzing.');
      return;
    }
    if (action === 'qa' && !question.trim()) {
      setError('Please enter a question for Q&A mode.');
      return;
    }

    setError('');
    setResult('');
    setIsGenerating(true);

    const systemPrompt = buildSystemPrompt(action);
    const userMessage = buildUserMessage(action, content, question.trim());

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          ...(systemPrompt ? { system_prompt: systemPrompt } : {}),
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
        setError(`Error: ${e.message}. Make sure the backend is running.`);
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [transcript, action, question, selectedModel]);

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

  // Render transcript lines with clickable timestamps (only when entries available)
  const transcriptDisplay = transcriptEntries.length > 0 ? (
    <div
      className="w-full px-3 py-2 rounded-lg text-xs font-mono overflow-y-auto resize-none"
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        color: 'var(--text-primary)',
        maxHeight: inputMode === 'url' ? '140px' : '280px',
        minHeight: '80px',
      }}
    >
      {transcriptEntries.map((entry, i) => (
        <div key={i} className="flex gap-2 mb-0.5 leading-relaxed">
          <button
            onClick={() => setSeekTime(entry.start)}
            className="flex-shrink-0 font-mono text-[10px] px-1 rounded hover:opacity-80 transition-opacity"
            style={{
              color: 'var(--accent)',
              background: 'transparent',
              cursor: videoId ? 'pointer' : 'default',
            }}
            title={videoId ? `Seek to ${formatTimestamp(entry.start)}` : undefined}
          >
            [{formatTimestamp(entry.start)}]
          </button>
          <span style={{ color: 'var(--text-primary)' }}>{entry.text}</span>
        </div>
      ))}
    </div>
  ) : (
    <textarea
      value={transcript}
      onChange={(e) => setTranscript(e.target.value)}
      placeholder="Paste the video transcript here..."
      rows={inputMode === 'url' ? 5 : 10}
      className="w-full px-3 py-2 rounded-lg text-sm resize-none outline-none transition-colors"
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        color: 'var(--text-primary)',
      }}
    />
  );

  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div
        className="px-4 py-3 border-b flex-shrink-0 flex items-center gap-3"
        style={{ borderColor: 'var(--border-default)' }}
      >
        <div>
          <h2 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            YouTube Analysis
          </h2>
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-secondary)' }}>
            Analyze YouTube videos via transcript
          </p>
        </div>
      </div>

      {/* Body: two columns on md+ */}
      <div className="flex flex-col md:flex-row flex-1 min-h-0 overflow-hidden">
        {/* Left: input controls */}
        <div
          className="flex flex-col md:w-[380px] md:min-w-[280px] flex-shrink-0 border-b md:border-b-0 md:border-r overflow-y-auto"
          style={{ borderColor: 'var(--border-default)' }}
        >
          <div className="p-4 space-y-4">
            {/* Mode toggle */}
            <div
              className="flex rounded-lg border overflow-hidden text-xs"
              style={{ borderColor: 'var(--border-default)' }}
            >
              {(['url', 'transcript'] as InputMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setInputMode(m)}
                  className="flex-1 py-1.5 capitalize transition-colors"
                  style={{
                    background: inputMode === m ? 'var(--accent)' : 'transparent',
                    color: inputMode === m ? '#fff' : 'var(--text-secondary)',
                  }}
                >
                  {m === 'url' ? 'URL + Player' : 'Transcript Only'}
                </button>
              ))}
            </div>

            {/* URL input */}
            {inputMode === 'url' && (
              <div className="space-y-3">
                <input
                  type="text"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  placeholder="https://youtube.com/watch?v=..."
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none transition-colors"
                  style={{
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    color: 'var(--text-primary)',
                  }}
                />

                {/* Fetch Transcript button — shown when valid video ID */}
                {videoId && (
                  <button
                    onClick={handleFetchTranscript}
                    disabled={isFetching}
                    className="w-full py-1.5 rounded-lg text-xs font-medium transition-opacity flex items-center justify-center gap-2"
                    style={{
                      background: 'var(--surface-2)',
                      border: '1px solid var(--border-default)',
                      color: 'var(--text-primary)',
                      opacity: isFetching ? 0.6 : 1,
                      cursor: isFetching ? 'not-allowed' : 'pointer',
                    }}
                  >
                    {isFetching ? (
                      <>
                        <span
                          className="inline-block w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin"
                        />
                        Fetching transcript...
                      </>
                    ) : (
                      'Fetch Transcript'
                    )}
                  </button>
                )}

                {/* Fetch error */}
                {fetchError && (
                  <p
                    className="text-xs px-3 py-2 rounded-lg"
                    style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171' }}
                  >
                    {fetchError}
                    <span className="block mt-1 opacity-70">You can still paste a transcript manually below.</span>
                  </p>
                )}

                {/* Embedded player */}
                {videoId ? (
                  <div className="rounded-lg overflow-hidden aspect-video">
                    <iframe
                      ref={iframeRef}
                      src={`https://www.youtube.com/embed/${videoId}`}
                      title="YouTube video player"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                      allowFullScreen
                      className="w-full h-full border-none"
                    />
                  </div>
                ) : (
                  <div
                    className="rounded-lg aspect-video flex items-center justify-center text-xs text-center px-4"
                    style={{ background: 'var(--surface-1)', color: 'var(--text-secondary)' }}
                  >
                    Enter a YouTube URL to preview the video
                  </div>
                )}
              </div>
            )}

            {/* Transcript section */}
            <div className="space-y-1">
              <label
                className="text-[11px] font-medium"
                style={{ color: 'var(--text-secondary)' }}
              >
                Transcript
              </label>
              {transcriptDisplay}
              {transcript && (
                <p className="text-[10px]" style={{ color: 'var(--text-secondary)' }}>
                  {transcript.split(/\s+/).filter(Boolean).length.toLocaleString()} words
                  {transcriptEntries.length > 0 && ' · Click timestamps to seek'}
                </p>
              )}
              {transcriptEntries.length > 0 && (
                <button
                  onClick={() => {
                    setTranscriptEntries([]);
                    setTranscript('');
                  }}
                  className="text-[10px] underline"
                  style={{ color: 'var(--text-secondary)', background: 'transparent' }}
                >
                  Clear & paste manually
                </button>
              )}
            </div>

            {/* Action selector */}
            <div className="space-y-1">
              <label
                className="text-[11px] font-medium"
                style={{ color: 'var(--text-secondary)' }}
              >
                Action
              </label>
              <div
                className="grid grid-cols-2 gap-1.5 rounded-lg border p-1"
                style={{ borderColor: 'var(--border-default)', background: 'var(--surface-1)' }}
              >
                {ACTIONS.map((a) => (
                  <button
                    key={a.value}
                    onClick={() => setAction(a.value)}
                    className="py-1.5 rounded-md text-xs font-medium transition-colors"
                    style={{
                      background: action === a.value ? 'var(--accent)' : 'transparent',
                      color: action === a.value ? '#fff' : 'var(--text-secondary)',
                    }}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Q&A question input */}
            {action === 'qa' && (
              <div className="space-y-1">
                <label
                  className="text-[11px] font-medium"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  Your Question
                </label>
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="What is the main argument of the video?"
                  className="w-full px-3 py-2 rounded-lg text-sm outline-none transition-colors"
                  style={{
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    color: 'var(--text-primary)',
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !isGenerating) handleAnalyze();
                  }}
                />
              </div>
            )}

            {/* Model selector */}
            <div ref={modelMenuRef} className="relative">
              <label
                className="text-[11px] font-medium block mb-1"
                style={{ color: 'var(--text-secondary)' }}
              >
                Model
              </label>
              <button
                type="button"
                onClick={() => setShowModelMenu((p) => !p)}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-md w-full"
                style={{
                  background: 'var(--surface-1)',
                  border: '1px solid var(--border-default)',
                  color: 'var(--text-secondary)',
                }}
              >
                <span className="flex-1 text-left truncate">
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto (recommended)'}
                </span>
                <svg
                  className="w-3 h-3 opacity-50 flex-shrink-0"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </button>

              {showModelMenu && availableModels.length > 0 && (
                <div
                  className="absolute bottom-full mb-1 left-0 w-full max-h-52 overflow-y-auto rounded-lg z-50"
                  style={{
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                  }}
                >
                  <div className="p-1 space-y-0.5">
                    <button
                      onClick={() => { setSelectedModel(null); setShowModelMenu(false); }}
                      className="w-full text-left px-2.5 py-1.5 rounded-md text-xs transition-colors"
                      style={{
                        background: !selectedModel ? 'var(--surface-3)' : 'transparent',
                        color: !selectedModel ? 'var(--text-primary)' : 'var(--text-secondary)',
                      }}
                    >
                      Auto (recommended)
                    </button>
                    {availableModels.map((m) => (
                      <button
                        key={m}
                        onClick={() => { setSelectedModel(m); setShowModelMenu(false); }}
                        className="w-full text-left px-2.5 py-1.5 rounded-md text-xs truncate transition-colors"
                        style={{
                          background: selectedModel === m ? 'var(--surface-3)' : 'transparent',
                          color: selectedModel === m ? 'var(--text-primary)' : 'var(--text-secondary)',
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <p
                className="text-xs px-3 py-2 rounded-lg"
                style={{ background: 'rgba(239,68,68,0.1)', color: '#f87171' }}
              >
                {error}
              </p>
            )}

            {/* Analyze / Stop button */}
            <button
              onClick={isGenerating ? handleStop : handleAnalyze}
              disabled={!isGenerating && !transcript.trim()}
              className="w-full py-2 rounded-lg text-sm font-medium transition-opacity flex items-center justify-center gap-2"
              style={{
                background: isGenerating ? 'var(--surface-2)' : 'var(--accent)',
                color: isGenerating ? 'var(--text-secondary)' : '#fff',
                opacity: (!isGenerating && !transcript.trim()) ? 0.4 : 1,
                cursor: (!isGenerating && !transcript.trim()) ? 'not-allowed' : 'pointer',
              }}
            >
              {isGenerating ? (
                <>
                  <StopIcon className="w-4 h-4" />
                  Stop
                </>
              ) : (
                <>
                  Analyze
                </>
              )}
            </button>
          </div>
        </div>

        {/* Right: streaming output */}
        <div className="flex-1 flex flex-col min-h-0 min-w-0">
          {/* Output toolbar */}
          <div
            className="flex items-center justify-between px-4 py-2 border-b flex-shrink-0"
            style={{ borderColor: 'var(--border-default)' }}
          >
            <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>
              {isGenerating ? (
                <span className="flex items-center gap-2">
                  <span
                    className="inline-block w-1.5 h-1.5 rounded-full animate-pulse"
                    style={{ background: 'var(--accent)' }}
                  />
                  Analyzing...
                </span>
              ) : result ? (
                'Result'
              ) : (
                'Output will appear here'
              )}
            </span>
            {result && (
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md transition-colors"
                style={{
                  background: copied ? 'rgba(34,197,94,0.15)' : 'var(--surface-1)',
                  color: copied ? '#4ade80' : 'var(--text-secondary)',
                  border: '1px solid var(--border-default)',
                }}
              >
                {copied ? (
                  <>
                    <CheckIcon className="w-3.5 h-3.5" />
                    Copied
                  </>
                ) : (
                  <>
                    <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                    Copy
                  </>
                )}
              </button>
            )}
          </div>

          {/* Output body */}
          <div className="flex-1 overflow-y-auto p-4">
            {result ? (
              <div className="relative">
                <pre
                  className="text-sm whitespace-pre-wrap leading-relaxed font-sans"
                  style={{ color: 'var(--text-primary)' }}
                >
                  {result}
                  {isGenerating && (
                    <span
                      className="inline-block w-1.5 h-4 align-middle animate-pulse ml-0.5"
                      style={{ background: 'var(--accent)' }}
                    />
                  )}
                </pre>
                <div ref={resultEndRef} />
              </div>
            ) : (
              <div
                className="flex flex-col items-center justify-center h-full gap-3 text-center"
                style={{ color: 'var(--text-secondary)' }}
              >
                <svg
                  className="w-10 h-10 opacity-20"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-2.88 2.5 2.89 2.89 0 01-2.89-2.89 2.89 2.89 0 012.89-2.89c.28 0 .54.04.79.1V9.01a6.27 6.27 0 00-.79-.05 6.34 6.34 0 00-6.34 6.34 6.34 6.34 0 006.34 6.34 6.34 6.34 0 006.33-6.34V8.69a8.18 8.18 0 004.78 1.52V6.76a4.85 4.85 0 01-1.01-.07z" />
                </svg>
                <div>
                  <p className="text-sm">Paste a transcript and click Analyze</p>
                  <p className="text-xs mt-1 opacity-60">
                    Supports Summarize, Key Moments, Q&A, and Study Notes
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
