import { useState, useRef, useEffect, useCallback } from 'react';
import {
  MicrophoneIcon,
  StopIcon,
  PlayIcon,
  PauseIcon,
  ArrowDownTrayIcon,
  ClipboardDocumentIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

/* ── Types ── */
type ProcessAction = 'cleanup' | 'summarize' | 'actions';

const PROCESS_PROMPTS: Record<ProcessAction, string> = {
  cleanup:
    'Clean up this transcript. Fix grammar, remove filler words (um, uh, like), add proper punctuation, and organize into paragraphs. Preserve the original meaning.',
  summarize:
    'Summarize this transcript concisely. Cover the main topics and key points.',
  actions:
    'Extract action items, decisions, and next steps from this transcript as a numbered list.',
};

const PROCESS_LABELS: Record<ProcessAction, string> = {
  cleanup: 'Clean Up Transcript',
  summarize: 'Summarize',
  actions: 'Extract Action Items',
};

/* ── Helpers ── */
function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = (seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

/* ── Main Component ── */
export function RecordPanel() {
  /* Recording state */
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  /* Playback state */
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);

  /* Transcript state */
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingAction, setProcessingAction] = useState<ProcessAction | null>(null);
  const [copied, setCopied] = useState(false);

  /* Model selector */
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showModelMenu, setShowModelMenu] = useState(false);

  /* Waveform bars */
  const [waveformBars, setWaveformBars] = useState<number[]>(Array(40).fill(0));

  /* Error */
  const [error, setError] = useState<string | null>(null);

  /* Refs */
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recognitionRef = useRef<any | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const finalTranscriptRef = useRef('');

  /* Fetch models */
  useEffect(() => {
    apiFetch('/api/models')
      .then(r => r.json())
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

  /* Cleanup on unmount */
  useEffect(() => {
    return () => {
      stopEverything();
      abortRef.current?.abort();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  /* Waveform animation loop */
  const startWaveformLoop = useCallback(() => {
    const analyser = analyserRef.current;
    if (!analyser) return;
    const data = new Uint8Array(analyser.frequencyBinCount);

    const tick = () => {
      analyser.getByteFrequencyData(data);
      const bars = Array.from({ length: 40 }, (_, i) => {
        const idx = Math.floor((i / 40) * data.length);
        return data[idx] / 255;
      });
      setWaveformBars(bars);
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  const stopEverything = useCallback(() => {
    /* Stop animation */
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    /* Stop timer */
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    /* Stop recognition */
    try { recognitionRef.current?.stop(); } catch {}
    recognitionRef.current = null;
    /* Stop stream tracks */
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    /* Close audio context */
    audioContextRef.current?.close().catch(() => {});
    audioContextRef.current = null;
    analyserRef.current = null;
    setWaveformBars(Array(40).fill(0));
  }, []);

  const startRecording = useCallback(async () => {
    setError(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('MediaRecorder is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      /* Audio context for waveform */
      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      startWaveformLoop();

      /* MediaRecorder */
      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        const url = URL.createObjectURL(blob);
        audioUrlRef.current = url;
        setAudioUrl(url);
      };

      recorder.start();

      /* Timer */
      setDuration(0);
      timerRef.current = setInterval(() => {
        setDuration(d => d + 1);
      }, 1000);

      /* Speech recognition */
      const SpeechRecognitionCtor =
        (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

      if (SpeechRecognitionCtor) {
        finalTranscriptRef.current = transcript; // preserve existing text
        const recognition = new SpeechRecognitionCtor();
        recognitionRef.current = recognition;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = navigator.language || 'en-US';

        recognition.onresult = (event: any) => {
          let interim = '';
          let newFinal = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const text = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              newFinal += text + ' ';
            } else {
              interim += text;
            }
          }
          if (newFinal) {
            finalTranscriptRef.current += newFinal;
            setTranscript(finalTranscriptRef.current);
          }
          setInterimTranscript(interim);
        };

        recognition.onerror = () => { /* silent — mic may be shared */ };

        try { recognition.start(); } catch {}
      }

      setIsRecording(true);
    } catch (err: any) {
      setError(`Microphone access denied: ${err.message}`);
    }
  }, [transcript, startWaveformLoop]);

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop();
    stopEverything();
    setInterimTranscript('');
    setIsRecording(false);
  }, [stopEverything]);

  /* Playback */
  const handlePlayPause = useCallback(() => {
    const audio = audioElRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  }, [isPlaying]);

  const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioElRef.current;
    if (!audio) return;
    audio.currentTime = Number(e.target.value);
    setPlaybackTime(Number(e.target.value));
  }, []);

  /* When audioUrl changes, wire up audio element events */
  useEffect(() => {
    const audio = audioElRef.current;
    if (!audio || !audioUrl) return;

    audio.src = audioUrl;
    audio.load();

    const onTimeUpdate = () => setPlaybackTime(audio.currentTime);
    const onDurationChange = () => setAudioDuration(audio.duration || 0);
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onEnded = () => { setIsPlaying(false); setPlaybackTime(0); };

    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('durationchange', onDurationChange);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onEnded);

    return () => {
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('durationchange', onDurationChange);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onEnded);
    };
  }, [audioUrl]);

  /* Post-processing via /api/generate/raw */
  const handleProcess = useCallback(async (action: ProcessAction) => {
    const text = transcript.trim();
    if (!text || isProcessing) return;

    setIsProcessing(true);
    setProcessingAction(action);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch('/api/generate/raw', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          system_prompt: PROCESS_PROMPTS[action],
          history: [],
          ...(selectedModel ? { model: selectedModel } : {}),
        }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      let result = '';

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
                if (t) result += t;
              } catch {
                result += data;
              }
            } else if (line.trim() && !line.startsWith(':')) {
              result += line;
            }
          }
        }
      } else {
        result = await res.text();
      }

      if (result.trim()) setTranscript(result.trim());
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(`Processing failed: ${e.message}`);
      }
    } finally {
      setIsProcessing(false);
      setProcessingAction(null);
      abortRef.current = null;
    }
  }, [transcript, isProcessing, selectedModel]);

  /* Copy transcript */
  const handleCopy = useCallback(() => {
    if (!transcript) return;
    navigator.clipboard.writeText(transcript).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [transcript]);

  /* Download audio */
  const handleDownloadAudio = useCallback(() => {
    if (!audioBlob) return;
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aura-recording-${Date.now()}.webm`;
    a.click();
    URL.revokeObjectURL(url);
  }, [audioBlob]);

  /* ── Render ── */
  return (
    <div className="flex flex-col h-full overflow-hidden bg-surface-0">
      {/* Header */}
      <div className="px-4 py-3 border-b border-chat-border flex-shrink-0">
        <h2 className="text-sm font-semibold text-chat-text">Record & Transcribe</h2>
        <p className="text-[10px] text-chat-text-secondary mt-0.5">Record audio and get an AI-powered transcript</p>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-5 p-4">

        {/* Error banner */}
        {error && (
          <div
            className="text-xs px-3 py-2 rounded-lg"
            style={{ background: 'rgba(239,68,68,0.12)', color: '#f87171', border: '1px solid rgba(239,68,68,0.25)' }}
          >
            {error}
            <button
              onClick={() => setError(null)}
              className="ml-2 opacity-60 hover:opacity-100"
            >
              ✕
            </button>
          </div>
        )}

        {/* Record button + waveform */}
        <div className="flex flex-col items-center gap-4">
          {/* Duration timer */}
          <div
            className="text-2xl font-mono font-semibold tabular-nums"
            style={{ color: isRecording ? '#ef4444' : 'var(--text-secondary)' }}
          >
            {formatDuration(duration)}
          </div>

          {/* Waveform */}
          <div
            className="w-full flex items-center justify-center gap-[2px] rounded-xl overflow-hidden"
            style={{
              height: 56,
              background: 'var(--surface-1)',
              border: '1px solid var(--border-subtle)',
              padding: '0 12px',
            }}
          >
            {waveformBars.map((v, i) => (
              <div
                key={i}
                style={{
                  width: 4,
                  borderRadius: 2,
                  height: `${Math.max(4, v * 48)}px`,
                  background: isRecording
                    ? `rgba(239,68,68,${0.4 + v * 0.6})`
                    : 'var(--border-default)',
                  transition: 'height 0.05s ease',
                  flexShrink: 0,
                }}
              />
            ))}
          </div>

          {/* Big record / stop button */}
          <button
            onClick={isRecording ? stopRecording : startRecording}
            title={isRecording ? 'Stop recording' : 'Start recording'}
            style={{
              width: 72,
              height: 72,
              borderRadius: '50%',
              border: isRecording ? '3px solid rgba(239,68,68,0.4)' : '3px solid var(--border-default)',
              background: isRecording ? '#ef4444' : 'var(--surface-2)',
              color: isRecording ? '#fff' : 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
              boxShadow: isRecording ? '0 0 0 6px rgba(239,68,68,0.15)' : 'none',
            }}
          >
            {isRecording
              ? <StopIcon style={{ width: 28, height: 28 }} />
              : <MicrophoneIcon style={{ width: 28, height: 28 }} />
            }
          </button>

          {isRecording && (
            <span
              className="text-[11px] font-medium animate-pulse"
              style={{ color: '#ef4444' }}
            >
              Recording…
            </span>
          )}
        </div>

        {/* Playback controls */}
        {audioUrl && !isRecording && (
          <div
            className="flex flex-col gap-2 rounded-xl p-3"
            style={{ background: 'var(--surface-1)', border: '1px solid var(--border-subtle)' }}
          >
            {/* Hidden audio element */}
            <audio ref={audioElRef} preload="metadata" />

            <div className="flex items-center gap-3">
              <button
                onClick={handlePlayPause}
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: '50%',
                  background: 'var(--chat-accent, #7c3aed)',
                  color: '#fff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                {isPlaying
                  ? <PauseIcon style={{ width: 14, height: 14 }} />
                  : <PlayIcon style={{ width: 14, height: 14 }} />
                }
              </button>

              <input
                type="range"
                min={0}
                max={audioDuration || 0}
                step={0.1}
                value={playbackTime}
                onChange={handleSeek}
                className="flex-1 accent-purple-500"
                style={{ height: 4, cursor: 'pointer' }}
              />

              <span className="text-[10px] text-chat-text-secondary tabular-nums flex-shrink-0">
                {formatDuration(Math.floor(playbackTime))} / {formatDuration(Math.floor(audioDuration))}
              </span>

              <button
                onClick={handleDownloadAudio}
                title="Download .webm"
                className="text-chat-text-secondary hover:text-chat-text transition-colors"
              >
                <ArrowDownTrayIcon style={{ width: 16, height: 16 }} />
              </button>
            </div>
          </div>
        )}

        {/* Transcript area */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-chat-text">Transcript</span>
            <button
              onClick={handleCopy}
              disabled={!transcript}
              title="Copy transcript"
              className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text disabled:opacity-30 transition-colors"
            >
              <ClipboardDocumentIcon style={{ width: 13, height: 13 }} />
              {copied ? 'Copied!' : 'Copy'}
            </button>
          </div>

          <div className="relative">
            <textarea
              value={transcript + (interimTranscript ? ` ${interimTranscript}` : '')}
              onChange={(e) => {
                setTranscript(e.target.value);
                finalTranscriptRef.current = e.target.value;
              }}
              placeholder={isRecording ? 'Transcribing in real-time…' : 'Transcript will appear here after recording…'}
              className="w-full rounded-xl text-sm resize-none outline-none text-chat-text placeholder-chat-text-secondary/50"
              style={{
                background: 'var(--surface-1)',
                border: '1px solid var(--border-default)',
                padding: '10px 12px',
                minHeight: 140,
                lineHeight: 1.6,
              }}
              rows={6}
            />
            {interimTranscript && (
              <span
                className="absolute bottom-3 right-3 text-[9px] animate-pulse"
                style={{ color: 'var(--text-secondary)' }}
              >
                live
              </span>
            )}
          </div>
        </div>

        {/* Post-processing actions */}
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <span className="text-xs font-medium text-chat-text">AI Actions</span>

            {/* Model selector */}
            <div className="relative" ref={modelMenuRef}>
              <button
                type="button"
                onClick={() => setShowModelMenu(p => !p)}
                disabled={isProcessing}
                className="flex items-center gap-1 text-[10px] text-chat-text-secondary hover:text-chat-text transition-colors px-2 py-1 rounded-md disabled:opacity-40"
                style={{ background: 'var(--border-subtle)' }}
              >
                <span className="max-w-[140px] truncate">
                  {selectedModel ? selectedModel.split('/').pop() : 'Auto'}
                </span>
                <svg className="w-2.5 h-2.5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showModelMenu && availableModels.length > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    bottom: 28,
                    right: 0,
                    width: 220,
                    maxHeight: 260,
                    background: 'var(--surface-1)',
                    border: '1px solid var(--border-default)',
                    borderRadius: 10,
                    overflow: 'hidden',
                    zIndex: 50,
                  }}
                >
                  <div style={{ maxHeight: 260, overflowY: 'auto', padding: 4 }}>
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

          <div className="flex flex-wrap gap-2">
            {(Object.keys(PROCESS_LABELS) as ProcessAction[]).map((action) => (
              <button
                key={action}
                onClick={() => handleProcess(action)}
                disabled={!transcript.trim() || isProcessing || isRecording}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg transition-all disabled:opacity-30"
                style={{
                  background: processingAction === action
                    ? 'var(--chat-accent, #7c3aed)'
                    : 'var(--surface-2)',
                  color: processingAction === action
                    ? '#fff'
                    : 'var(--text-secondary)',
                  border: '1px solid var(--border-subtle)',
                  cursor: !transcript.trim() || isProcessing || isRecording ? 'not-allowed' : 'pointer',
                }}
              >
                <SparklesIcon style={{ width: 13, height: 13 }} />
                {processingAction === action ? 'Processing…' : PROCESS_LABELS[action]}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
