import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Mic, MonitorSpeaker, Square, Play, Pause, Copy, Check, FileText, ListChecks, Sparkles } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, apiFetch } from '../api';
import { md } from '../markdown';

/* ── Types ── */
type RecordingSource = 'tab' | 'mic' | null;
type Phase = 'idle' | 'recording' | 'recorded' | 'transcribing' | 'transcribed';

/* ── Waveform bars ── */
const BAR_COUNT = 24;

function WaveformBars({ analyser }: { analyser: AnalyserNode | null }) {
  const barsRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    if (!analyser || !barsRef.current) return;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    const bars = barsRef.current.children;

    const draw = () => {
      analyser.getByteFrequencyData(dataArray);
      const step = Math.floor(dataArray.length / BAR_COUNT);
      for (let i = 0; i < BAR_COUNT; i++) {
        const val = dataArray[i * step] / 255;
        const h = Math.max(3, val * 32);
        (bars[i] as HTMLElement).style.height = h + 'px';
      }
      rafRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(rafRef.current);
  }, [analyser]);

  return (
    <div
      ref={barsRef}
      style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: 2,
        height: 36,
        padding: '0 4px',
      }}
    >
      {Array.from({ length: BAR_COUNT }, (_, i) => (
        <div
          key={i}
          style={{
            width: 3,
            height: 3,
            borderRadius: 1.5,
            background: `linear-gradient(180deg, #a78bfa, #7c3aed)`,
            transition: 'height 0.06s ease-out',
            flexShrink: 0,
          }}
        />
      ))}
    </div>
  );
}

/* ── Timer display ── */
function Timer({ seconds }: { seconds: number }) {
  const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');
  return (
    <span style={{ fontFamily: 'monospace', fontSize: '18px', fontWeight: 600, color: 'var(--tx)', letterSpacing: 1 }}>
      {mm}:{ss}
    </span>
  );
}

/* ── Main panel ── */
export default function RecordPanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();

  // Recording state
  const [phase, setPhase] = useState<Phase>('idle');
  const [source, setSource] = useState<RecordingSource>(null);
  const [duration, setDuration] = useState(0);
  const [status, setStatus] = useState('');

  // Audio results
  const [audioBlobUrl, setAudioBlobUrl] = useState<string | null>(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Transcript + notes
  const [transcript, setTranscript] = useState('');
  const [notesHtml, setNotesHtml] = useState('');
  const [notesMode, setNotesMode] = useState<'summary' | 'meeting' | null>(null);
  const [copied, setCopied] = useState<'transcript' | 'notes' | null>(null);

  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioElRef = useRef<HTMLAudioElement | null>(null);

  const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);

  // Streaming
  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  /* ── Cleanup on unmount ── */
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      if (audioCtxRef.current?.state !== 'closed') {
        audioCtxRef.current?.close().catch(() => {});
      }
      if (audioBlobUrl) URL.revokeObjectURL(audioBlobUrl);
    };
  }, []);

  /* ── Start recording ── */
  const startRecording = useCallback(async (src: RecordingSource) => {
    if (!src) return;

    // Clean up previous recording
    if (audioBlobUrl) {
      URL.revokeObjectURL(audioBlobUrl);
      setAudioBlobUrl(null);
    }
    setAudioBlob(null);
    setTranscript('');
    setNotesHtml('');
    setNotesMode(null);
    setStatus('');

    let mediaStream: MediaStream;

    try {
      if (src === 'tab') {
        // Try getDisplayMedia for tab audio capture
        // This prompts the user to pick a tab/screen — include audio
        mediaStream = await navigator.mediaDevices.getDisplayMedia({
          audio: true,
          video: true, // required by spec, we discard the video track
        });
        // Stop video track immediately — we only want audio
        mediaStream.getVideoTracks().forEach(t => t.stop());

        // Check if we actually got an audio track
        if (mediaStream.getAudioTracks().length === 0) {
          setStatus('No audio track — make sure "Share tab audio" is checked');
          return;
        }
      } else {
        // Microphone
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
    } catch (err: any) {
      if (err.name === 'NotAllowedError' || err.name === 'AbortError') {
        setStatus(src === 'tab' ? 'Tab capture cancelled' : 'Microphone access denied');
      } else {
        setStatus('Could not access audio: ' + (err.message || err.name));
      }
      return;
    }

    streamRef.current = mediaStream;

    // Set up Web Audio analyser for waveform
    try {
      const actx = new AudioContext();
      audioCtxRef.current = actx;
      const sourceNode = actx.createMediaStreamSource(mediaStream);
      const analyser = actx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.7;
      sourceNode.connect(analyser);
      analyserRef.current = analyser;
      setAnalyserNode(analyser);
    } catch {
      // Waveform won't work, but recording still can
    }

    // Set up MediaRecorder
    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    const recorder = new MediaRecorder(mediaStream, { mimeType });
    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      const url = URL.createObjectURL(blob);
      setAudioBlob(blob);
      setAudioBlobUrl(url);
      setPhase('recorded');

      // Cleanup stream tracks
      mediaStream.getTracks().forEach(t => t.stop());
      streamRef.current = null;

      // Close audio context
      if (audioCtxRef.current?.state !== 'closed') {
        audioCtxRef.current?.close().catch(() => {});
      }
      setAnalyserNode(null);
    };

    recorder.start(1000); // collect chunks every second
    mediaRecorderRef.current = recorder;

    // Start timer
    setDuration(0);
    timerRef.current = setInterval(() => {
      setDuration(prev => prev + 1);
    }, 1000);

    setSource(src);
    setPhase('recording');
    setStatus(src === 'tab' ? 'Recording tab audio...' : 'Recording microphone...');
  }, [audioBlobUrl]);

  /* ── Stop recording ── */
  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    mediaRecorderRef.current?.stop();
    setStatus('Recording stopped');
  }, []);

  /* ── Transcribe ── */
  const transcribe = useCallback(async () => {
    if (!audioBlob) return;
    setPhase('transcribing');
    setStatus('Transcribing...');

    const form = new FormData();
    form.append('file', audioBlob, 'recording.webm');

    try {
      const data = await apiFetch(`${HTTP}/api/transcribe`, { method: 'POST', body: form });
      const text = data.text || '';
      setTranscript(text);
      setPhase('transcribed');
      setStatus(text ? 'Transcription complete' : 'No speech detected');
    } catch (err: any) {
      setPhase('recorded');
      setStatus('Transcription failed: ' + err.message);
    }
  }, [audioBlob]);

  /* ── AI actions (summarize / meeting notes) ── */
  const generateNotes = useCallback((mode: 'summary' | 'meeting') => {
    if (!transcript.trim()) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
      setStatus('AURA is offline');
      return;
    }
    if (activeStream) return;

    setNotesHtml('');
    setNotesMode(mode);

    const prompt = mode === 'meeting'
      ? `Turn this transcript into structured meeting notes with these sections:\n\n## Agenda / Topics Discussed\n## Key Decisions\n## Action Items\n## Follow-ups\n\nTranscript:\n\n${transcript}`
      : `Summarize this recording transcript concisely, highlighting the key points:\n\n${transcript}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setNotesHtml(''),
      onDone: (rawText) => {
        setNotesHtml(md(rawText));
      },
    });

    ws!.send(JSON.stringify({
      type: 'chat',
      message: prompt,
      model: getModel('voice'),
      conversation_id: null,
    }));
  }, [transcript, wsReady, ws, activeStream, setActiveStream, getModel]);

  /* ── Copy to clipboard ── */
  const copyText = useCallback((text: string, which: 'transcript' | 'notes') => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(which);
      setTimeout(() => setCopied(null), 1500);
    });
  }, []);

  /* ── Audio playback ── */
  const togglePlayback = useCallback(() => {
    if (!audioElRef.current) return;
    if (isPlaying) {
      audioElRef.current.pause();
    } else {
      audioElRef.current.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  /* ── Render ── */
  const isRecording = phase === 'recording';

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">

      {/* Header */}
      <div className="flex items-center gap-2" style={{ flexShrink: 0 }}>
        <div
          style={{
            width: 28, height: 28, borderRadius: 'var(--r-sm)',
            background: 'linear-gradient(135deg, #7c3aed22, #a78bfa22)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <Mic size={14} style={{ color: '#a78bfa' }} />
        </div>
        <div>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>REC Note</div>
          <div style={{ fontSize: '10px', color: 'var(--mu)' }}>Record &amp; transcribe</div>
        </div>
      </div>

      {/* Recording controls — shown in idle state */}
      {phase === 'idle' && (
        <div className="flex flex-col gap-2">
          <button
            onClick={() => startRecording('tab')}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              background: 'linear-gradient(135deg, #7c3aed, #6d28d9)',
              border: 'none', borderRadius: 'var(--r-md)',
              color: 'white', padding: '11px', cursor: 'pointer',
              fontSize: '13px', fontFamily: 'inherit', fontWeight: 500,
            }}
          >
            <MonitorSpeaker size={16} /> Record Tab Audio
          </button>
          <button
            onClick={() => startRecording('mic')}
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', color: 'var(--tx)',
              padding: '11px', cursor: 'pointer',
              fontSize: '13px', fontFamily: 'inherit', fontWeight: 500,
            }}
          >
            <Mic size={16} /> Record Microphone
          </button>
          <div style={{ fontSize: '10.5px', color: 'var(--mu)', lineHeight: 1.5, padding: '0 2px' }}>
            Tab audio captures sound from the current tab (meetings, videos, music).
            Check "Share tab audio" when prompted.
          </div>
        </div>
      )}

      {/* Active recording view */}
      {isRecording && (
        <div className="flex flex-col items-center gap-3" style={{ padding: '8px 0' }}>
          {/* Source badge */}
          <div style={{
            fontSize: '10px', fontWeight: 500, color: '#a78bfa',
            background: '#a78bfa18', padding: '3px 10px',
            borderRadius: 'var(--r-pill)', textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>
            {source === 'tab' ? 'Tab Audio' : 'Microphone'}
          </div>

          {/* Timer */}
          <Timer seconds={duration} />

          {/* Waveform */}
          <WaveformBars analyser={analyserNode} />

          {/* Stop button */}
          <button
            onClick={stopRecording}
            style={{
              width: 52, height: 52, borderRadius: '50%',
              background: 'var(--rd)', border: '3px solid rgba(239,68,68,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer',
              animation: 'recPulse 1.5s ease-in-out infinite',
            }}
          >
            <Square size={18} fill="white" color="white" />
          </button>
        </div>
      )}

      {/* Post-recording: audio player + actions */}
      {(phase === 'recorded' || phase === 'transcribing' || phase === 'transcribed') && (
        <>
          {/* Audio player */}
          {audioBlobUrl && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8,
              background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', padding: '8px 10px',
            }}>
              <button
                onClick={togglePlayback}
                style={{
                  width: 32, height: 32, borderRadius: '50%',
                  background: 'var(--p)', border: 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', flexShrink: 0,
                }}
              >
                {isPlaying
                  ? <Pause size={14} fill="white" color="white" />
                  : <Play size={14} fill="white" color="white" style={{ marginLeft: 2 }} />
                }
              </button>
              <audio
                ref={audioElRef}
                src={audioBlobUrl}
                onEnded={() => setIsPlaying(false)}
                style={{ flex: 1, height: 28 }}
                controls
              />
              <div style={{ fontSize: '10px', color: 'var(--mu)', whiteSpace: 'nowrap' }}>
                {String(Math.floor(duration / 60)).padStart(2, '0')}:{String(duration % 60).padStart(2, '0')}
              </div>
            </div>
          )}

          {/* Transcribe button */}
          {phase === 'recorded' && (
            <button
              onClick={transcribe}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                background: 'linear-gradient(135deg, #7c3aed, #6d28d9)',
                border: 'none', borderRadius: 'var(--r-md)',
                color: 'white', padding: '10px', cursor: 'pointer',
                fontSize: '13px', fontFamily: 'inherit', fontWeight: 500,
              }}
            >
              <FileText size={15} /> Transcribe
            </button>
          )}

          {/* Transcribing spinner */}
          {phase === 'transcribing' && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              gap: 8, padding: '10px', color: 'var(--mu)', fontSize: '12px',
            }}>
              <div className="dots"><span /><span /><span /></div>
              Transcribing...
            </div>
          )}

          {/* Transcript display */}
          {transcript && (
            <>
              <div style={{ position: 'relative' }}>
                <div
                  style={{
                    background: 'var(--s2)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-md)', padding: '10px',
                    fontSize: '12.5px', color: 'var(--tx)',
                    maxHeight: 140, overflowY: 'auto',
                    lineHeight: 1.6, whiteSpace: 'pre-wrap',
                  }}
                >
                  {transcript}
                </div>
                <button
                  onClick={() => copyText(transcript, 'transcript')}
                  title="Copy transcript"
                  style={{
                    position: 'absolute', top: 6, right: 6,
                    background: 'var(--s3)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '4px',
                    cursor: 'pointer', display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  {copied === 'transcript'
                    ? <Check size={12} style={{ color: 'var(--gr)' }} />
                    : <Copy size={12} style={{ color: 'var(--mu)' }} />
                  }
                </button>
              </div>

              {/* Action buttons row */}
              <div className="flex items-center justify-between flex-shrink-0">
                <ModelPill featureKey="voice" />
                <div className="flex gap-2">
                  <button
                    onClick={() => generateNotes('summary')}
                    disabled={!!activeStream}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      background: activeStream ? 'var(--s3)' : 'var(--s2)',
                      border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                      color: activeStream ? 'var(--di)' : 'var(--mu)',
                      padding: '6px 10px', cursor: activeStream ? 'not-allowed' : 'pointer',
                      fontSize: '11px', fontFamily: 'inherit',
                    }}
                  >
                    <Sparkles size={12} /> Summarize
                  </button>
                  <button
                    onClick={() => generateNotes('meeting')}
                    disabled={!!activeStream}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 5,
                      background: activeStream ? 'var(--s3)' : 'var(--p)',
                      border: 'none', borderRadius: 'var(--r-md)',
                      color: activeStream ? 'var(--di)' : 'white',
                      padding: '6px 10px', cursor: activeStream ? 'not-allowed' : 'pointer',
                      fontSize: '11px', fontFamily: 'inherit', fontWeight: 500,
                    }}
                  >
                    <ListChecks size={12} /> Meeting Notes
                  </button>
                </div>
              </div>
            </>
          )}

          {/* AI-generated notes display */}
          {(isStreaming || notesHtml) && (
            <div
              className="flex-1 overflow-y-auto"
              style={{
                background: 'var(--s1)', border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)', padding: '10px',
                position: 'relative',
              }}
            >
              {/* Copy notes button */}
              {notesHtml && !isStreaming && (
                <button
                  onClick={() => {
                    // Extract text content from the rendered notes
                    const el = document.createElement('div');
                    el.innerHTML = notesHtml;
                    copyText(el.textContent || '', 'notes');
                  }}
                  title="Copy notes"
                  style={{
                    position: 'absolute', top: 6, right: 6,
                    background: 'var(--s3)', border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)', padding: '4px',
                    cursor: 'pointer', display: 'flex',
                    alignItems: 'center', justifyContent: 'center',
                    zIndex: 1,
                  }}
                >
                  {copied === 'notes'
                    ? <Check size={12} style={{ color: 'var(--gr)' }} />
                    : <Copy size={12} style={{ color: 'var(--mu)' }} />
                  }
                </button>
              )}

              {isStreaming && !streamText ? (
                <div className="dots"><span /><span /><span /></div>
              ) : (
                <div
                  className="md-body"
                  style={{ fontSize: '12.5px', lineHeight: 1.65 }}
                  dangerouslySetInnerHTML={{ __html: notesHtml || md(streamText || '') }}
                />
              )}
            </div>
          )}

          {/* New recording button */}
          {!isRecording && phase !== 'transcribing' && (
            <button
              onClick={() => {
                setPhase('idle');
                setSource(null);
                setDuration(0);
                setStatus('');
                if (audioBlobUrl) URL.revokeObjectURL(audioBlobUrl);
                setAudioBlobUrl(null);
                setAudioBlob(null);
                setTranscript('');
                setNotesHtml('');
                setNotesMode(null);
              }}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                background: 'transparent', border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)', color: 'var(--mu)',
                padding: '8px', cursor: 'pointer',
                fontSize: '11px', fontFamily: 'inherit',
                flexShrink: 0,
              }}
            >
              <Mic size={12} /> New Recording
            </button>
          )}
        </>
      )}

      {/* Status bar */}
      {status && phase !== 'recording' && (
        <div style={{
          color: status.includes('failed') || status.includes('denied') || status.includes('cancelled')
            ? 'var(--rd)'
            : status.includes('complete') || status.includes('stopped')
              ? 'var(--gr)'
              : 'var(--mu)',
          fontSize: '11px',
          flexShrink: 0,
        }}>
          {status}
        </div>
      )}

      {/* Pulse animation style */}
      <style>{`
        @keyframes recPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
          50% { box-shadow: 0 0 0 10px rgba(239,68,68,0); }
        }
      `}</style>
    </div>
  );
}
