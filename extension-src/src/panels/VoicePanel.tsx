import React, { useState, useRef } from 'react';
import { Mic, MicOff, Upload, BookOpen, Volume2 } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, apiFetch } from '../api';
import { md } from '../markdown';
import { speak } from '../tts';

export default function VoicePanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel, autoSpeak, setAutoSpeak } = useStore();
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [status, setStatus] = useState('');
  const [notesHtml, setNotesHtml] = useState('');
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const prevNotesRef = useRef('');

  const startRec = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunksRef.current = [];
      const mr = new MediaRecorder(stream);
      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        await transcribeBlob(blob, 'recording.webm');
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      setStatus('Recording…');
    } catch {
      setStatus('⚠ Microphone access denied');
    }
  };

  const stopRec = () => {
    mediaRef.current?.stop();
    setRecording(false);
    setStatus('Transcribing…');
  };

  const transcribeBlob = async (blob: Blob, name: string) => {
    const form = new FormData();
    form.append('file', blob, name);
    try {
      const data = await apiFetch(`${HTTP}/api/transcribe`, { method: 'POST', body: form });
      setTranscript(data.text || '');
      setStatus('✓ Transcribed');
    } catch (err: any) {
      setStatus('⚠ ' + err.message);
    }
  };

  const uploadFile = () => fileRef.current?.click();

  const summarize = () => {
    if (!transcript.trim()) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { setStatus('AURA is offline.'); return; }
    if (activeStream) return;

    setNotesHtml('');
    const prompt = `Turn this voice transcript into structured, well-organized notes with headings and bullet points:\n\n${transcript}`;

    setActiveStream({
      type: 'chat',
      rawText: '',
      onFirstChunk: () => setNotesHtml(''),
      onDone: (rawText) => {
        setNotesHtml(md(rawText));
        if (useStore.getState().autoSpeak) {
          speak(rawText);
        }
      },
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('voice'), conversation_id: null }));
  };

  const saveToWisebase = async () => {
    if (!transcript.trim()) return;
    try {
      await apiFetch(`${HTTP}/api/knowledge/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: transcript.trim(),
          title: `Voice note ${new Date().toLocaleDateString()}`,
          source_type: 'voice_note',
        }),
      });
      setStatus('✓ Saved to Wisebase');
    } catch (err: any) {
      setStatus('⚠ Save failed: ' + err.message);
    }
  };

  const stream = useStore(s => s.activeStream);
  const isStreaming = stream && stream !== true;
  const streamText = isStreaming ? (stream as any).rawText : null;

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      {/* Record controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={recording ? stopRec : startRec}
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            background: recording ? 'rgba(239,68,68,0.15)' : 'var(--p)',
            border: recording ? '1px solid var(--rd)' : 'none',
            borderRadius: 'var(--r-md)',
            color: recording ? 'var(--rd)' : 'white',
            padding: '10px',
            cursor: 'pointer',
            fontSize: '13px',
            fontFamily: 'inherit',
            fontWeight: 500,
          }}
        >
          {recording ? <><MicOff size={16} /> Stop</> : <><Mic size={16} /> Record</>}
        </button>
        <button
          onClick={uploadFile}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 6,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--mu)',
            padding: '10px 14px',
            cursor: 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          <Upload size={14} /> Upload
        </button>
      </div>

      {/* Auto-speak toggle */}
      <label
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: '11.5px',
          color: 'var(--mu)',
          cursor: 'pointer',
          userSelect: 'none',
        }}
      >
        <Volume2 size={13} style={{ opacity: autoSpeak ? 1 : 0.5 }} />
        <span>Auto-speak responses</span>
        <div
          onClick={() => setAutoSpeak(!autoSpeak)}
          style={{
            marginLeft: 'auto',
            width: 32,
            height: 18,
            borderRadius: 9,
            background: autoSpeak ? 'var(--p)' : 'var(--s3)',
            position: 'relative',
            transition: 'background 0.2s',
            cursor: 'pointer',
            flexShrink: 0,
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: 2,
              left: autoSpeak ? 16 : 2,
              width: 14,
              height: 14,
              borderRadius: '50%',
              background: 'white',
              transition: 'left 0.2s',
            }}
          />
        </div>
      </label>

      <input
        ref={fileRef}
        type="file"
        accept="audio/*"
        style={{ display: 'none' }}
        onChange={e => {
          const f = e.target.files?.[0];
          if (f) { setStatus('Transcribing…'); transcribeBlob(f, f.name); }
        }}
      />

      {status && (
        <div style={{ color: status.startsWith('⚠') ? 'var(--rd)' : status.startsWith('✓') ? 'var(--gr)' : 'var(--mu)', fontSize: '12px' }}>
          {status}
        </div>
      )}

      {/* Transcript */}
      {transcript && (
        <>
          <div
            style={{
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)',
              padding: '10px',
              fontSize: '12.5px',
              color: 'var(--tx)',
              maxHeight: 120,
              overflowY: 'auto',
              lineHeight: 1.6,
              whiteSpace: 'pre-wrap',
            }}
          >
            {transcript}
          </div>

          <div className="flex items-center justify-between flex-shrink-0">
            <ModelPill featureKey="voice" />
            <div className="flex gap-2">
              <button
                onClick={saveToWisebase}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--mu)',
                  padding: '6px 12px',
                  cursor: 'pointer',
                  fontSize: '11px',
                  fontFamily: 'inherit',
                }}
              >
                <BookOpen size={12} /> Save
              </button>
              <button
                onClick={summarize}
                disabled={!!activeStream}
                style={{
                  background: activeStream ? 'var(--s3)' : 'var(--p)',
                  border: 'none',
                  borderRadius: 'var(--r-md)',
                  color: 'white',
                  padding: '6px 14px',
                  cursor: activeStream ? 'not-allowed' : 'pointer',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                }}
              >
                {activeStream ? '…' : 'Summarize'}
              </button>
            </div>
          </div>

          {(isStreaming || notesHtml) && (
            <div
              className="flex-1 overflow-y-auto"
              style={{
                background: 'var(--s1)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-md)',
                padding: '10px',
              }}
            >
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
        </>
      )}
    </div>
  );
}
