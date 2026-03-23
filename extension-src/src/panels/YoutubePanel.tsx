import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Copy, Check } from 'lucide-react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
import { md } from '../markdown';

interface Segment {
  start: number;
  dur: number;
  text: string;
}

interface Chapter {
  title: string;
  startMs: number;
}

interface VideoMeta {
  videoId: string;
  title: string;
  duration: number;
  description: string;
  channelName: string;
  chapters: Chapter[];
}

function formatTime(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function formatDuration(sec: number): string {
  if (!sec) return '';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function YoutubePanel() {
  const { getModel, activePanel, setPanel } = useStore();
  const [url, setUrl] = useState('');
  const [status, setStatus] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [snippetOpen, setSnippetOpen] = useState(false);
  const [autoUrl, setAutoUrl] = useState('');
  const [autoTitle, setAutoTitle] = useState('');

  // Intercepted data
  const [segments, setSegments] = useState<Segment[]>([]);
  const [videoMeta, setVideoMeta] = useState<VideoMeta | null>(null);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const transcriptRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [copied, setCopied] = useState(false);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup abort on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
    };
  }, []);

  // Listen for YT tab detection
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      setAutoUrl(detail.url || '');
      setAutoTitle(detail.title || detail.url || '');
    };
    window.addEventListener('yt-detected', handler);
    if ((window as any).__ytAutoUrl) {
      setAutoUrl((window as any).__ytAutoUrl);
      setAutoTitle((window as any).__ytAutoTitle || (window as any).__ytAutoUrl);
    }
    return () => window.removeEventListener('yt-detected', handler);
  }, []);

  // Listen for intercepted subtitles
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail.segments?.length) {
        setSegments(detail.segments);
      }
    };
    window.addEventListener('yt-subtitles', handler);
    return () => window.removeEventListener('yt-subtitles', handler);
  }, []);

  // Listen for intercepted metadata
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (detail.videoId) {
        setVideoMeta({
          videoId: detail.videoId,
          title: detail.title || '',
          duration: detail.duration || 0,
          description: detail.description || '',
          channelName: detail.channelName || '',
          chapters: detail.chapters || [],
        });
        // Also update auto-title if we have better info
        if (detail.title) setAutoTitle(detail.title);
      }
    };
    window.addEventListener('yt-metadata', handler);
    return () => window.removeEventListener('yt-metadata', handler);
  }, []);

  useEffect(() => {
    if (activePanel === 'youtube' && autoUrl) setUrl(autoUrl);
  }, [activePanel, autoUrl]);

  const summarize = async (urlToUse?: string) => {
    const target = (urlToUse || url).trim();
    if (!target) return;
    setLoading(true);
    setStatus('Fetching transcript...');
    setResult(null);
    try {
      if (abortRef.current) abortRef.current.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      // Abort after 90s timeout
      const timeoutId = setTimeout(() => ctrl.abort(), 90000);
      const resp = await fetch(`${HTTP}/api/youtube/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ url: target }),
        signal: ctrl.signal,
      });
      clearTimeout(timeoutId);
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('Error: ' + ((d as any).detail || `${resp.status}`));
        return;
      }
      const data = await resp.json();
      setResult(data);
      setStatus('');
    } catch (err: any) {
      setStatus('Error: ' + (err.name === 'AbortError' ? 'Request timed out.' : err.message || 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  // Build full transcript text for chat context
  const getFullTranscript = useCallback((): string => {
    if (!segments.length) return '';
    return segments.map((s) => `[${formatTime(s.start)}] ${s.text}`).join('\n');
  }, [segments]);

  const chatWithTranscript = useCallback(() => {
    const transcript = getFullTranscript();
    if (!transcript) return;
    const title = videoMeta?.title || autoTitle || 'YouTube Video';
    const context = `Video: ${title}\n${videoMeta?.channelName ? `Channel: ${videoMeta.channelName}\n` : ''}${videoMeta?.duration ? `Duration: ${formatDuration(videoMeta.duration)}\n` : ''}\n--- Transcript ---\n${transcript}`;

    // Store as pending context and switch to chat panel
    const store = useStore.getState();
    store.setPendingCtx({
      text: `Here is the transcript of "${title}". I'll ask questions about it.\n\n${context}`,
      action: 'ask',
      url: autoUrl || url,
      title,
    });
    setPanel('chat');
  }, [segments, videoMeta, autoTitle, autoUrl, url, setPanel, getFullTranscript]);

  // Filter segments by search query
  const filteredSegments = searchQuery.trim()
    ? segments.filter((s) => s.text.toLowerCase().includes(searchQuery.toLowerCase()))
    : segments;

  // Find which chapter a timestamp belongs to
  const getChapterForTime = (sec: number): string | null => {
    if (!videoMeta?.chapters?.length) return null;
    let current = '';
    for (const ch of videoMeta.chapters) {
      if (ch.startMs / 1000 <= sec) current = ch.title;
      else break;
    }
    return current || null;
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Auto-detected banner with metadata */}
      {(autoUrl || videoMeta) && (
        <div
          className="flex flex-col gap-1 px-3 py-2 flex-shrink-0"
          style={{ background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.2)' }}
        >
          <div className="flex items-center gap-2">
            <span style={{ fontSize: '12px', color: 'var(--tx)', flex: 1, fontWeight: 500 }}>
              {videoMeta?.title || autoTitle || 'YouTube Video'}
            </span>
            <button
              onClick={() => summarize(autoUrl)}
              disabled={loading}
              style={{
                background: '#ef4444',
                border: 'none',
                borderRadius: 'var(--r-sm)',
                color: 'white',
                fontSize: '11px',
                padding: '4px 10px',
                cursor: 'pointer',
                fontFamily: 'inherit',
                flexShrink: 0,
              }}
            >
              Summarize
            </button>
          </div>
          <div className="flex gap-3" style={{ fontSize: '10.5px', color: 'var(--mu)' }}>
            {videoMeta?.channelName && <span>{videoMeta.channelName}</span>}
            {videoMeta?.duration ? <span>{formatDuration(videoMeta.duration)}</span> : null}
            {segments.length > 0 && (
              <span style={{ color: 'rgba(34,197,94,0.8)' }}>
                {segments.length} subtitle segments captured
              </span>
            )}
          </div>
        </div>
      )}

      {/* URL input */}
      <div className="flex gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <input
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') summarize(); }}
          placeholder="YouTube URL..."
          style={{
            flex: 1,
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            color: 'var(--tx)',
            fontSize: '12px',
            padding: '7px 10px',
            outline: 'none',
            fontFamily: 'inherit',
          }}
        />
        <button
          onClick={() => summarize()}
          disabled={loading}
          style={{
            background: '#ef4444',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '7px 14px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {loading ? '...' : 'Summarize'}
        </button>
      </div>

      <div className="flex items-center gap-2 px-3 py-1.5 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        <ModelPill featureKey="youtube" />
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3">
        {status && (
          <div style={{ color: status.startsWith('Error') ? 'var(--rd)' : 'var(--mu)', fontSize: '12px', marginBottom: 12 }}>
            {loading && !status.startsWith('Error') && <span className="inline-flex gap-1 mr-2"><span className="dots"><span /><span /><span /></span></span>}
            {status}
          </div>
        )}

        {/* Intercepted transcript section */}
        {segments.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <div className="flex items-center gap-2" style={{ marginBottom: 8 }}>
              <button
                onClick={() => setTranscriptOpen(!transcriptOpen)}
                style={{
                  background: 'none',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)',
                  color: 'var(--tx)',
                  fontSize: '11px',
                  padding: '4px 10px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                }}
              >
                {transcriptOpen ? 'Hide' : 'Show'} Transcript ({segments.length} segments)
              </button>
              <button
                onClick={chatWithTranscript}
                style={{
                  background: 'rgba(124,58,237,0.15)',
                  border: '1px solid rgba(124,58,237,0.3)',
                  borderRadius: 'var(--r-sm)',
                  color: 'rgba(167,139,250,1)',
                  fontSize: '11px',
                  padding: '4px 10px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                }}
              >
                Chat with transcript
              </button>
            </div>

            {transcriptOpen && (
              <div ref={transcriptRef}>
                {/* Search within transcript */}
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search transcript..."
                  style={{
                    width: '100%',
                    background: 'var(--s2)',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: 'var(--tx)',
                    fontSize: '11px',
                    padding: '5px 8px',
                    outline: 'none',
                    fontFamily: 'inherit',
                    marginBottom: 8,
                    boxSizing: 'border-box',
                  }}
                />

                {/* Chapter markers */}
                {videoMeta?.chapters && videoMeta.chapters.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <div style={{
                      fontSize: '10px',
                      fontWeight: 600,
                      letterSpacing: '0.06em',
                      textTransform: 'uppercase',
                      color: 'var(--mu)',
                      marginBottom: 4,
                    }}>
                      Chapters
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {videoMeta.chapters.map((ch, i) => (
                        <span
                          key={i}
                          style={{
                            fontSize: '10px',
                            background: 'rgba(124,58,237,0.1)',
                            border: '1px solid rgba(124,58,237,0.2)',
                            borderRadius: 'var(--r-sm)',
                            padding: '2px 6px',
                            color: 'var(--mu)',
                          }}
                        >
                          {ch.title}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Transcript segments */}
                <div style={{
                  maxHeight: 400,
                  overflowY: 'auto',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-sm)',
                  background: 'var(--s2)',
                }}>
                  {filteredSegments.map((seg, i) => {
                    const chapter = getChapterForTime(seg.start);
                    // Show chapter header when it changes
                    const prevChapter = i > 0 ? getChapterForTime(filteredSegments[i - 1].start) : null;
                    const showChapterHeader = chapter && chapter !== prevChapter;

                    return (
                      <React.Fragment key={i}>
                        {showChapterHeader && (
                          <div style={{
                            padding: '4px 8px',
                            fontSize: '10px',
                            fontWeight: 600,
                            color: 'rgba(124,58,237,0.8)',
                            background: 'rgba(124,58,237,0.06)',
                            borderBottom: '1px solid var(--b1)',
                          }}>
                            {chapter}
                          </div>
                        )}
                        <div
                          style={{
                            display: 'flex',
                            gap: 8,
                            padding: '3px 8px',
                            borderBottom: '1px solid rgba(255,255,255,0.03)',
                            fontSize: '11.5px',
                            lineHeight: 1.5,
                          }}
                        >
                          <span style={{
                            color: 'rgba(124,58,237,0.7)',
                            fontSize: '10px',
                            fontFamily: 'monospace',
                            flexShrink: 0,
                            paddingTop: 1,
                            minWidth: 42,
                          }}>
                            {formatTime(seg.start)}
                          </span>
                          <span style={{ color: 'var(--tx)' }}>{seg.text}</span>
                        </div>
                      </React.Fragment>
                    );
                  })}
                  {filteredSegments.length === 0 && searchQuery && (
                    <div style={{ padding: 12, fontSize: '11px', color: 'var(--mu)', textAlign: 'center' }}>
                      No matches for "{searchQuery}"
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Server-side summary result */}
        {result && (
          <div className="flex flex-col gap-3" style={{ position: 'relative' }}>
            <div style={{ fontWeight: 600, fontSize: '13.5px', color: 'var(--tx)', paddingRight: 32 }}>{result.title || 'Untitled'}</div>
            <div className="flex gap-3" style={{ fontSize: '11px', color: 'var(--mu)' }}>
              {result.channel && <span>{result.channel}</span>}
              {result.duration && <span>{result.duration}</span>}
            </div>

            {/* Copy summary button */}
            <button
              onClick={() => {
                const text = [
                  result.title || '',
                  result.summary || '',
                  result.key_points?.length ? '\nKey Points:\n' + result.key_points.join('\n') : '',
                ].filter(Boolean).join('\n\n');
                navigator.clipboard.writeText(text).then(() => {
                  setCopied(true);
                  if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
                  copiedTimerRef.current = setTimeout(() => setCopied(false), 2000);
                }).catch(() => {});
              }}
              title="Copy summary"
              style={{
                position: 'absolute',
                top: 0,
                right: 0,
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                color: copied ? 'var(--green, #22c55e)' : 'var(--mu)',
                cursor: 'pointer',
                padding: '4px',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
            </button>

            <div
              className="md-body"
              style={{ fontSize: '12.5px', lineHeight: 1.65 }}
              dangerouslySetInnerHTML={{ __html: md(result.summary || 'No summary available.') }}
            />

            {result.key_points?.length > 0 && (
              <div>
                <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
                  Key Points
                </div>
                <ul style={{ paddingLeft: 16, fontSize: '12px', color: 'var(--tx)' }}>
                  {result.key_points.map((pt: string, i: number) => (
                    <li key={i} style={{ marginBottom: 4 }}>{pt}</li>
                  ))}
                </ul>
              </div>
            )}

            {result.transcript_snippet && (
              <div>
                <button
                  onClick={() => setSnippetOpen(!snippetOpen)}
                  style={{
                    background: 'none',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: 'var(--mu)',
                    fontSize: '11px',
                    padding: '4px 10px',
                    cursor: 'pointer',
                    fontFamily: 'inherit',
                  }}
                >
                  {snippetOpen ? 'Hide' : 'Show'} transcript snippet
                </button>
                {snippetOpen && (
                  <div style={{ marginTop: 8, fontSize: '11.5px', color: 'var(--mu)', fontStyle: 'italic', lineHeight: 1.6 }}>
                    {result.transcript_snippet}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {!loading && !status && !result && segments.length === 0 && (
          <div style={{ color: 'var(--mu)', fontSize: '12px', textAlign: 'center', marginTop: 32 }}>
            Paste a YouTube URL to summarize, or navigate to a YouTube video to auto-capture subtitles
          </div>
        )}
      </div>
    </div>
  );
}
