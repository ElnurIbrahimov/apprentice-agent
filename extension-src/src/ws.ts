import { WS_URL, HTTP } from './api';
import { useStore } from './store';

let _wsRetryDelay = 1000;
let _chunkBuf = '';
let _rafId: number | null = null;

function flushChunks(): void {
  _rafId = null;
  if (!_chunkBuf) return;
  const buf = _chunkBuf;
  _chunkBuf = '';
  const { activeStream } = useStore.getState();
  if (!activeStream || activeStream === true) return;
  useStore.setState({
    activeStream: { ...activeStream, rawText: activeStream.rawText + buf },
  });
}

export function connectWS() {
  const { ws } = useStore.getState();
  if (ws && ws.readyState <= 1) return;

  const socket = new WebSocket(WS_URL);

  socket.onopen = () => {
    useStore.getState().setWsReady(true);
    useStore.getState().setWs(socket);
    _wsRetryDelay = 1000;
  };

  socket.onclose = () => {
    useStore.getState().setWsReady(false);
    useStore.getState().setWs(null);
    // Flush any pending chunk buffer before finalization
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_chunkBuf) { flushChunks(); }
    const { activeStream } = useStore.getState();
    if (activeStream && activeStream !== true) {
      // Finalize with error so panels update correctly
      if (activeStream.onFirstChunk) activeStream.onFirstChunk();
      if (activeStream.onDone) activeStream.onDone(activeStream.rawText ? activeStream.rawText + '\n\n⚠ Connection lost' : '⚠ Connection lost');
    }
    useStore.getState().setActiveStream(null);
    // Reconnect with exponential backoff + jitter
    const jitter = Math.random() * _wsRetryDelay * 0.3;
    setTimeout(() => {
      _wsRetryDelay = Math.min(_wsRetryDelay * 2, 30000);
      connectWS();
    }, _wsRetryDelay + jitter);
  };

  socket.onerror = () => {
    useStore.getState().setWsReady(false);
  };

  socket.onmessage = (ev) => {
    let d: any;
    try { d = JSON.parse(ev.data); } catch { return; }

    const { activeStream } = useStore.getState();
    if (!activeStream || activeStream === true) return;

    if (d.type === 'chunk') {
      const content = d.content || '';
      if (activeStream.onFirstChunk) {
        activeStream.onFirstChunk();
        useStore.setState({
          activeStream: { ...activeStream, onFirstChunk: null },
        });
      }
      // Batch chunk updates via rAF to avoid O(n²) string concat per frame
      _chunkBuf += content;
      if (!_rafId) _rafId = requestAnimationFrame(flushChunks);
      // No separate scheduleMdRender — the setState above already triggers re-renders

    } else if (d.type === 'done') {
      // Flush any pending chunks before finalizing
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      if (_chunkBuf) { flushChunks(); }
      const finalStream = useStore.getState().activeStream;
      if (finalStream && finalStream !== true && finalStream.onDone) {
        finalStream.onDone(finalStream.rawText);
      }
      useStore.getState().setActiveStream(null);

    } else if (d.type === 'error') {
      // Flush pending chunks before error handling
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      _chunkBuf = '';
      const errStream = useStore.getState().activeStream;
      if (errStream && errStream !== true) {
        if (errStream.onFirstChunk) errStream.onFirstChunk();
        const errMsg = `⚠ ${d.content || d.error || 'Error'}`;
        useStore.setState({ activeStream: { ...errStream, rawText: errMsg, onFirstChunk: null } });
        if (errStream.onDone) errStream.onDone(errMsg);
      }
      useStore.getState().setActiveStream(null);

    } else if (d.type === 'conversation_id') {
      useStore.getState().setConversationId(d.id);
    }
  };

  useStore.getState().setWs(socket);
}

export async function fetchStatus() {
  try {
    const r = await fetch(`${HTTP}/api/status`, { signal: AbortSignal.timeout(4000) });
    if (!r.ok) { useStore.getState().setWsReady(false); return; }
    const d = await r.json();
    if (!useStore.getState().wsReady) connectWS();
    const m = (d.last_model_used || d.model || '').replace(/:cloud$/, '');
    useStore.getState().setModelName(m.length > 22 ? m.slice(-22) : m);
    if (d.mood?.emoji) useStore.getState().setMood(d.mood.emoji);
  } catch {
    useStore.getState().setWsReady(false);
  }
}

// Page-context keywords — auto-inject page content when matched
export const PAGE_KEYWORDS =
  /\b(this page|this site|this article|this post|this video|current page|what('s| is) (this|the) (page|site|article)|summarize this|explain this|what does this (say|mean)|translate this|tldr|tl;dr)\b/i;
