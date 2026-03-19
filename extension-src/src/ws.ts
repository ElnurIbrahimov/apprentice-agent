import { WS_URL, HTTP, API_KEY } from './api';
import { useStore } from './store';

let _wsRetryDelay = 1000;
let _wsConsecutiveFailures = 0;
let _chunkBuf = '';
let _thinkBuf = '';
let _rafId: number | null = null;
let _connTimer: ReturnType<typeof setTimeout> | null = null;
let _staleTimer: ReturnType<typeof setTimeout> | null = null;
const WS_CONNECT_TIMEOUT = 5_000;
const WS_STALE_TIMEOUT = 90_000;

function clearWsTimers() {
  if (_connTimer) { clearTimeout(_connTimer); _connTimer = null; }
  if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
}

function resetStaleTimer(socket: WebSocket) {
  if (_staleTimer) clearTimeout(_staleTimer);
  const { activeStream } = useStore.getState();
  if (activeStream && activeStream !== true) {
    _staleTimer = setTimeout(() => {
      console.warn('[Aura] WebSocket stale (no message for 90s during stream), reconnecting');
      socket.close();
    }, WS_STALE_TIMEOUT);
  }
}

function flushChunks(): void {
  _rafId = null;
  if (!_chunkBuf && !_thinkBuf) return;
  const buf = _chunkBuf;
  const tbuf = _thinkBuf;
  _chunkBuf = '';
  _thinkBuf = '';
  const { activeStream } = useStore.getState();
  if (!activeStream || activeStream === true) return;
  const update: any = { ...activeStream };
  if (buf) update.rawText = activeStream.rawText + buf;
  if (tbuf) update.thinkingText = (activeStream.thinkingText || '') + tbuf;
  useStore.setState({ activeStream: update });
}

export function connectWS() {
  const { ws } = useStore.getState();
  if (ws && ws.readyState <= 1) return;

  // Pass API key via query param (browsers cannot send custom headers on WebSocket)
  const wsUrl = API_KEY ? `${WS_URL}?api_key=${encodeURIComponent(API_KEY)}` : WS_URL;
  const socket = new WebSocket(wsUrl);

  // 5s connection timeout — if not open by then, close and let onclose retry
  _connTimer = setTimeout(() => {
    _connTimer = null;
    if (socket.readyState !== WebSocket.OPEN) {
      console.warn('[Aura] WebSocket connection timeout (5s), retrying');
      socket.close();
    }
  }, WS_CONNECT_TIMEOUT);

  socket.onopen = () => {
    if (_connTimer) { clearTimeout(_connTimer); _connTimer = null; }
    useStore.getState().setWsReady(true);
    useStore.getState().setWs(socket);
    useStore.getState().setBackendStatus('online');
    _wsRetryDelay = 1000;
    _wsConsecutiveFailures = 0;
  };

  socket.onclose = () => {
    clearWsTimers();
    useStore.getState().setWsReady(false);
    useStore.getState().setWs(null);
    _wsConsecutiveFailures++;
    // Don't override 'online' status from successful HTTP — WebSocket is optional for streaming
    // Only set offline if HTTP fetch also failed (backendStatus was already not 'online')
    const currentStatus = useStore.getState().backendStatus;
    if (currentStatus !== 'online') {
      if (_wsConsecutiveFailures >= 3) {
        useStore.getState().setBackendStatus('offline');
      } else {
        useStore.getState().setBackendStatus('connecting');
      }
    }
    // Flush any pending chunk buffer before finalization
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_chunkBuf || _thinkBuf) { flushChunks(); }
    const { activeStream } = useStore.getState();
    if (activeStream && activeStream !== true) {
      // Finalize with error so panels update correctly
      if (activeStream.onFirstChunk) activeStream.onFirstChunk();
      if (activeStream.onDone) activeStream.onDone(activeStream.rawText ? activeStream.rawText + '\n\n⚠ Connection lost' : '⚠ Connection lost');
    }
    useStore.getState().setActiveStream(null);
    // Reconnect with exponential backoff + jitter — stop after 5 failures
    if (_wsConsecutiveFailures <= 5) {
      const jitter = Math.random() * _wsRetryDelay * 0.3;
      setTimeout(() => {
        _wsRetryDelay = Math.min(_wsRetryDelay * 2, 30000);
        connectWS();
      }, _wsRetryDelay + jitter);
    }
  };

  socket.onerror = () => {
    useStore.getState().setWsReady(false);
  };

  socket.onmessage = (ev) => {
    resetStaleTimer(socket);

    let d: any;
    try { d = JSON.parse(ev.data); } catch { return; }

    const { activeStream } = useStore.getState();
    if (!activeStream || activeStream === true) return;

    if (d.type === 'thinking_chunk') {
      const content = d.content || '';
      // Mark as thinking phase if not already
      if (!activeStream.isThinkingPhase) {
        useStore.setState({
          activeStream: {
            ...activeStream,
            isThinkingPhase: true,
            thinkingStartTime: activeStream.thinkingStartTime || Date.now(),
            thinkingText: activeStream.thinkingText || '',
          },
        });
      }
      if (activeStream.onFirstChunk) {
        activeStream.onFirstChunk();
        useStore.setState({
          activeStream: { ...useStore.getState().activeStream as any, onFirstChunk: null },
        });
      }
      _thinkBuf += content;
      if (!_rafId) _rafId = requestAnimationFrame(flushChunks);

    } else if (d.type === 'chunk') {
      const content = d.content || '';
      // Transition from thinking phase to responding phase
      if (activeStream.isThinkingPhase) {
        useStore.setState({
          activeStream: { ...activeStream, isThinkingPhase: false },
        });
      }
      if (activeStream.onFirstChunk) {
        activeStream.onFirstChunk();
        useStore.setState({
          activeStream: { ...useStore.getState().activeStream as any, onFirstChunk: null },
        });
      }
      // Batch chunk updates via rAF to avoid O(n²) string concat per frame
      _chunkBuf += content;
      if (!_rafId) _rafId = requestAnimationFrame(flushChunks);

    } else if (d.type === 'done') {
      // Stream complete — clear stale timer
      if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
      // Flush any pending chunks before finalizing
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      if (_chunkBuf || _thinkBuf) { flushChunks(); }
      const finalStream = useStore.getState().activeStream;
      if (finalStream && finalStream !== true && finalStream.onDone) {
        finalStream.onDone(finalStream.rawText, finalStream.thinkingText);
      }
      useStore.getState().setActiveStream(null);

    } else if (d.type === 'error') {
      // Stream error — clear stale timer
      if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
      // Flush pending chunks before error handling
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      _chunkBuf = '';
      _thinkBuf = '';
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
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30_000);
    try {
      const headers: Record<string, string> = {};
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const r = await fetch(`${HTTP}/api/status`, { signal: ctrl.signal, headers });
      if (!r.ok) {
        useStore.getState().setWsReady(false);
        useStore.getState().setBackendStatus('offline');
        return;
      }
      const d = await r.json();
      const store = useStore.getState();
      // Recovered from offline/connecting — mark online
      if (store.backendStatus !== 'online') {
        store.setBackendStatus('online');
      }
      if (!store.wsReady) connectWS();
      const m = (d.last_model_used || d.model || '').replace(/:cloud$/, '');
      store.setModelName(m.length > 22 ? m.slice(-22) : m);
      if (d.mood?.emoji) store.setMood(d.mood.emoji);
    } finally {
      clearTimeout(timer);
    }
  } catch {
    useStore.getState().setWsReady(false);
    useStore.getState().setBackendStatus('offline');
  }
}

// Page-context keywords — auto-inject page content when matched
export const PAGE_KEYWORDS =
  /\b(this page|this site|this article|this post|this video|current page|what('s| is) (this|the) (page|site|article)|summarize this|explain this|what does this (say|mean)|translate this|tldr|tl;dr)\b/i;
