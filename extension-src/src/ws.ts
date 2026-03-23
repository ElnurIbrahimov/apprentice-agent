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
const WS_STALE_TIMEOUT = 300_000; // 5 min — complex ops (deep research, web creator) can be slow

function clearWsTimers() {
  if (_connTimer) { clearTimeout(_connTimer); _connTimer = null; }
  if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
}

function resetStaleTimer(socket: WebSocket) {
  if (_staleTimer) clearTimeout(_staleTimer);
  const { activeStream } = useStore.getState();
  if (activeStream && activeStream !== true) {
    _staleTimer = setTimeout(() => {
      console.warn('[Aura] WebSocket stale (no message for 5min during stream), reconnecting');
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

/**
 * Quick HTTP health check. Returns true if the backend responds to /api/status.
 * Used to distinguish "WS failed but server is up" from "server is down".
 */
async function httpHealthCheck(): Promise<boolean> {
  try {
    const headers: Record<string, string> = {};
    if (API_KEY) headers['X-API-Key'] = API_KEY;
    const r = await fetch(`${HTTP}/api/status`, {
      headers,
      signal: AbortSignal.timeout(5000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

export function connectWS() {
  const { ws } = useStore.getState();
  if (ws && ws.readyState <= 1) return;

  // Read fresh WS_URL and API_KEY (live bindings from api.ts)
  // API key sent as first message after connect (not in query string — avoids log exposure)
  console.log('[Aura] Connecting WebSocket to:', WS_URL);
  const socket = new WebSocket(WS_URL);

  // 5s connection timeout -- if not open by then, close and let onclose retry
  _connTimer = setTimeout(() => {
    _connTimer = null;
    if (socket.readyState !== WebSocket.OPEN) {
      console.warn('[Aura] WebSocket connection timeout (5s), retrying');
      socket.close();
    }
  }, WS_CONNECT_TIMEOUT);

  socket.onopen = () => {
    if (_connTimer) { clearTimeout(_connTimer); _connTimer = null; }
    // Authenticate via first message (keeps key out of URL/logs)
    if (API_KEY) {
      socket.send(JSON.stringify({ type: 'auth', api_key: API_KEY }));
    }
    // Ask if the agent is ready
    socket.send(JSON.stringify({ type: 'ready_check' }));
    console.log('[Aura] WebSocket connected');
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
    console.warn(`[Aura] WebSocket closed (failure #${_wsConsecutiveFailures})`);

    // Flush any pending chunk buffer before finalization
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_chunkBuf || _thinkBuf) { flushChunks(); }
    const { activeStream } = useStore.getState();
    if (activeStream && activeStream !== true) {
      // Finalize with error so panels update correctly
      if (activeStream.onFirstChunk) activeStream.onFirstChunk();
      if (activeStream.onDone) activeStream.onDone(activeStream.rawText ? activeStream.rawText + '\n\n[Connection lost]' : '[Connection lost]');
    }
    useStore.getState().setActiveStream(null);

    // Don't immediately declare offline -- check HTTP first.
    // WS can fail (401, firewall, nginx miscfg) while HTTP works fine.
    httpHealthCheck().then((httpOk) => {
      if (httpOk) {
        // Server is up, just WS is broken. Keep status as 'online' so
        // the user can still chat via HTTP fallback.
        useStore.getState().setBackendStatus('online');
        console.log('[Aura] WebSocket down but HTTP is working -- staying online (HTTP fallback)');
      } else {
        // Both WS and HTTP are down
        if (_wsConsecutiveFailures >= 3) {
          useStore.getState().setBackendStatus('offline');
        } else {
          useStore.getState().setBackendStatus('connecting');
        }
      }
    });

    // Reconnect with exponential backoff + jitter -- stop after 8 failures
    if (_wsConsecutiveFailures <= 8) {
      const jitter = Math.random() * _wsRetryDelay * 0.3;
      setTimeout(() => {
        _wsRetryDelay = Math.min(_wsRetryDelay * 2, 30000);
        connectWS();
      }, _wsRetryDelay + jitter);
    }
  };

  socket.onerror = (ev) => {
    console.warn('[Aura] WebSocket error:', ev);
    useStore.getState().setWsReady(false);
  };

  socket.onmessage = (ev) => {
    resetStaleTimer(socket);

    let d: any;
    try { d = JSON.parse(ev.data); } catch { return; }

    // Handle control messages that don't require an active stream
    if (d.type === 'auth_ok' || d.type === 'pong') return;
    if (d.type === 'ready_status') {
      useStore.getState().setAgentReady(d.ready);
      return;
    }

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
      // Batch chunk updates via rAF to avoid O(n^2) string concat per frame
      _chunkBuf += content;
      if (!_rafId) _rafId = requestAnimationFrame(flushChunks);

    } else if (d.type === 'done') {
      // Stream complete -- clear stale timer
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
      // Stream error -- clear stale timer
      if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
      // Flush pending chunks before error handling
      if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
      _chunkBuf = '';
      _thinkBuf = '';
      const errStream = useStore.getState().activeStream;
      if (errStream && errStream !== true) {
        if (errStream.onFirstChunk) errStream.onFirstChunk();
        const errMsg = `[Error] ${d.content || d.error || 'Error'}`;
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

/** Reset WS retry state -- call when user manually changes server URL. */
export function resetWsRetry() {
  _wsRetryDelay = 1000;
  _wsConsecutiveFailures = 0;
}

export async function fetchStatus() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15_000);
    try {
      const headers: Record<string, string> = {};
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const r = await fetch(`${HTTP}/api/status`, { signal: ctrl.signal, headers });
      if (!r.ok) {
        // HTTP failed -- but don't wipe wsReady if WS is actually open
        const store = useStore.getState();
        if (!store.wsReady) {
          store.setBackendStatus('offline');
        }
        return;
      }
      const d = await r.json();
      const store = useStore.getState();
      // HTTP works -- server is online regardless of WS state
      store.setBackendStatus('online');
      // Agent is ready if model is reported (not "initializing...")
      const modelStr = d.last_model_used || d.model || '';
      store.setAgentReady(modelStr !== '' && modelStr !== 'initializing...');
      if (!store.wsReady) connectWS();
      const m = modelStr.replace(/:cloud$/, '');
      store.setModelName(m.length > 22 ? m.slice(-22) : m);
      if (d.mood?.emoji) store.setMood(d.mood.emoji);
    } finally {
      clearTimeout(timer);
    }
  } catch {
    // Network error -- only declare offline if WS is also down
    const store = useStore.getState();
    if (!store.wsReady) {
      store.setBackendStatus('offline');
    }
  }
}

// Page-context keywords -- auto-inject page content when matched
export const PAGE_KEYWORDS =
  /\b(this page|this site|this article|this post|this video|current page|what('s| is) (this|the) (page|site|article)|summarize this|explain this|what does this (say|mean)|translate this|tldr|tl;dr)\b/i;
