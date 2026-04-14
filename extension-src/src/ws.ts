import { HTTP, API_KEY } from './api';
import { useStore, registerReconnectHandler } from './store';

/**
 * WebSocket transport lives in the offscreen document (see offscreen.ts).
 * This module is a thin client: it exposes a WebSocket-shaped proxy so
 * panels can keep calling `socket.send(...)` unchanged, listens for
 * `AURA_WS_IN` runtime messages from offscreen, and applies the same
 * message switch that the single-process version used.
 */

const WS_STALE_TIMEOUT = 300_000; // 5 min during active streams
let _chunkBuf = '';
let _thinkBuf = '';
let _rafId: number | null = null;
let _staleTimer: ReturnType<typeof setTimeout> | null = null;

// --- WsProxy: WebSocket-shaped handle for the store ------------------------
// Keeps the store's `ws` field contract ({ readyState, send, close }) so
// every panel's `socket?.readyState === WebSocket.OPEN && socket.send(...)`
// pattern keeps working without modification. The real WebSocket lives in
// the offscreen document (see offscreen.ts).
//
// `close()` here means "tear down and reconnect" — this matches the only
// caller's intent (SettingsPanel after a URL change). True close-without-
// reconnect isn't a useful semantic for an offscreen-hosted socket.

class WsProxy {
  readyState: number = WebSocket.CLOSED;
  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    const payload = typeof data === 'string' ? data : String(data);
    chrome.runtime.sendMessage({ type: 'AURA_WS_OUT', payload }).catch(() => {});
  }
  close(): void {
    chrome.runtime.sendMessage({ type: 'AURA_WS_RECONNECT' }).catch(() => {});
  }
}

const wsProxy = new WsProxy();

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

function resetStaleTimer(): void {
  if (_staleTimer) clearTimeout(_staleTimer);
  const { activeStream } = useStore.getState();
  if (activeStream && activeStream !== true) {
    _staleTimer = setTimeout(() => {
      console.warn('[Aura] WebSocket stale during stream, requesting reconnect');
      chrome.runtime.sendMessage({ type: 'AURA_WS_RECONNECT' }).catch(() => {});
    }, WS_STALE_TIMEOUT);
  }
}

/** Quick HTTP health check for "WS failed but server is up" detection. */
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

// --- Incoming message dispatcher -------------------------------------------

function handleWsFrame(raw: string): void {
  let d: any;
  try { d = JSON.parse(raw); } catch { return; }

  resetStaleTimer();

  if (d.type === 'auth_ok' || d.type === 'pong') return;
  if (d.type === 'ready_status') {
    useStore.getState().setAgentReady(d.ready);
    return;
  }

  if (d.type === 'proactive' && d.id && d.text) {
    useStore.getState().addProactiveMessage({
      id: d.id,
      text: d.text,
      timestamp: d.timestamp || Date.now(),
    });
    return;
  }

  if (d.type === 'routing') {
    useStore.getState().setLastRoutingResult(d);
    return;
  }

  // Hands events — route to the store's hand-event handlers.
  const store = useStore.getState();
  if (d.type === 'hand_event') {
    store.applyHandEvent(d);
    return;
  }
  if (d.type === 'hand_approval_request') {
    store.applyHandApprovalRequest(d);
    return;
  }
  if (d.type === 'action_trace') {
    store.applyHandActionTrace(d);
    return;
  }

  const { activeStream } = useStore.getState();
  if (!activeStream || activeStream === true) return;

  if (d.type === 'thinking_chunk') {
    const content = d.content || '';
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
      const current = useStore.getState().activeStream;
      if (current && current !== true) {
        useStore.setState({ activeStream: { ...current, onFirstChunk: null } });
      }
    }
    _thinkBuf += content;
    if (!_rafId) _rafId = requestAnimationFrame(flushChunks);

  } else if (d.type === 'chunk') {
    const content = d.content || '';
    if (activeStream.isThinkingPhase) {
      useStore.setState({
        activeStream: { ...activeStream, isThinkingPhase: false },
      });
    }
    if (activeStream.onFirstChunk) {
      activeStream.onFirstChunk();
      const current = useStore.getState().activeStream;
      if (current && current !== true) {
        useStore.setState({ activeStream: { ...current, onFirstChunk: null } });
      }
    }
    _chunkBuf += content;
    if (!_rafId) _rafId = requestAnimationFrame(flushChunks);

  } else if (d.type === 'done') {
    if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_chunkBuf || _thinkBuf) { flushChunks(); }
    const finalStream = useStore.getState().activeStream;
    if (finalStream && finalStream !== true && finalStream.onDone) {
      finalStream.onDone(finalStream.rawText, finalStream.thinkingText);
    }
    useStore.getState().setActiveStream(null);

  } else if (d.type === 'error') {
    if (_staleTimer) { clearTimeout(_staleTimer); _staleTimer = null; }
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
}

// --- Runtime message listener (from offscreen via SW broadcast) ------------

chrome.runtime?.onMessage?.addListener((msg) => {
  if (!msg || typeof msg !== 'object') return false;

  if (msg.type === 'AURA_WS_IN' && typeof msg.payload === 'string') {
    handleWsFrame(msg.payload);
    return false;
  }

  if (msg.type === 'AURA_WS_STATUS') {
    const rs = Number(msg.readyState ?? 3);
    wsProxy.readyState = rs;
    const store = useStore.getState();
    store.setWsReady(rs === WebSocket.OPEN);
    if (rs === WebSocket.OPEN) {
      store.setBackendStatus('online');
      store.setConnectionAbandoned(false);
    }
    // Set the proxy handle once — skip if already in place.
    if (store.ws !== (wsProxy as unknown as WebSocket)) {
      store.setWs(wsProxy as unknown as WebSocket);
    }
    return false;
  }

  if (msg.type === 'AURA_WS_CLOSED') {
    wsProxy.readyState = WebSocket.CLOSED;
    const store = useStore.getState();
    store.setWsReady(false);

    // Flush pending chunks + finalize any live stream with a "Connection lost" error.
    if (_rafId) { cancelAnimationFrame(_rafId); _rafId = null; }
    if (_chunkBuf || _thinkBuf) { flushChunks(); }
    const { activeStream } = store;
    if (activeStream && activeStream !== true) {
      if (activeStream.onFirstChunk) activeStream.onFirstChunk();
      if (activeStream.onDone) {
        const lostMsg = activeStream.rawText ? activeStream.rawText + '\n\n[Connection lost]' : '[Connection lost]';
        activeStream.onDone(lostMsg);
      }
    }
    store.setActiveStream(null);

    // Probe HTTP to distinguish "server down" from "WS glitch".
    httpHealthCheck().then((httpOk) => {
      const s = useStore.getState();
      if (httpOk) {
        s.setBackendStatus('online');
      } else if ((msg.consecutiveFailures ?? 0) >= 3) {
        s.setBackendStatus('offline');
      } else {
        s.setBackendStatus('connecting');
      }
    });
    return false;
  }

  if (msg.type === 'AURA_WS_ABANDONED') {
    useStore.getState().setConnectionAbandoned(true);
    return false;
  }

  return false;
});

// --- Public API (unchanged signatures so main.tsx works as-is) -------------

export function connectWS(): void {
  // Ensure the store has the proxy handle early so UI code calling
  // store.ws?.send() before the first AURA_WS_STATUS arrives is a no-op
  // rather than a crash.
  const store = useStore.getState();
  if (store.ws !== (wsProxy as unknown as WebSocket)) {
    store.setWs(wsProxy as unknown as WebSocket);
  }
  // Request a status broadcast from the offscreen doc; service worker
  // will create the offscreen doc if it doesn't exist yet.
  chrome.runtime.sendMessage({ type: 'AURA_ENSURE_OFFSCREEN' }).catch(() => {});
  chrome.runtime.sendMessage({ type: 'AURA_WS_STATUS_REQUEST' }).catch(() => {});
}

/**
 * Deprecated: the offscreen document now owns the retry counter, and
 * `AURA_WS_RECONNECT` (fired by `reconnectWS()` or `WsProxy.close()`) resets
 * it as a side effect. Kept as a no-op so `SettingsPanel.tsx` doesn't break.
 */
export function resetWsRetry(): void { /* no-op */ }

export function reconnectWS(): void {
  useStore.getState().setConnectionAbandoned(false);
  chrome.runtime.sendMessage({ type: 'AURA_WS_RECONNECT' }).catch(() => {});
}

registerReconnectHandler(reconnectWS);

export async function fetchStatus(): Promise<void> {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 15_000);
    try {
      const headers: Record<string, string> = {};
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const r = await fetch(`${HTTP}/api/status`, { signal: ctrl.signal, headers });
      if (!r.ok) {
        const store = useStore.getState();
        if (!store.wsReady) store.setBackendStatus('offline');
        return;
      }
      const d = await r.json();
      const store = useStore.getState();
      store.setBackendStatus('online');
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
    const store = useStore.getState();
    if (!store.wsReady) store.setBackendStatus('offline');
  }
}

// --- Proactive polling fallback (same as before) ---------------------------

let _proactivePollTimer: ReturnType<typeof setInterval> | null = null;

async function pollProactiveMessages(): Promise<void> {
  const store = useStore.getState();
  if (store.backendStatus !== 'online') return;
  try {
    const headers: Record<string, string> = {};
    if (API_KEY) headers['X-API-Key'] = API_KEY;
    const r = await fetch(`${HTTP}/api/proactive/messages`, {
      headers,
      signal: AbortSignal.timeout(8000),
    });
    if (!r.ok) return;
    const data = await r.json();
    const msgs: any[] = Array.isArray(data) ? data : (data.messages || []);
    for (const m of msgs) {
      if (m.id && m.text) {
        store.addProactiveMessage({
          id: m.id,
          text: m.text,
          timestamp: m.timestamp || Date.now(),
        });
      }
    }
  } catch { /* non-fatal */ }
}

export function startProactivePoll(): void {
  if (_proactivePollTimer) return;
  // Only poll when the WebSocket is not alive. When WS is connected, the
  // backend pushes proactive messages directly and the stuck-signal path
  // handles contextual triggers — the old 60s unconditional poll is gone.
  setTimeout(() => {
    if (!useStore.getState().wsReady) pollProactiveMessages();
  }, 5000);
  _proactivePollTimer = setInterval(() => {
    if (!useStore.getState().wsReady) pollProactiveMessages();
  }, 60_000);
}

export function stopProactivePoll(): void {
  if (_proactivePollTimer) { clearInterval(_proactivePollTimer); _proactivePollTimer = null; }
}

export const PAGE_KEYWORDS =
  /\b(this page|this site|this article|this post|this video|current page|what('s| is) (this|the) (page|site|article)|summarize this|explain this|what does this (say|mean)|translate this|tldr|tl;dr)\b/i;
