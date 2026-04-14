/// <reference types="chrome" />

/**
 * AURA Offscreen Document
 *
 * Hosts the single persistent WebSocket to the Aura backend.
 * The service worker creates this document on install; it stays alive
 * independent of the sidebar's lifecycle so Hetzner push messages
 * (proactive, hand_event, hand_approval_request) reach the user even
 * when the sidebar is closed.
 *
 * Incoming messages are broadcast via `chrome.runtime.sendMessage` as
 * `{ type: 'AURA_WS_IN', payload: <raw data string> }`. The sidebar's
 * `ws.ts` applies them to the store; the service worker independently
 * fires `chrome.notifications` for high-signal event types.
 *
 * Sidebar → backend sends arrive as `AURA_WS_OUT { payload }` and are
 * forwarded to `socket.send(payload)`.
 *
 * Status updates (`AURA_WS_STATUS { readyState }`) let the sidebar's
 * proxy socket expose a WebSocket-shaped API without holding the real
 * socket.
 */

const WS_CONNECT_TIMEOUT = 5_000;
const PING_INTERVAL = 25_000;
const PING_GRACE = 5_000;

let socket: WebSocket | null = null;
let wsUrl = '';
let apiKey = '';
let retryDelay = 1000;
let consecutiveFailures = 0;
let connTimer: ReturnType<typeof setTimeout> | null = null;
let pingTimer: ReturnType<typeof setInterval> | null = null;
let lastMessageTime = 0;
let reconnectPending: ReturnType<typeof setTimeout> | null = null;

function deriveWsUrl(httpUrl: string): string {
  let base = httpUrl.replace(/\/+$/, '');
  if (base.startsWith('https://')) base = 'wss://' + base.slice('https://'.length);
  else if (base.startsWith('http://')) base = 'ws://' + base.slice('http://'.length);
  return base + '/api/chat/stream';
}

function broadcast(message: Record<string, unknown>): void {
  chrome.runtime.sendMessage(message).catch(() => {});
}

function broadcastStatus(): void {
  broadcast({
    type: 'AURA_WS_STATUS',
    readyState: socket ? socket.readyState : 3 /* CLOSED */,
    consecutiveFailures,
  });
}

function stopPing(): void {
  if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
}

function startPing(): void {
  stopPing();
  pingTimer = setInterval(() => {
    if (socket?.readyState === WebSocket.OPEN && (Date.now() - lastMessageTime) > PING_GRACE) {
      try { socket.send(JSON.stringify({ type: 'ping' })); } catch { /* noop */ }
    }
  }, PING_INTERVAL);
}

async function loadConfig(): Promise<void> {
  // `chrome.storage` is not guaranteed to be exposed to offscreen documents
  // across Chrome versions — proxy through the service worker which always
  // has access. Fall through to the default server if anything fails.
  let httpUrl = 'https://aura-elnur.duckdns.org';
  let key = '';
  try {
    // Prefer direct access if available (newer Chrome)…
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      const data = await chrome.storage.local.get(['backendUrl', 'apiKey']);
      const u = (data.backendUrl as string | undefined)?.trim()?.replace(/\/+$/, '');
      if (u) httpUrl = u;
      key = ((data.apiKey as string | undefined) || '').trim();
    } else {
      // …otherwise ask the service worker.
      const resp = await chrome.runtime.sendMessage({ type: 'AURA_OFFSCREEN_CONFIG_REQUEST' });
      if (resp?.backendUrl) httpUrl = String(resp.backendUrl).trim().replace(/\/+$/, '');
      if (resp?.apiKey) key = String(resp.apiKey).trim();
    }
  } catch { /* use defaults */ }
  wsUrl = deriveWsUrl(httpUrl);
  apiKey = key;
}

function connect(): void {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

  if (!wsUrl) {
    // Config not loaded yet — try again in a moment.
    setTimeout(connect, 200);
    return;
  }

  try {
    socket = new WebSocket(wsUrl);
  } catch (err) {
    console.warn('[Aura/offscreen] WebSocket construction failed:', err);
    scheduleReconnect();
    return;
  }

  connTimer = setTimeout(() => {
    connTimer = null;
    if (socket && socket.readyState !== WebSocket.OPEN) {
      try { socket.close(); } catch { /* noop */ }
    }
  }, WS_CONNECT_TIMEOUT);

  socket.onopen = () => {
    if (connTimer) { clearTimeout(connTimer); connTimer = null; }
    if (apiKey && socket) socket.send(JSON.stringify({ type: 'auth', api_key: apiKey }));
    if (socket) socket.send(JSON.stringify({ type: 'ready_check' }));
    retryDelay = 1000;
    consecutiveFailures = 0;
    startPing();
    broadcastStatus();
  };

  socket.onclose = () => {
    if (connTimer) { clearTimeout(connTimer); connTimer = null; }
    stopPing();
    consecutiveFailures++;
    broadcastStatus();
    broadcast({ type: 'AURA_WS_CLOSED' });
    if (consecutiveFailures <= 8) {
      scheduleReconnect();
    } else {
      broadcast({ type: 'AURA_WS_ABANDONED' });
    }
  };

  socket.onerror = () => {
    broadcastStatus();
  };

  socket.onmessage = (ev: MessageEvent) => {
    lastMessageTime = Date.now();
    broadcast({ type: 'AURA_WS_IN', payload: typeof ev.data === 'string' ? ev.data : String(ev.data) });
  };
}

function scheduleReconnect(): void {
  if (reconnectPending) return;
  const jitter = Math.random() * retryDelay * 0.3;
  reconnectPending = setTimeout(() => {
    reconnectPending = null;
    retryDelay = Math.min(retryDelay * 2, 30_000);
    connect();
  }, retryDelay + jitter);
}

function forceReconnect(): void {
  if (reconnectPending) { clearTimeout(reconnectPending); reconnectPending = null; }
  retryDelay = 1000;
  consecutiveFailures = 0;
  if (socket) {
    try { socket.close(); } catch { /* noop */ }
    socket = null;
  }
  connect();
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (!msg || typeof msg !== 'object') return false;

  switch (msg.type) {
    case 'AURA_WS_OUT':
      if (socket?.readyState === WebSocket.OPEN && typeof msg.payload === 'string') {
        try { socket.send(msg.payload); } catch { /* noop */ }
      }
      sendResponse({ ok: true });
      return true;

    case 'AURA_WS_RECONNECT':
      loadConfig().then(() => forceReconnect()).catch(() => forceReconnect());
      sendResponse({ ok: true });
      return true;

    case 'AURA_WS_STATUS_REQUEST':
      broadcastStatus();
      sendResponse({ ok: true });
      return true;

    case 'AURA_WS_CONFIG_CHANGED':
      loadConfig().then(() => forceReconnect()).catch(() => {});
      sendResponse({ ok: true });
      return true;
  }
  return false;
});

// Watch storage directly when available; silently skip if not exposed in
// this offscreen context. The service worker also relays config changes
// via `AURA_WS_CONFIG_CHANGED` runtime messages as a backup.
if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.onChanged) {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local') return;
    if (changes.backendUrl || changes.apiKey) {
      loadConfig().then(() => forceReconnect()).catch(() => {});
    }
  });
}

// Boot
loadConfig().then(() => connect()).catch(() => connect());
