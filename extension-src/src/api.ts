import ext from './ext';

// ---------------------------------------------------------------------------
// Backend URL configuration
//
// Defaults to localhost for local development. Users can configure a remote
// server URL via the Settings panel, which persists to chrome.storage.local.
// On startup, initBackendUrl() loads the saved URL and updates HTTP + WS_URL.
// ---------------------------------------------------------------------------

const DEFAULT_HTTP = 'http://89.167.107.134';
const DEFAULT_WS = 'ws://89.167.107.134/api/chat/stream';

export let HTTP = DEFAULT_HTTP;
export let WS_URL = DEFAULT_WS;
export let API_KEY = 'i-L5ShpMkY2B7loNb8VS4EAAT-Ronh-K8cIgRILGjnQ';

/** Derive the WebSocket URL from an HTTP base URL. */
function deriveWsUrl(httpUrl: string): string {
  let base = httpUrl.replace(/\/+$/, '');
  if (base.startsWith('https://')) {
    base = 'wss://' + base.slice('https://'.length);
  } else if (base.startsWith('http://')) {
    base = 'ws://' + base.slice('http://'.length);
  }
  return base + '/api/chat/stream';
}

/**
 * Load saved backend URL and API key from chrome.storage.local.
 * Call this once at extension startup (before connecting WebSocket).
 * Returns a promise that resolves when HTTP/WS_URL/API_KEY are set.
 */
export function initBackendUrl(): Promise<void> {
  // Force server URL — route through nginx on port 80
  HTTP = 'http://89.167.107.134';
  WS_URL = 'ws://89.167.107.134/api/chat/stream';
  API_KEY = 'i-L5ShpMkY2B7loNb8VS4EAAT-Ronh-K8cIgRILGjnQ';
  return Promise.resolve();
}

/**
 * Update the backend URL at runtime. Persists to storage.
 * Pass empty string to reset to localhost default.
 */
export function setBackendUrl(url: string): void {
  const cleaned = url.trim().replace(/\/+$/, '');
  if (!cleaned) {
    HTTP = DEFAULT_HTTP;
    WS_URL = DEFAULT_WS;
    ext?.storage?.local?.remove(['backendUrl']);
  } else {
    HTTP = cleaned;
    WS_URL = deriveWsUrl(cleaned);
    ext?.storage?.local?.set({ backendUrl: cleaned });
  }
}

/**
 * Update the API key at runtime. Persists to storage.
 */
export function setApiKey(key: string): void {
  API_KEY = key.trim();
  if (API_KEY) {
    ext?.storage?.local?.set({ apiKey: API_KEY });
  } else {
    ext?.storage?.local?.remove(['apiKey']);
  }
}

/**
 * Get the currently configured backend URL (for display in settings).
 */
export function getBackendUrl(): string {
  return HTTP;
}

/**
 * Returns headers object with API key included (if set).
 * Use this when making raw fetch() calls instead of apiFetch.
 * Example: fetch(url, { headers: { ...getAuthHeaders(), 'Content-Type': 'application/json' } })
 */
export function getAuthHeaders(): Record<string, string> {
  return API_KEY ? { 'X-API-Key': API_KEY } : {};
}

export async function apiFetch(url: string, opts: RequestInit = {}): Promise<any> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 30_000);
  try {
    // Inject API key header if configured
    const headers: Record<string, string> = {};
    if (API_KEY) {
      headers['X-API-Key'] = API_KEY;
    }
    // Merge with any existing headers from opts
    const mergedHeaders = { ...headers, ...(opts.headers as Record<string, string> || {}) };

    const r = await fetch(url, {
      ...opts,
      headers: mergedHeaders,
      signal: opts.signal ?? ctrl.signal,
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error((d as any).detail || `HTTP ${r.status}`);
    }
    return r.json();
  } finally {
    clearTimeout(timer);
  }
}
