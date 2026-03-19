import ext from './ext';

// ---------------------------------------------------------------------------
// Backend URL configuration
//
// Defaults to the remote server. Users can configure a different URL via the
// Settings panel, which persists to chrome.storage.local.
// On startup, initBackendUrl() loads the saved URL and updates HTTP + WS_URL.
// ---------------------------------------------------------------------------

const DEFAULT_HTTP = 'http://89.167.107.134';
const DEFAULT_API_KEY = 'i-L5ShpMkY2B7loNb8VS4EAAT-Ronh-K8cIgRILGjnQ';

export let HTTP = DEFAULT_HTTP;
export let WS_URL = deriveWsUrl(DEFAULT_HTTP);
export let API_KEY = DEFAULT_API_KEY;

/** Derive the WebSocket URL from an HTTP base URL. */
export function deriveWsUrl(httpUrl: string): string {
  let base = httpUrl.replace(/\/+$/, '');
  if (base.startsWith('https://')) {
    base = 'wss://' + base.slice('https://'.length);
  } else if (base.startsWith('http://')) {
    base = 'ws://' + base.slice('http://'.length);
  }
  return base + '/api/chat/stream';
}

/**
 * Extract a display-friendly server label from the HTTP URL.
 * e.g. "http://89.167.107.134" -> "89.167.107.134"
 *      "http://localhost:8000" -> "localhost:8000"
 */
export function getServerLabel(): string {
  try {
    const u = new URL(HTTP);
    return u.host;
  } catch {
    return HTTP.replace(/^https?:\/\//, '').replace(/\/+$/, '');
  }
}

/**
 * Load saved backend URL and API key from chrome.storage.local.
 * Call this once at extension startup (before connecting WebSocket).
 * Returns a promise that resolves when HTTP/WS_URL/API_KEY are set.
 */
export function initBackendUrl(): Promise<void> {
  return new Promise((resolve) => {
    if (!ext?.storage?.local) {
      // No storage API (dev mode / non-extension context) — use defaults
      HTTP = DEFAULT_HTTP;
      WS_URL = deriveWsUrl(HTTP);
      API_KEY = DEFAULT_API_KEY;
      resolve();
      return;
    }
    ext.storage.local.get(['backendUrl', 'apiKey'], (d: any) => {
      const savedUrl = d?.backendUrl?.trim?.();
      const savedKey = d?.apiKey?.trim?.();

      if (savedUrl) {
        HTTP = savedUrl.replace(/\/+$/, '');
      } else {
        HTTP = DEFAULT_HTTP;
      }
      WS_URL = deriveWsUrl(HTTP);

      if (savedKey) {
        API_KEY = savedKey;
      } else {
        API_KEY = DEFAULT_API_KEY;
      }
      resolve();
    });
  });
}

/**
 * Update the backend URL at runtime. Persists to storage.
 * Pass empty string to reset to localhost default.
 */
export function setBackendUrl(url: string): void {
  const cleaned = url.trim().replace(/\/+$/, '');
  if (!cleaned) {
    HTTP = DEFAULT_HTTP;
    WS_URL = deriveWsUrl(DEFAULT_HTTP);
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
