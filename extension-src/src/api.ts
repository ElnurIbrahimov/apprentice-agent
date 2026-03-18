const DEFAULT_HTTP = 'http://localhost:8000';
const DEFAULT_WS = 'ws://localhost:8000/api/chat/stream';

const ALLOWED_ORIGINS = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/;

/** Validate a URL is localhost-only; returns fallback if not. */
function validateLocalhostUrl(url: string, fallback: string): string {
  if (ALLOWED_ORIGINS.test(url)) return url;
  console.warn(`[Aura] Blocked non-localhost URL: ${url}, using fallback`);
  return fallback;
}

export let HTTP = DEFAULT_HTTP;
export let WS_URL = DEFAULT_WS;

/** Call once at startup if URLs may come from external config / storage. */
export function setUrls(http?: string, ws?: string) {
  HTTP = http ? validateLocalhostUrl(http, DEFAULT_HTTP) : DEFAULT_HTTP;
  WS_URL = ws ? validateLocalhostUrl(ws, DEFAULT_WS) : DEFAULT_WS;
}

export async function apiFetch(url: string, opts: RequestInit = {}): Promise<any> {
  if (!ALLOWED_ORIGINS.test(url)) {
    throw new Error(`Blocked non-localhost request: ${url}`);
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 30_000);
  try {
    const r = await fetch(url, { ...opts, signal: opts.signal ?? ctrl.signal });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error((d as any).detail || `HTTP ${r.status}`);
    }
    return r.json();
  } finally {
    clearTimeout(timer);
  }
}

/** Result type for safeFetch — never throws on network errors */
export type ApiResult<T = any> =
  | { ok: true; data: T }
  | { ok: false; error: string };

/**
 * Safe wrapper around apiFetch that catches network/backend errors
 * and returns a structured result instead of throwing.
 * Panels should use this to avoid crashing when backend is down.
 */
export async function safeFetch<T = any>(url: string, opts: RequestInit = {}): Promise<ApiResult<T>> {
  try {
    const data = await apiFetch(url, opts);
    return { ok: true, data };
  } catch (err: any) {
    const msg = err?.message || String(err);
    // Distinguish network errors from HTTP errors
    if (msg.includes('Failed to fetch') || msg.includes('NetworkError') || msg.includes('abort')) {
      return { ok: false, error: 'Backend offline' };
    }
    return { ok: false, error: msg };
  }
}
