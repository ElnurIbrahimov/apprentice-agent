const ALLOWED_ORIGINS = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?/;

export let HTTP = 'http://localhost:8000';
export let WS_URL = 'ws://localhost:8000/api/chat/stream';

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
