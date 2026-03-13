export const HTTP = 'http://localhost:8000';
export const WS_URL = 'ws://localhost:8000/api/chat/stream';

export async function apiFetch(url: string, opts: RequestInit = {}): Promise<any> {
  const r = await fetch(url, opts);
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error((d as any).detail || `HTTP ${r.status}`);
  }
  return r.json();
}
