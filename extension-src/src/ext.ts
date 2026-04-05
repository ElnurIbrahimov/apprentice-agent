// Firefox/Chrome compatibility shim
declare const browser: typeof chrome | undefined;

const ext: typeof chrome =
  typeof browser !== 'undefined' ? (browser as typeof chrome) : chrome;

export default ext;

// Page content cache — 30s TTL
const _pageCache = new Map<string, { resp: any; ts: number }>();
const PAGE_CACHE_TTL = 30000;

export async function getPageContentCached(): Promise<any> {
  const now = Date.now();
  for (const [key, val] of _pageCache) {
    if (now - val.ts < PAGE_CACHE_TTL) return val.resp;
    _pageCache.delete(key);
  }
  const resp = await new Promise<any>(r =>
    ext.runtime.sendMessage({ type: 'GET_PAGE_CONTENT' }, r)
  );
  if (resp?.ok && resp.url) _pageCache.set(resp.url, { resp, ts: now });
  return resp;
}

export function clearPageContentCache(): void {
  _pageCache.clear();
}

export async function getCurrentTab(): Promise<any> {
  return new Promise(r => ext.runtime.sendMessage({ type: 'GET_CURRENT_TAB' }, r));
}

export function sendMsg(msg: any): Promise<any> {
  return new Promise(r => ext.runtime.sendMessage(msg, r));
}
