/**
 * Link Preview module — Link Preview on Hover
 * Migrated from content.ts lines ~3566-3726
 */

import type { ContentModule } from './types';

interface LinkPreviewData {
  title: string;
  description: string;
  domain: string;
}

interface LinkPreviewResponse {
  ok: boolean;
  title?: string;
  description?: string;
  domain?: string;
  error?: string;
}

interface OpenWithTextMessage {
  type: 'OPEN_WITH_TEXT';
  action: string;
  text: string;
  url: string;
  title: string;
}

const LP_CACHE_MAX = 50;

export function createLinkPreview(): ContentModule {
  let _ext!: typeof chrome;

  const _linkPreviewCache = new Map<string, LinkPreviewData>();

  function lpCacheSet(cacheUrl: string, cacheData: LinkPreviewData): void {
    if (_linkPreviewCache.size >= LP_CACHE_MAX) {
      const oldest = _linkPreviewCache.keys().next().value;
      if (oldest) _linkPreviewCache.delete(oldest);
    }
    _linkPreviewCache.set(cacheUrl, cacheData);
  }

  function lpCacheGet(cacheUrl: string): LinkPreviewData | undefined {
    const d = _linkPreviewCache.get(cacheUrl);
    if (d) { _linkPreviewCache.delete(cacheUrl); _linkPreviewCache.set(cacheUrl, d); }
    return d;
  }

  const lpHost: HTMLDivElement = document.createElement('div');
  lpHost.id = 'aura-link-preview-host';
  Object.assign(lpHost.style, { position: 'fixed', top: '0', left: '0', zIndex: '2147483646', pointerEvents: 'none' });
  document.documentElement.appendChild(lpHost);
  const lpShadow: ShadowRoot = lpHost.attachShadow({ mode: 'closed' });

  const lpCss: HTMLStyleElement = document.createElement('style');
  lpCss.textContent = [
    '@keyframes lp-in { from { opacity:0; transform:translateY(4px) scale(0.96); } to { opacity:1; transform:translateY(0) scale(1); } }',
    '@keyframes lp-shimmer { 0% { background-position:-200px 0; } 100% { background-position:200px 0; } }',
    '.lp-popup { position:fixed; width:320px; max-height:280px; background:rgba(10,8,24,0.92); backdrop-filter:blur(20px) saturate(1.5); -webkit-backdrop-filter:blur(20px) saturate(1.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:14px 16px 12px; pointer-events:auto; animation:lp-in 0.2s cubic-bezier(0.16,1,0.3,1) forwards; box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.05) inset; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column; gap:8px; }',
    '.lp-domain { display:inline-block; background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.25); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; color:rgba(160,148,210,0.9); letter-spacing:0.3px; max-width:fit-content; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }',
    '.lp-title { font-size:13px; font-weight:600; color:rgba(226,232,240,0.95); line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; }',
    '.lp-description { font-size:12px; font-weight:400; color:rgba(226,232,240,0.65); line-height:1.45; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin:0; }',
    '.lp-shimmer { height:12px; border-radius:4px; background:linear-gradient(90deg,rgba(124,58,237,0.08) 25%,rgba(124,58,237,0.18) 50%,rgba(124,58,237,0.08) 75%); background-size:400px 100%; animation:lp-shimmer 1.5s infinite linear; }',
    '.lp-shimmer.short { width:60%; } .lp-shimmer.long { width:90%; } .lp-shimmer+.lp-shimmer { margin-top:6px; }',
    '.lp-loading-label { font-size:11px; color:rgba(160,148,210,0.5); margin-bottom:4px; }',
    '.lp-actions { display:flex; gap:6px; margin-top:4px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }',
    '.lp-btn { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px 10px; font-size:11px; font-weight:500; font-family:inherit; color:rgba(200,180,255,0.9); cursor:pointer; transition:background 0.15s,border-color 0.15s,color 0.15s; white-space:nowrap; }',
    '.lp-btn:hover { background:rgba(124,58,237,0.25); border-color:rgba(124,58,237,0.4); color:#fff; }',
    '.lp-btn:active { background:rgba(124,58,237,0.35); }',
  ].join('\n');
  lpShadow.appendChild(lpCss);

  const lpBox: HTMLDivElement = document.createElement('div');
  lpShadow.appendChild(lpBox);

  let _lpPopup: HTMLDivElement | null = null;
  let _lpHoverTmr: ReturnType<typeof setTimeout> | null = null;
  let _lpDismissTmr: ReturnType<typeof setTimeout> | null = null;
  let _lpCurLink: HTMLAnchorElement | null = null;
  let _lpMouseIsDown = false;

  const onMouseDown = () => { _lpMouseIsDown = true; };
  const onMouseUp = () => { _lpMouseIsDown = false; };

  function lpIsExternal(a: HTMLAnchorElement): boolean {
    try { return new URL(a.href, location.href).hostname !== location.hostname; } catch { return false; }
  }

  function lpShouldShow(a: HTMLAnchorElement): boolean {
    const h = a.href || '';
    if (!h.startsWith('http://') && !h.startsWith('https://')) return false;
    try { const u = new URL(h, location.href); if (u.hostname === location.hostname && u.pathname === location.pathname && u.hash) return false; } catch { return false; }
    if ((a.textContent || '').trim().length < 10) return false;
    return lpIsExternal(a);
  }

  function lpRemove(): void { if (_lpPopup) { _lpPopup.remove(); _lpPopup = null; } _lpCurLink = null; }

  function lpCancelTimers(): void {
    if (_lpHoverTmr) { clearTimeout(_lpHoverTmr); _lpHoverTmr = null; }
    if (_lpDismissTmr) { clearTimeout(_lpDismissTmr); _lpDismissTmr = null; }
  }

  function lpStartDismiss(): void {
    if (_lpDismissTmr) clearTimeout(_lpDismissTmr);
    _lpDismissTmr = setTimeout(() => { lpRemove(); _lpDismissTmr = null; }, 300);
  }

  function lpCancelDismiss(): void { if (_lpDismissTmr) { clearTimeout(_lpDismissTmr); _lpDismissTmr = null; } }

  function lpPosition(a: HTMLAnchorElement): void {
    if (!_lpPopup) return;
    const r = a.getBoundingClientRect();
    _lpPopup.style.visibility = 'hidden'; _lpPopup.style.display = 'flex';
    const ph = _lpPopup.offsetHeight || 180;
    _lpPopup.style.visibility = '';
    let l = r.left + (r.width / 2) - 160;
    if (l < 8) l = 8; if (l + 320 > window.innerWidth - 8) l = window.innerWidth - 328;
    let t = r.bottom + 8;
    if (t + ph > window.innerHeight - 8) { t = r.top - ph - 8; if (t < 8) t = 8; }
    _lpPopup.style.top = Math.round(t) + 'px';
    _lpPopup.style.left = Math.round(l) + 'px';
  }

  function lpUpdate(lw: HTMLElement, te: HTMLElement, d: LinkPreviewData): void {
    lw.innerHTML = ''; lw.style.display = 'none';
    if (d.title && d.title !== te.textContent) te.textContent = d.title;
    if (d.description) { const de = document.createElement('div'); de.className = 'lp-description'; de.textContent = d.description; te.after(de); }
    if (_lpPopup && _lpCurLink) lpPosition(_lpCurLink);
  }

  function lpBuild(a: HTMLAnchorElement, href: string): void {
    lpRemove(); _lpCurLink = a;
    let dom = ''; try { dom = new URL(href).hostname; } catch { dom = href; }
    const txt = (a.textContent || '').trim();

    _lpPopup = document.createElement('div'); _lpPopup.className = 'lp-popup';

    const dEl = document.createElement('div'); dEl.className = 'lp-domain'; dEl.textContent = dom; _lpPopup.appendChild(dEl);
    const tEl = document.createElement('div'); tEl.className = 'lp-title'; tEl.textContent = txt; _lpPopup.appendChild(tEl);

    const lw = document.createElement('div');
    const ll = document.createElement('div'); ll.className = 'lp-loading-label'; ll.textContent = 'Loading preview\u2026';
    const s1 = document.createElement('div'); s1.className = 'lp-shimmer long';
    const s2 = document.createElement('div'); s2.className = 'lp-shimmer short';
    lw.appendChild(ll); lw.appendChild(s1); lw.appendChild(s2); _lpPopup.appendChild(lw);

    const acts = document.createElement('div'); acts.className = 'lp-actions';
    const ob = document.createElement('button'); ob.className = 'lp-btn'; ob.textContent = 'Open';
    ob.addEventListener('click', (ev: MouseEvent) => { ev.preventDefault(); ev.stopPropagation(); window.open(href, '_blank', 'noopener'); lpRemove(); });
    const sb = document.createElement('button'); sb.className = 'lp-btn'; sb.textContent = 'Summarize in AURA';
    sb.addEventListener('click', (ev: MouseEvent) => {
      ev.preventDefault(); ev.stopPropagation();
      try {
        _ext.runtime.sendMessage({ type: 'OPEN_WITH_TEXT', action: 'summarize', text: 'Summarize this page: ' + href, url: href, title: txt } as OpenWithTextMessage);
      } catch (_e) { /* context invalidated */ }
      lpRemove();
    });
    acts.appendChild(ob); acts.appendChild(sb); _lpPopup.appendChild(acts);

    _lpPopup.addEventListener('mouseenter', lpCancelDismiss);
    _lpPopup.addEventListener('mouseleave', lpStartDismiss);
    lpBox.appendChild(_lpPopup); lpPosition(a);

    const c = lpCacheGet(href);
    if (c) { lpUpdate(lw, tEl, c); return; }

    try {
      _ext.runtime.sendMessage({ type: 'LINK_PREVIEW', url: href } as any, (rsp: LinkPreviewResponse) => {
        if (_ext.runtime.lastError || !rsp) return;
        if (!_lpPopup || _lpCurLink !== a) return;
        const pd: LinkPreviewData = { title: rsp.title || txt, description: rsp.description || '', domain: rsp.domain || dom };
        lpCacheSet(href, pd); lpUpdate(lw, tEl, pd);
      });
    } catch { /* invalidated */ }
  }

  const onMouseOver = (me: MouseEvent) => {
    if (_lpMouseIsDown) return;
    const a = (me.target as HTMLElement).closest('a') as HTMLAnchorElement | null;
    if (!a || !lpShouldShow(a)) return;
    if (_lpCurLink === a && _lpPopup) { lpCancelDismiss(); return; }
    lpCancelTimers();
    _lpHoverTmr = setTimeout(() => { if (_lpMouseIsDown) return; lpBuild(a, a.href); _lpHoverTmr = null; }, 800);
  };

  const onMouseOut = (me: MouseEvent) => {
    const a = (me.target as HTMLElement).closest('a') as HTMLAnchorElement | null;
    if (a && a === _lpCurLink) { const rel = me.relatedTarget as Node | null; if (rel && lpHost.contains(rel)) return; lpStartDismiss(); }
    if (a && _lpHoverTmr) lpCancelTimers();
  };

  const onScroll = () => {
    if (_lpPopup && _lpCurLink) { const r = _lpCurLink.getBoundingClientRect(); if (r.bottom < 0 || r.top > window.innerHeight) { lpCancelTimers(); lpRemove(); } else { lpPosition(_lpCurLink); } }
  };

  return {
    init(_container: HTMLElement, _store: any, ext: typeof chrome) {
      _ext = ext;

      document.addEventListener('mousedown', onMouseDown, true);
      document.addEventListener('mouseup', onMouseUp, true);
      document.addEventListener('mouseover', onMouseOver, true);
      document.addEventListener('mouseout', onMouseOut, true);
      window.addEventListener('scroll', onScroll, { passive: true });
    },

    destroy() {
      document.removeEventListener('mousedown', onMouseDown, true);
      document.removeEventListener('mouseup', onMouseUp, true);
      document.removeEventListener('mouseover', onMouseOver, true);
      document.removeEventListener('mouseout', onMouseOut, true);
      window.removeEventListener('scroll', onScroll);
      lpCancelTimers();
      lpRemove();
      lpHost.remove();
    },
  };
}
