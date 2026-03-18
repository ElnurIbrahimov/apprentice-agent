/**
 * AURA Chrome Extension — Content Script (content.ts)
 * Injects a Shadow DOM floating toolbar on text selection.
 * Runs on every page via manifest content_scripts.
 */

// ── Interfaces ──────────────────────────────────────────────────────────────

interface SerializedElement {
  index: number;
  type: string;
  text: string;
  selector: string;
  href: string;
}

interface ExecActionParams {
  action: 'click' | 'type' | 'scroll' | 'selectOption';
  selector?: string;
  text?: string;
  url?: string;
  amount?: number;
  value?: string;
}

interface ExecActionResult {
  ok: boolean;
  error?: string;
}

interface OcrOverlayResult {
  ok: boolean;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  dpr?: number;
}

interface ExtractPageResponse {
  text: string;
  url: string;
  title: string;
  wordCount: number;
  isPdf?: boolean;
  isYouTube?: boolean;
  videoTitle?: string;
  transcript?: string;
}

interface GetDomResponse {
  ok: boolean;
  dom: SerializedElement[];
  url: string;
  title: string;
}

// Messages sent from content → background
interface OpenPanelMessage {
  type: 'OPEN_PANEL';
  panel: string;
}

interface OpenWithTextMessage {
  type: 'OPEN_WITH_TEXT';
  action: string;
  text: string;
  url: string;
  title: string;
}

interface SaveKnowledgeMessage {
  type: 'SAVE_KNOWLEDGE';
  text: string;
  url: string;
  title: string;
}

type OutboundMessage = OpenPanelMessage | OpenWithTextMessage | SaveKnowledgeMessage;

// Messages received from background → content
interface ExtractPageMsg {
  type: 'EXTRACT_PAGE';
}

interface GetDomMsg {
  type: 'GET_DOM';
}

interface ExecActionMsg {
  type: 'EXEC_ACTION';
  action: ExecActionParams;
}

interface ShowOcrOverlayMsg {
  type: 'SHOW_OCR_OVERLAY';
  dataUrl: string;
}

type InboundMessage = ExtractPageMsg | GetDomMsg | ExecActionMsg | ShowOcrOverlayMsg;

interface SaveKnowledgeResponse {
  ok: boolean;
}

interface DockItemDef {
  svg: string;
  action: string;
  tip: string;
}

interface ToolbarButtonDef {
  label: string;
  action: string;
}

// Extend Window for our global guard flag
declare global {
  interface Window {
    __auraToolbarMounted?: boolean;
  }
}

// ── Firefox compatibility shim ──────────────────────────────────────────────

declare const browser: typeof chrome | undefined;

const ext: typeof chrome =
  typeof browser !== 'undefined' ? (browser as typeof chrome) : chrome;

// ── Safe send ───────────────────────────────────────────────────────────────

/**
 * Wraps sendMessage to gracefully handle "Extension context invalidated" (happens
 * when the extension is reloaded while this content script is still running on a tab).
 * On invalidation, removes orphaned AURA UI elements from the page.
 */
function safeSend(msg: OutboundMessage, cb?: (response: SaveKnowledgeResponse) => void): void {
  try {
    if (cb) {
      ext.runtime.sendMessage(msg, cb);
    } else {
      ext.runtime.sendMessage(msg);
    }
  } catch (e: unknown) {
    const err = e as Error | undefined;
    if (
      err?.message?.includes('Extension context invalidated') ||
      err?.message?.includes('context invalidated')
    ) {
      document.getElementById('aura-dock-host')?.remove();
      document.getElementById('aura-host')?.remove();
      window.__auraToolbarMounted = false;
    }
  }
}

// ── Main IIFE ───────────────────────────────────────────────────────────────

(function (): void {
  // Guard: if already mounted in this window, do nothing (handles duplicate injection)
  if (window.__auraToolbarMounted) return;
  window.__auraToolbarMounted = true;

  // Remove stale elements from any previous injection (handles extension reload)
  const _prevDock = document.getElementById('aura-dock-host');
  if (_prevDock) _prevDock.remove();
  const _prevHost = document.getElementById('aura-host');
  if (_prevHost) _prevHost.remove();

  // ── Shadow DOM Host ─────────────────────────────────────────────────────

  const host: HTMLDivElement = document.createElement('div');
  host.id = 'aura-host';
  Object.assign(host.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    zIndex: '2147483647',
    pointerEvents: 'none',
  });
  document.documentElement.appendChild(host);

  const shadow: ShadowRoot = host.attachShadow({ mode: 'open' });

  // ── Styles (inside Shadow DOM — isolated from page) ───────────────────

  const style: HTMLStyleElement = document.createElement('style');
  style.textContent = `
    #toolbar {
      display: none;
      position: fixed;
      background: #1a1a2e;
      border: 1px solid #7c3aed;
      border-radius: 8px;
      padding: 4px 6px;
      gap: 4px;
      box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
      pointer-events: auto;
      z-index: 2147483647;
      align-items: center;
    }
    #toolbar.visible {
      display: flex;
    }
    .aura-btn {
      background: transparent;
      border: none;
      color: #e2e8f0;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      padding: 4px 8px;
      border-radius: 5px;
      cursor: pointer;
      white-space: nowrap;
      transition: background 0.15s;
    }
    .aura-btn:hover {
      background: #7c3aed;
      color: #fff;
    }
    .aura-divider {
      width: 1px;
      height: 16px;
      background: #3d3d5c;
    }
    #toast {
      display: none;
      position: fixed;
      background: #059669;
      color: #fff;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      padding: 6px 12px;
      border-radius: 6px;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    #toast.visible {
      display: block;
    }
  `;
  shadow.appendChild(style);

  // ── Toolbar DOM ───────────────────────────────────────────────────────────

  const toolbar: HTMLDivElement = document.createElement('div');
  toolbar.id = 'toolbar';

  const buttons: ToolbarButtonDef[] = [
    { label: '✦ Explain', action: 'explain' },
    { label: '◈ Summarize', action: 'summarize' },
    { label: '◉ Ask AURA', action: 'ask' },
  ];

  buttons.forEach((btn, i) => {
    const el: HTMLButtonElement = document.createElement('button');
    el.className = 'aura-btn';
    el.textContent = btn.label;
    el.dataset.action = btn.action;
    toolbar.appendChild(el);
    if (i < buttons.length - 1) {
      const div: HTMLDivElement = document.createElement('div');
      div.className = 'aura-divider';
      toolbar.appendChild(div);
    }
  });

  // Save button with divider
  const divSave: HTMLDivElement = document.createElement('div');
  divSave.className = 'aura-divider';
  toolbar.appendChild(divSave);

  const saveBtn: HTMLButtonElement = document.createElement('button');
  saveBtn.className = 'aura-btn';
  saveBtn.textContent = '⊕ Save';
  saveBtn.dataset.action = 'save';
  toolbar.appendChild(saveBtn);

  shadow.appendChild(toolbar);

  // ── Toast ─────────────────────────────────────────────────────────────────

  const toast: HTMLDivElement = document.createElement('div');
  toast.id = 'toast';
  shadow.appendChild(toast);

  // ── Floating Action Dock (Shadow DOM isolated) ────────

  const dockShadowHost: HTMLDivElement = document.createElement('div');
  dockShadowHost.id = 'aura-dock-shadow';
  Object.assign(dockShadowHost.style, { position: 'fixed', right: '0', top: '0', zIndex: '2147483647', pointerEvents: 'none' });
  document.body.appendChild(dockShadowHost);
  const dockShadow = dockShadowHost.attachShadow({ mode: 'closed' });

  const dockHost: HTMLDivElement = document.createElement('div');
  dockHost.id = 'aura-dock-host';
  Object.assign(dockHost.style, {
    position: 'fixed',
    right: '0',
    top: '50%',
    transform: 'translateY(-50%)',
    zIndex: '2147483647',
    pointerEvents: 'auto',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0',
    padding: '7px 4px',
    background: 'rgba(7, 5, 18, 0.92)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: '1px solid rgba(124, 58, 237, 0.3)',
    borderRight: 'none',
    borderRadius: '12px 0 0 12px',
    boxShadow: '-3px 0 20px rgba(0,0,0,0.5)',
    transition: 'border-color 0.2s',
    boxSizing: 'border-box',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  });
  dockShadow.appendChild(dockHost);

  // Logo — always visible
  const dockLogo: HTMLDivElement = document.createElement('div');
  Object.assign(dockLogo.style, {
    width: '32px',
    height: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'rgba(160, 148, 210, 0.9)',
    cursor: 'default',
    flexShrink: '0',
  });
  dockLogo.innerHTML = `<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
  dockHost.appendChild(dockLogo);

  // Action buttons container — hidden until hover
  const dockActions: HTMLDivElement = document.createElement('div');
  Object.assign(dockActions.style, {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '3px',
    overflow: 'hidden',
    maxHeight: '0',
    opacity: '0',
    transition: 'max-height 0.25s ease, opacity 0.2s ease, padding-top 0.2s ease',
    paddingTop: '0',
  });
  dockHost.appendChild(dockActions);

  const _dockItems: (DockItemDef | null)[] = [
    { svg: '<path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/>', action: 'dock-chat', tip: 'Chat with AURA' },
    { svg: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>', action: 'dock-search', tip: 'Search' },
    null,
    { svg: '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><path d="M9 21V9"/>', action: 'dock-thispage', tip: 'This Page' },
    { svg: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: 'dock-translate', tip: 'Translate' },
    null,
    { svg: '<path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>', action: 'dock-save', tip: 'Save to Memory' },
  ];

  _dockItems.forEach((item: DockItemDef | null) => {
    if (!item) {
      const sep: HTMLDivElement = document.createElement('div');
      Object.assign(sep.style, {
        width: '18px',
        height: '1px',
        background: 'rgba(255,255,255,0.08)',
        margin: '2px 0',
        flexShrink: '0',
      });
      dockActions.appendChild(sep);
      return;
    }
    const btn: HTMLButtonElement = document.createElement('button');
    btn.dataset.action = item.action;
    btn.title = item.tip;
    Object.assign(btn.style, {
      width: '32px',
      height: '32px',
      minWidth: '32px',
      minHeight: '32px',
      borderRadius: '8px',
      background: 'transparent',
      border: 'none',
      padding: '0',
      margin: '0',
      color: 'rgba(160, 148, 210, 0.6)',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      flexShrink: '0',
      boxSizing: 'border-box',
      outline: 'none',
    });
    btn.innerHTML = `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${item.svg}</svg>`;
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(124, 58, 237, 0.2)';
      btn.style.color = 'rgba(224, 214, 255, 1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'transparent';
      btn.style.color = 'rgba(160, 148, 210, 0.6)';
    });
    dockActions.appendChild(btn);
  });

  // Expand/collapse on hover via JS (more reliable than CSS :hover on injected elements)
  dockHost.addEventListener('mouseenter', () => {
    dockActions.style.maxHeight = '320px';
    dockActions.style.opacity = '1';
    dockActions.style.paddingTop = '5px';
    dockHost.style.borderColor = 'rgba(124, 58, 237, 0.5)';
    dockHost.style.boxShadow = '-4px 0 28px rgba(0,0,0,0.55)';
  });
  dockHost.addEventListener('mouseleave', () => {
    dockActions.style.maxHeight = '0';
    dockActions.style.opacity = '0';
    dockActions.style.paddingTop = '0';
    dockHost.style.borderColor = 'rgba(124, 58, 237, 0.3)';
    dockHost.style.boxShadow = '-3px 0 20px rgba(0,0,0,0.5)';
  });

  // Click handler
  dockHost.addEventListener('click', (e: MouseEvent) => {
    const target = e.target as HTMLElement;
    const btn: HTMLElement | null = target.closest('[data-action]');
    if (!btn) return;
    const action: string = (btn as HTMLElement).dataset.action!;
    const url: string = window.location.href;
    const title: string = document.title;

    if (action === 'dock-chat') {
      safeSend({ type: 'OPEN_PANEL', panel: 'chat' });
    } else if (action === 'dock-search') {
      safeSend({ type: 'OPEN_PANEL', panel: 'search' });
    } else if (action === 'dock-translate') {
      safeSend({ type: 'OPEN_PANEL', panel: 'translate' });
    } else if (action === 'dock-thispage') {
      const extracted = extractMainContent();
      safeSend({ type: 'OPEN_WITH_TEXT', action: 'ask', text: extracted.text, url, title });
    } else if (action === 'dock-save') {
      const selText: string = getSelectionText();
      const textToSave: string = selText || `${title}\n${url}`;
      safeSend(
        { type: 'SAVE_KNOWLEDGE', text: textToSave, url, title },
        (response: SaveKnowledgeResponse) => {
          if (response && response.ok) showToast('Saved to AURA memory ✓');
          else showToast('Save failed — is backend running?', 3000);
        }
      );
    }
  });

  let _toastTimer: ReturnType<typeof setTimeout> | null = null;

  function showToast(message: string, durationMs: number = 2000): void {
    if (_toastTimer) { clearTimeout(_toastTimer); _toastTimer = null; }
    toast.textContent = message;
    toast.classList.add('visible');
    // Position toast: near toolbar if visible, otherwise top-center
    if (toolbar.classList.contains('visible') && toolbar.style.top) {
      toast.style.top = (parseInt(toolbar.style.top) + 40) + 'px';
      toast.style.left = toolbar.style.left;
    } else {
      toast.style.top = '20px';
      toast.style.left = Math.round(window.innerWidth / 2 - 100) + 'px';
    }
    _toastTimer = setTimeout(() => { toast.classList.remove('visible'); _toastTimer = null; }, durationMs);
  }

  // ── Selection Handling ──────────────────────────────────────────────────

  function getSelectionText(): string {
    const sel: Selection | null = window.getSelection();
    if (!sel || sel.rangeCount === 0) return '';
    return sel.toString().trim();
  }

  function positionToolbar(): void {
    const sel: Selection | null = window.getSelection();
    if (!sel || sel.rangeCount === 0) return;
    const range: Range = sel.getRangeAt(0);
    const rect: DOMRect = range.getBoundingClientRect();
    const TOOLBAR_HEIGHT = 38;
    const MARGIN = 8;
    let left: number = rect.left + (rect.width / 2) - 100;
    if (left < 4) left = 4;
    if (left + 200 > window.innerWidth) left = window.innerWidth - 204;
    toolbar.style.top = `${Math.round(rect.top - TOOLBAR_HEIGHT - MARGIN)}px`;
    toolbar.style.left = `${Math.round(left)}px`;
  }

  function showToolbar(): void {
    toolbar.classList.add('visible');
    positionToolbar();
    host.style.pointerEvents = 'auto';
  }

  function hideToolbar(): void {
    toolbar.classList.remove('visible');
    host.style.pointerEvents = 'none';
  }

  document.addEventListener('mouseup', () => {
    setTimeout(() => {
      const text: string = getSelectionText();
      if (text.length > 0) showToolbar();
      else hideToolbar();
    }, 50);
  });

  document.addEventListener('mousedown', (e: MouseEvent) => {
    if (!host.contains(e.target as Node)) hideToolbar();
  });

  document.addEventListener('selectionchange', () => {
    if (getSelectionText().length === 0) hideToolbar();
  });

  // ── Button Handlers ───────────────────────────────────────────────────

  toolbar.addEventListener('click', (e: MouseEvent) => {
    const target = e.target as HTMLElement;
    const btn: HTMLElement | null = target.closest('.aura-btn');
    if (!btn) return;
    const action: string = (btn as HTMLElement).dataset.action!;
    const text: string = getSelectionText();
    if (!text) return;
    const url: string = window.location.href;
    const title: string = document.title;

    if (action === 'save') {
      safeSend(
        { type: 'SAVE_KNOWLEDGE', text, url, title },
        (response: SaveKnowledgeResponse) => {
          if (response && response.ok) showToast('Saved to AURA memory ✓');
          else showToast('Save failed — is backend running?', 3000);
        }
      );
    } else {
      safeSend({ type: 'OPEN_WITH_TEXT', action, text, url, title });
    }
    hideToolbar();
  });

  // ── DOM Serializer (for Browser Agent) ────────────────────────────────

  function bestSelector(el: HTMLElement): string {
    if (el.id) return '#' + CSS.escape(el.id);
    const al: string | null = el.getAttribute('aria-label');
    if (al) return `[aria-label="${al}"]`;
    const path: string[] = [];
    let cur: HTMLElement | null = el;
    for (let i = 0; i < 4 && cur && cur !== document.body; i++, cur = cur.parentElement) {
      const s: string = cur.tagName.toLowerCase();
      if (cur.id) { path.unshift('#' + CSS.escape(cur.id)); break; }
      const siblings: Element[] = [...(cur.parentElement?.children || [])];
      const idx: number = siblings.indexOf(cur) + 1;
      path.unshift(s + ':nth-child(' + idx + ')');
    }
    return path.join('>');
  }

  function serializeDOM(): SerializedElement[] {
    const els: SerializedElement[] = [];
    // Stop collecting once we have 80 visible interactive elements — avoids iterating thousands of nodes
    const nodes: NodeListOf<Element> = document.querySelectorAll(
      'a,button,input,textarea,select,[role="button"],[onclick]'
    );
    let idx = 0;
    for (const el of nodes) {
      if (els.length >= 80) break;
      const r: DOMRect = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const htmlEl = el as HTMLElement;
      const inputEl = el as HTMLInputElement;
      els.push({
        index: idx++,
        type: el.tagName.toLowerCase(),
        text: (htmlEl.innerText || inputEl.value || inputEl.placeholder || htmlEl.title || '').slice(0, 80).trim(),
        selector: bestSelector(htmlEl),
        href: (el as HTMLAnchorElement).href || '',
      });
    }
    return els;
  }

  function execAction(action: ExecActionParams): ExecActionResult {
    if (action.action === 'scroll') {
      window.scrollBy(0, action.amount || 300);
      return { ok: true };
    }
    let el: Element | null;
    try {
      el = document.querySelector(action.selector!);
    } catch (e) {
      return { ok: false, error: 'Invalid selector: ' + action.selector };
    }
    if (!el) return { ok: false, error: 'Element not found: ' + action.selector };
    if (action.action === 'click') { (el as HTMLElement).click(); return { ok: true }; }
    if (action.action === 'type') {
      (el as HTMLElement).focus();
      (el as HTMLInputElement).value = action.text || '';
      el.dispatchEvent(new Event('input', { bubbles: true }));
      el.dispatchEvent(new Event('change', { bubbles: true }));
      return { ok: true };
    }
    if (action.action === 'selectOption') {
      if (el.tagName.toLowerCase() !== 'select') return { ok: false, error: 'Element is not a <select>' };
      const selectEl = el as HTMLSelectElement;
      const opt: HTMLOptionElement | undefined = [...selectEl.options].find(
        (o: HTMLOptionElement) => o.value === action.value || o.text === action.value
      );
      if (!opt) return { ok: false, error: 'Option not found: ' + action.value };
      selectEl.value = opt.value;
      selectEl.dispatchEvent(new Event('change', { bubbles: true }));
      return { ok: true };
    }
    return { ok: false, error: 'Unknown action: ' + action.action };
  }

  // ── OCR Overlay ───────────────────────────────────────────────────────

  function showOcrOverlay(
    dataUrl: string,
    sendResponse: (result: OcrOverlayResult) => void
  ): void {
    // Create fullscreen overlay
    const overlay: HTMLDivElement = document.createElement('div');
    Object.assign(overlay.style, {
      position: 'fixed', top: '0', left: '0', width: '100vw', height: '100vh',
      zIndex: '2147483646', cursor: 'crosshair', background: 'rgba(0,0,0,0.4)',
    });

    // Show the screenshot as background for reference
    const img: HTMLImageElement = new Image();
    img.src = dataUrl;
    img.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;';
    overlay.appendChild(img);

    // Canvas for drawing selection rect
    const canvas: HTMLCanvasElement = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    Object.assign(canvas.style, {
      position: 'absolute', top: '0', left: '0', width: '100%', height: '100%',
    });
    overlay.appendChild(canvas);
    const ctx: CanvasRenderingContext2D | null = canvas.getContext('2d');

    const hint: HTMLDivElement = document.createElement('div');
    Object.assign(hint.style, {
      position: 'fixed', top: '12px', left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(0,0,0,0.75)', color: '#fff', padding: '6px 14px',
      borderRadius: '6px', fontSize: '13px', pointerEvents: 'none',
    });
    hint.textContent = 'Drag to select region • Press Esc to cancel';
    overlay.appendChild(hint);

    document.body.appendChild(overlay);

    let startX = 0;
    let startY = 0;
    let dragging = false;
    const dpr: number = window.devicePixelRatio || 1;

    function drawRect(x: number, y: number, w: number, h: number): void {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = '#7c3aed';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(124,58,237,0.12)';
      ctx.fillRect(x, y, w, h);
    }

    overlay.addEventListener('mousedown', (e: MouseEvent) => {
      startX = e.clientX; startY = e.clientY; dragging = true;
    });

    overlay.addEventListener('mousemove', (e: MouseEvent) => {
      if (!dragging) return;
      drawRect(startX, startY, e.clientX - startX, e.clientY - startY);
    });

    // onEsc must be declared before mouseup so we can remove it from both paths
    function onEsc(e: KeyboardEvent): void {
      if (e.key === 'Escape') {
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
        document.removeEventListener('keydown', onEsc);
        sendResponse({ ok: false });
      }
    }

    overlay.addEventListener('mouseup', (e: MouseEvent) => {
      dragging = false;
      const x: number = Math.min(startX, e.clientX);
      const y: number = Math.min(startY, e.clientY);
      const w: number = Math.abs(e.clientX - startX);
      const h: number = Math.abs(e.clientY - startY);
      // Clean up Esc listener on mouseup path too, or it leaks and may double-fire
      document.removeEventListener('keydown', onEsc);
      if (document.body.contains(overlay)) document.body.removeChild(overlay);
      if (w < 5 || h < 5) { sendResponse({ ok: false }); return; }
      sendResponse({ ok: true, x, y, w, h, dpr });
    });

    document.addEventListener('keydown', onEsc);

    // Safety cleanup: remove listener if overlay is removed by page navigation or other means
    const ocrCleanupObserver = new MutationObserver(() => {
      if (!document.body.contains(overlay)) {
        document.removeEventListener('keydown', onEsc);
        ocrCleanupObserver.disconnect();
      }
    });
    ocrCleanupObserver.observe(document.body, { childList: true });
  }

  // ── Smart Content Extraction ─────────────────────────────────────────

  const MAX_TEXT_BYTES = 50000; // ~50KB limit

  /** Selectors for elements likely to be the main content area */
  const CONTENT_SELECTORS = [
    'article',
    'main',
    '[role="main"]',
    '.post-content',
    '.article-body',
    '.entry-content',
    '.post-body',
    '.article-content',
    '.story-body',
    '.content-body',
    '#article-body',
    '#content',
    '.markdown-body', // GitHub
    '.wiki-content',
  ];

  /** Selectors for junk elements to strip from the clone */
  const JUNK_SELECTORS = [
    'nav', 'header', 'footer', 'aside',
    'script', 'style', 'noscript', 'iframe',
    '.sidebar', '.menu', '.nav', '.navigation',
    '.cookie', '.cookie-banner', '.cookie-consent',
    '.popup', '.modal', '.overlay',
    '.ad', '.ads', '.advert', '.advertisement',
    '.social-share', '.share-buttons', '.social-buttons',
    '.comments', '.comment-section', '#comments',
    '.related-posts', '.recommended',
    '.newsletter', '.subscribe',
    '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]',
    '[role="complementary"]', '[aria-hidden="true"]',
    '.sr-only', '.visually-hidden',
  ];

  /**
   * Finds the best content root element on the page.
   * Tries semantic selectors first, falls back to document.body.
   */
  function findContentRoot(): Element {
    for (const sel of CONTENT_SELECTORS) {
      const el = document.querySelector(sel);
      if (el && el.textContent && el.textContent.trim().length > 200) {
        return el;
      }
    }
    return document.body;
  }

  /**
   * Clones the content root and strips junk elements from the clone.
   * Never modifies the live DOM.
   */
  function cleanClone(root: Element): Element {
    const clone = root.cloneNode(true) as Element;
    for (const sel of JUNK_SELECTORS) {
      clone.querySelectorAll(sel).forEach((el) => el.remove());
    }
    return clone;
  }

  /**
   * Walks a cleaned DOM tree and produces structured plain text:
   * - Headings get #/##/### prefixes
   * - List items get - prefix
   * - Links become "text (url)"
   * - Block elements get paragraph breaks
   */
  function domToStructuredText(root: Element): string {
    const parts: string[] = [];
    const BLOCK_TAGS = new Set([
      'P', 'DIV', 'SECTION', 'ARTICLE', 'BLOCKQUOTE', 'PRE',
      'H1', 'H2', 'H3', 'H4', 'H5', 'H6',
      'UL', 'OL', 'LI', 'TABLE', 'TR', 'DT', 'DD',
      'FIGURE', 'FIGCAPTION', 'HR', 'BR',
    ]);

    function walk(node: Node): void {
      if (node.nodeType === Node.TEXT_NODE) {
        const text = (node.textContent || '').replace(/\s+/g, ' ');
        if (text.trim()) parts.push(text);
        return;
      }

      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const el = node as Element;
      const tag = el.tagName;

      // Skip hidden elements
      if (el.hasAttribute('hidden') || (el as HTMLElement).style?.display === 'none') return;

      // Headings
      if (/^H[1-6]$/.test(tag)) {
        const level = parseInt(tag[1]);
        const prefix = '#'.repeat(Math.min(level, 3)) + ' ';
        const headingText = (el.textContent || '').trim();
        if (headingText) {
          parts.push('\n\n' + prefix + headingText + '\n');
        }
        return; // Don't walk children again
      }

      // List items
      if (tag === 'LI') {
        const text = (el.textContent || '').trim();
        if (text) {
          parts.push('\n- ' + text);
        }
        return;
      }

      // Links — inline, preserve URL
      if (tag === 'A') {
        const href = (el as HTMLAnchorElement).href;
        const text = (el.textContent || '').trim();
        if (text && href && !href.startsWith('javascript:')) {
          parts.push(text + ' (' + href + ')');
        } else if (text) {
          parts.push(text);
        }
        return;
      }

      // HR → separator
      if (tag === 'HR') {
        parts.push('\n\n---\n\n');
        return;
      }

      // BR → newline
      if (tag === 'BR') {
        parts.push('\n');
        return;
      }

      // Pre/code → preserve formatting
      if (tag === 'PRE') {
        const text = (el.textContent || '').trim();
        if (text) parts.push('\n\n```\n' + text + '\n```\n\n');
        return;
      }

      // Block element → paragraph break before
      const isBlock = BLOCK_TAGS.has(tag);
      if (isBlock) parts.push('\n\n');

      // Walk children
      for (const child of el.childNodes) {
        walk(child);
      }

      // Block element → paragraph break after
      if (isBlock) parts.push('\n');
    }

    walk(root);

    // Clean up excessive whitespace
    return parts
      .join('')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/[ \t]+/g, ' ')
      .trim();
  }

  /**
   * Main extraction: finds content, cleans it, converts to structured text.
   * Falls back to raw innerText on any error.
   */
  function extractMainContent(): ExtractPageResponse {
    try {
      const url = window.location.href;
      const title = document.title;

      // ── PDF detection ──
      if (
        url.match(/\.pdf($|\?|#)/i) ||
        document.contentType === 'application/pdf'
      ) {
        return {
          text: document.body?.innerText?.slice(0, MAX_TEXT_BYTES) || '[PDF document]',
          title,
          url,
          wordCount: 0,
          isPdf: true,
        };
      }

      // ── YouTube detection ──
      if (url.includes('youtube.com/watch') || url.includes('youtu.be/')) {
        return extractYouTubeContent();
      }

      // ── General page extraction ──
      const root = findContentRoot();
      const cleaned = cleanClone(root);
      let text = domToStructuredText(cleaned);

      // If smart extraction yielded very little, fall back to body innerText
      if (text.length < 100) {
        text = document.body?.innerText || '';
      }

      // Truncate
      if (text.length > MAX_TEXT_BYTES) {
        text = text.slice(0, MAX_TEXT_BYTES) + '\n\n[...truncated]';
      }

      const wordCount = text.split(/\s+/).filter(Boolean).length;

      return { text, title, url, wordCount };
    } catch (_e) {
      // Fallback: raw innerText
      const fallbackText = (document.body?.innerText || '').slice(0, MAX_TEXT_BYTES);
      return {
        text: fallbackText,
        title: document.title,
        url: window.location.href,
        wordCount: fallbackText.split(/\s+/).filter(Boolean).length,
      };
    }
  }

  /**
   * YouTube-specific extraction:
   * - Video title
   * - Transcript (if the transcript panel is open / segments exist in DOM)
   * - Falls back to description text
   */
  function extractYouTubeContent(): ExtractPageResponse {
    const url = window.location.href;

    // Video title — try specific selector first, then document.title
    const titleEl = document.querySelector<HTMLElement>(
      'h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1'
    );
    const videoTitle = titleEl?.textContent?.trim() || document.title.replace(/ - YouTube$/, '').trim();

    // Try to extract transcript segments from the page
    let transcript = '';
    const transcriptSegments = document.querySelectorAll<HTMLElement>(
      'ytd-transcript-segment-renderer .segment-text, ' +
      'yt-formatted-string.ytd-transcript-segment-renderer, ' +
      '#segments-container ytd-transcript-segment-renderer'
    );
    if (transcriptSegments.length > 0) {
      const lines: string[] = [];
      transcriptSegments.forEach((seg) => {
        const text = seg.textContent?.trim();
        if (text) lines.push(text);
      });
      transcript = lines.join(' ');
    }

    // Description text
    let description = '';
    const descEl = document.querySelector<HTMLElement>(
      'ytd-text-inline-expander #plain-snippet-text, ' +
      '#description-inline-expander, ' +
      'ytd-expander .content, ' +
      '#description .content'
    );
    if (descEl) {
      description = descEl.textContent?.trim() || '';
    }

    // Comments (first few)
    const commentEls = document.querySelectorAll<HTMLElement>(
      'ytd-comment-thread-renderer #content-text'
    );
    let comments = '';
    if (commentEls.length > 0) {
      const commentLines: string[] = [];
      commentEls.forEach((el, i) => {
        if (i >= 10) return; // max 10 comments
        const text = el.textContent?.trim();
        if (text) commentLines.push('- ' + text);
      });
      if (commentLines.length > 0) {
        comments = '\n\n## Top Comments\n' + commentLines.join('\n');
      }
    }

    // Build structured text
    let text = `# ${videoTitle}\n\n`;
    if (transcript) {
      text += `## Transcript\n${transcript}\n\n`;
    }
    if (description) {
      text += `## Description\n${description}\n\n`;
    }
    text += comments;

    if (text.length > MAX_TEXT_BYTES) {
      text = text.slice(0, MAX_TEXT_BYTES) + '\n\n[...truncated]';
    }

    const wordCount = text.split(/\s+/).filter(Boolean).length;

    return {
      text,
      title: videoTitle,
      url,
      wordCount,
      isYouTube: true,
      videoTitle,
      transcript: transcript || undefined,
    };
  }

  // ── Message Listener ──────────────────────────────────────────────────

  ext.runtime.onMessage.addListener(
    (
      msg: InboundMessage,
      _sender: chrome.runtime.MessageSender,
      sendResponse: (response: ExtractPageResponse | GetDomResponse | ExecActionResult | OcrOverlayResult) => void
    ): boolean | undefined => {
      if (msg.type === 'EXTRACT_PAGE') {
        sendResponse(extractMainContent());
        return false;
      }

      if (msg.type === 'GET_DOM') {
        sendResponse({ ok: true, dom: serializeDOM(), url: location.href, title: document.title });
        return false;
      }

      if (msg.type === 'EXEC_ACTION') {
        sendResponse(execAction(msg.action));
        return false;
      }

      if (msg.type === 'SHOW_OCR_OVERLAY') {
        showOcrOverlay(msg.dataUrl, sendResponse);
        return true; // async
      }

      return undefined;
    }
  );

})();
