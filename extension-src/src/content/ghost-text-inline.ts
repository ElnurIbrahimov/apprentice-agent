/**
 * Ghost-text inline — caret-relative suggestion rendering for <textarea>,
 * <input>, and contenteditable fields.
 *
 * Strategy:
 *  - textarea: mirror-div technique — a hidden div replicating the textarea's
 *    typography + content-up-to-caret with a zero-width marker, measured and
 *    overlaid with a ghost span in the DOM layer above the textarea.
 *  - input (single-line): simple absolute span positioned via canvas-based
 *    text width measurement.
 *  - contenteditable: Selection.getRangeAt(0).getBoundingClientRect() — native
 *    browser-provided caret position, zero fragility.
 *
 * Falls back silently when positioning fails (iframes, shadow DOM sinks, etc.)
 * — the old chip-style ghost-text.ts remains installed and can pick up where
 * this bails out.
 *
 * Controlled by chrome.storage.local.ghostTextMode: 'inline' | 'chip' | 'off'.
 * Default is 'inline' when this module is active.
 */

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

const DEBOUNCE_MS = 600;
const MIN_TEXT_LEN = 8;
const MAX_TEXT_LEN = 2000;
const MIN_WORDS = 3;
const OVERLAY_ID = 'aura-ghost-inline';
const MIRROR_ID = 'aura-ghost-mirror';

const SENSITIVE_HOSTS = [
  'accounts.google.com',
  'login.microsoftonline.com',
  'okta.com',
  '1password.com',
  'lastpass.com',
  'bitwarden.com',
  'dashlane.com',
  // Webmail — subject lines + recipient names + draft content are personally
  // identifying data we should never auto-send to the completion endpoint.
  'mail.google.com',
  'outlook.live.com',
  'outlook.office.com',
  'mail.yahoo.com',
  'mail.proton.me',
];

function isDenylisted(): boolean {
  const host = location.hostname;
  if (SENSITIVE_HOSTS.some((d) => host === d || host.endsWith('.' + d))) return true;
  if (/\bbank\b|\bpayment\b|\bcheckout\b/i.test(host)) return true;
  return false;
}

interface GhostState {
  el: HTMLElement;
  text: string;
  suggestion: string;
  debounceTimer: number | null;
  abortCtrl: AbortController | null;
}

let state: GhostState | null = null;
let overlay: HTMLDivElement | null = null;
let mirror: HTMLDivElement | null = null;
let enabled = true;
let mode: 'inline' | 'chip' | 'off' = 'inline';

// Cached canvas+ctx for text-width measurement in single-line <input>
// placement. Creating a new OffscreenCanvas on every keystroke is wasteful.
let _measureCanvas: HTMLCanvasElement | null = null;
let _measureCtx: CanvasRenderingContext2D | null = null;
function getMeasureCtx(): CanvasRenderingContext2D | null {
  if (_measureCtx) return _measureCtx;
  try {
    _measureCanvas = document.createElement('canvas');
    _measureCtx = _measureCanvas.getContext('2d');
    return _measureCtx;
  } catch {
    return null;
  }
}

function ensureOverlay(): HTMLDivElement {
  if (overlay && document.body.contains(overlay)) return overlay;
  const div = document.createElement('div');
  div.id = OVERLAY_ID;
  div.style.cssText = [
    'position:fixed',
    'z-index:2147483646',
    'pointer-events:none',
    'color:rgba(167,139,250,0.55)',
    'font-family:inherit',
    'white-space:pre-wrap',
    'overflow:hidden',
    'display:none',
  ].join(';');
  document.body.appendChild(div);
  overlay = div;
  return div;
}

function ensureMirror(): HTMLDivElement {
  if (mirror && document.body.contains(mirror)) return mirror;
  const div = document.createElement('div');
  div.id = MIRROR_ID;
  div.style.cssText = [
    'position:absolute',
    'top:0',
    'left:-9999px',
    'visibility:hidden',
    'white-space:pre-wrap',
    'word-wrap:break-word',
    'overflow-wrap:break-word',
    'pointer-events:none',
  ].join(';');
  document.body.appendChild(div);
  mirror = div;
  return div;
}

function copyTypography(src: HTMLElement, dst: HTMLElement): void {
  const cs = window.getComputedStyle(src);
  const props = [
    'boxSizing', 'width', 'height', 'overflow',
    'fontFamily', 'fontSize', 'fontWeight', 'fontStyle', 'fontVariant',
    'letterSpacing', 'textTransform', 'textIndent',
    'lineHeight', 'paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
    'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
    'borderTopStyle', 'borderRightStyle', 'borderBottomStyle', 'borderLeftStyle',
  ] as const;
  for (const p of props) {
    (dst.style as any)[p] = (cs as any)[p];
  }
}

function placeOverlayAtTextareaCaret(ta: HTMLTextAreaElement, suggestion: string): boolean {
  try {
    const m = ensureMirror();
    const rect = ta.getBoundingClientRect();
    copyTypography(ta, m);
    m.style.width = `${rect.width}px`;
    m.style.height = 'auto';

    // Replicate text up to caret, then insert a zero-width marker
    const caret = ta.selectionStart ?? ta.value.length;
    const before = ta.value.substring(0, caret).replace(/\n$/, '\n ');
    m.textContent = before;
    const markerSpan = document.createElement('span');
    markerSpan.textContent = '\u200b';
    m.appendChild(markerSpan);

    const markerRect = markerSpan.getBoundingClientRect();
    const mirrorRect = m.getBoundingClientRect();

    // Map marker offset inside mirror to position inside the textarea
    const localX = markerRect.left - mirrorRect.left;
    const localY = markerRect.top - mirrorRect.top;

    const absX = rect.left + localX - ta.scrollLeft;
    const absY = rect.top + localY - ta.scrollTop;

    // Guard: caret must be within the textarea's visible area
    if (absX < rect.left || absX > rect.right || absY < rect.top || absY > rect.bottom) {
      return false;
    }

    const ov = ensureOverlay();
    const cs = window.getComputedStyle(ta);
    ov.style.font = cs.font;
    ov.style.fontSize = cs.fontSize;
    ov.style.fontFamily = cs.fontFamily;
    ov.style.lineHeight = cs.lineHeight;
    ov.style.letterSpacing = cs.letterSpacing;
    ov.style.left = `${absX}px`;
    ov.style.top = `${absY}px`;
    ov.style.maxWidth = `${rect.right - absX - 4}px`;
    ov.textContent = suggestion;
    ov.style.display = '';
    return true;
  } catch {
    return false;
  }
}

function placeOverlayAtContentEditableCaret(suggestion: string): boolean {
  try {
    const sel = window.getSelection();
    if (!sel || sel.rangeCount === 0) return false;
    const range = sel.getRangeAt(0).cloneRange();
    range.collapse(false);
    // Insert a marker, measure, remove
    const marker = document.createElement('span');
    marker.textContent = '\u200b';
    range.insertNode(marker);
    const mRect = marker.getBoundingClientRect();
    const parent = marker.parentNode;
    parent?.removeChild(marker);

    if (mRect.width === 0 && mRect.height === 0) return false;

    const ov = ensureOverlay();
    // Inherit typography from the editor's actual element
    const anchor = sel.anchorNode?.parentElement;
    if (anchor) {
      const cs = window.getComputedStyle(anchor);
      ov.style.font = cs.font;
      ov.style.fontSize = cs.fontSize;
      ov.style.fontFamily = cs.fontFamily;
      ov.style.lineHeight = cs.lineHeight;
      ov.style.letterSpacing = cs.letterSpacing;
    }
    ov.style.left = `${mRect.left}px`;
    ov.style.top = `${mRect.top}px`;
    ov.style.maxWidth = `${Math.max(240, window.innerWidth - mRect.left - 20)}px`;
    ov.textContent = suggestion;
    ov.style.display = '';
    return true;
  } catch {
    return false;
  }
}

function placeOverlayAtInputCaret(input: HTMLInputElement, suggestion: string): boolean {
  try {
    const rect = input.getBoundingClientRect();
    const cs = window.getComputedStyle(input);
    const ctx = getMeasureCtx();
    if (!ctx) return false;
    ctx.font = cs.font;
    const caret = input.selectionStart ?? input.value.length;
    const textWidth = ctx.measureText(input.value.substring(0, caret)).width;
    const paddingLeft = parseFloat(cs.paddingLeft) || 0;
    const borderLeft = parseFloat(cs.borderLeftWidth) || 0;
    const absX = rect.left + borderLeft + paddingLeft + textWidth - input.scrollLeft;
    const absY = rect.top + (rect.height - parseFloat(cs.fontSize)) / 2;

    if (absX > rect.right - 8) return false;

    const ov = ensureOverlay();
    ov.style.font = cs.font;
    ov.style.fontSize = cs.fontSize;
    ov.style.fontFamily = cs.fontFamily;
    ov.style.letterSpacing = cs.letterSpacing;
    ov.style.left = `${absX}px`;
    ov.style.top = `${absY}px`;
    ov.style.maxWidth = `${rect.right - absX - 6}px`;
    ov.textContent = suggestion;
    ov.style.display = '';
    return true;
  } catch {
    return false;
  }
}

function placeOverlay(el: HTMLElement, suggestion: string): boolean {
  if (el.tagName === 'TEXTAREA') {
    return placeOverlayAtTextareaCaret(el as HTMLTextAreaElement, suggestion);
  }
  if (el.tagName === 'INPUT') {
    return placeOverlayAtInputCaret(el as HTMLInputElement, suggestion);
  }
  if (el.isContentEditable) {
    return placeOverlayAtContentEditableCaret(suggestion);
  }
  return false;
}

function hideOverlay(): void {
  if (overlay) {
    overlay.style.display = 'none';
    overlay.textContent = '';
  }
  if (state) state.suggestion = '';
}

function getElementText(el: HTMLElement): string {
  if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
    return (el as HTMLTextAreaElement | HTMLInputElement).value || '';
  }
  if (el.isContentEditable) {
    return el.textContent || '';
  }
  return '';
}

function countWords(s: string): number {
  return (s.trim().match(/\S+/g) || []).length;
}

function isEligibleTarget(el: EventTarget | null): el is HTMLElement {
  if (!(el instanceof HTMLElement)) return false;
  if (el.tagName === 'INPUT') {
    const type = ((el as HTMLInputElement).type || '').toLowerCase();
    if (['password', 'email', 'tel', 'number', 'search', 'url', 'hidden'].includes(type)) return false;
    if (el.getAttribute('autocomplete') === 'off') return false;
  }
  if (el.tagName === 'TEXTAREA') return true;
  if (el.isContentEditable) return true;
  if (el.tagName === 'INPUT') return true;
  return false;
}

async function requestCompletion(text: string, signal: AbortSignal): Promise<string | null> {
  return new Promise((resolve) => {
    try {
      ext.runtime.sendMessage(
        {
          type: 'GHOST_COMPLETE',
          text,
          url: location.href,
          title: document.title,
        },
        (response: any) => {
          if (signal.aborted) {
            resolve(null);
            return;
          }
          const suggestion = response?.continuation?.trim?.() || '';
          resolve(suggestion || null);
        },
      );
    } catch {
      resolve(null);
    }
  });
}

function handleInput(e: Event): void {
  if (!enabled || mode !== 'inline') return;
  const el = e.target as HTMLElement;
  if (!isEligibleTarget(el)) return;

  const text = getElementText(el);
  if (text.length < MIN_TEXT_LEN || text.length > MAX_TEXT_LEN) {
    hideOverlay();
    return;
  }
  if (countWords(text) < MIN_WORDS) {
    hideOverlay();
    return;
  }

  // Replace any active state
  if (state?.debounceTimer) clearTimeout(state.debounceTimer);
  state?.abortCtrl?.abort();

  state = {
    el,
    text,
    suggestion: '',
    debounceTimer: null,
    abortCtrl: null,
  };

  state.debounceTimer = window.setTimeout(async () => {
    if (!state || state.el !== el) return;
    const ctrl = new AbortController();
    state.abortCtrl = ctrl;
    const suggestion = await requestCompletion(text, ctrl.signal);
    if (!suggestion || ctrl.signal.aborted) {
      hideOverlay();
      return;
    }
    if (!state || state.el !== el) return;
    state.suggestion = suggestion;
    const ok = placeOverlay(el, suggestion);
    if (!ok) hideOverlay();
  }, DEBOUNCE_MS);
}

function handleKeyDown(e: KeyboardEvent): void {
  if (!state || !state.suggestion) return;
  if (e.target !== state.el) return;

  if (e.key === 'Tab') {
    e.preventDefault();
    const el = state.el;
    const suggestion = state.suggestion;
    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
      const field = el as HTMLTextAreaElement | HTMLInputElement;
      const caret = field.selectionStart ?? field.value.length;
      const before = field.value.substring(0, caret);
      const after = field.value.substring(caret);
      field.value = before + suggestion + after;
      field.selectionStart = field.selectionEnd = caret + suggestion.length;
      field.dispatchEvent(new Event('input', { bubbles: true }));
    } else if (el.isContentEditable) {
      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0) {
        const range = sel.getRangeAt(0);
        range.deleteContents();
        range.insertNode(document.createTextNode(suggestion));
        range.collapse(false);
      }
    }
    hideOverlay();
    state = null;
    return;
  }
  if (e.key === 'Escape') {
    hideOverlay();
    state = null;
  }
}

function handleBlur(): void {
  hideOverlay();
}

function handleScrollOrResize(): void {
  if (state?.suggestion && state.el) {
    placeOverlay(state.el, state.suggestion);
  }
}

function reloadMode(): void {
  try {
    ext.storage?.local?.get(['ghostTextMode'], (d: any) => {
      const m = d?.ghostTextMode;
      if (m === 'chip' || m === 'off' || m === 'inline') {
        mode = m;
      }
    });
  } catch { /* silent */ }
}

export function initGhostTextInline(): void {
  if (isDenylisted()) {
    enabled = false;
    return;
  }
  reloadMode();

  // Listen for runtime mode changes
  try {
    ext.storage?.onChanged?.addListener((changes: any, area: string) => {
      if (area === 'local' && changes.ghostTextMode) {
        const v = changes.ghostTextMode.newValue;
        if (v === 'chip' || v === 'off' || v === 'inline') mode = v;
        if (mode !== 'inline') hideOverlay();
      }
    });
  } catch { /* silent */ }

  document.addEventListener('input', handleInput, true);
  document.addEventListener('keydown', handleKeyDown, true);
  document.addEventListener('blur', handleBlur, true);
  window.addEventListener('scroll', handleScrollOrResize, true);
  window.addEventListener('resize', handleScrollOrResize);
}
