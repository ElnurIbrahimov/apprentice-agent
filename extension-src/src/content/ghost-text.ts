/**
 * Ghost-text inline assist — Aura suggests a continuation for any textarea
 * or contenteditable after 800 ms of typing pause.
 *
 * v1 UX: shows a labelled suggestion **chip** positioned directly below the
 * input field rather than an inline overlay — true caret-relative inline
 * ghost text requires a mirror-div trick that's fragile across sites and
 * often misaligns. The chip is honest: it says "Aura suggests" and the user
 * can Tab to accept, or ignore it.
 *
 * Denylisted on password fields, autocomplete=off inputs, banking / password
 * manager domains, and incognito tabs.
 */

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

const DEBOUNCE_MS = 800;
const MIN_TEXT_LEN = 3;
const MAX_TEXT_LEN = 2000;
const MAX_SUGGESTION_LEN = 200;
const CHIP_ID = 'aura-ghost-chip';

interface GhostState {
  el: HTMLTextAreaElement | HTMLElement;
  text: string;
  suggestion: string;
  debounceTimer: number | null;
  abortCtrl: AbortController | null;
}

let state: GhostState | null = null;
let chipNode: HTMLDivElement | null = null;
let enabled = true;

const SENSITIVE_HOSTS = [
  'accounts.google.com',
  'login.microsoftonline.com',
  'okta.com',
  '1password.com',
  'lastpass.com',
  'bitwarden.com',
  'dashlane.com',
];

function isDenylisted(): boolean {
  const host = location.hostname;
  if (SENSITIVE_HOSTS.some((d) => host === d || host.endsWith('.' + d))) return true;
  if (/\bbank\b|\bpayment\b|\bcheckout\b/i.test(host)) return true;
  return false;
}

function ensureChipNode(): HTMLDivElement {
  if (chipNode && document.body.contains(chipNode)) return chipNode;
  const node = document.createElement('div');
  node.id = CHIP_ID;
  node.setAttribute('role', 'tooltip');
  node.setAttribute('aria-live', 'polite');
  node.style.cssText = [
    'position:fixed',
    'z-index:2147483646',
    'pointer-events:none',
    'max-width:480px',
    'padding:6px 10px',
    'border-radius:8px',
    'background:rgba(20,20,28,0.92)',
    'color:rgba(220,220,235,0.92)',
    'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
    'font-size:12px',
    'line-height:1.4',
    'box-shadow:0 6px 24px rgba(0,0,0,0.35), 0 0 0 1px rgba(124,58,237,0.35)',
    'opacity:0',
    'transform:translateY(-4px)',
    'transition:opacity 0.12s ease, transform 0.12s ease',
    'display:none',
  ].join(';');
  document.body.appendChild(node);
  chipNode = node;
  return node;
}

function clearChip(): void {
  if (chipNode) {
    chipNode.style.opacity = '0';
    chipNode.style.transform = 'translateY(-4px)';
    chipNode.style.display = 'none';
    chipNode.textContent = '';
  }
  if (state) state.suggestion = '';
}

function showChipFor(el: HTMLElement, suggestion: string): void {
  const node = ensureChipNode();
  node.textContent = '';
  const label = document.createElement('span');
  label.textContent = 'Aura · Tab to accept';
  label.style.cssText = 'color:rgba(167,139,250,0.85);margin-right:8px;font-size:10.5px;text-transform:uppercase;letter-spacing:0.05em';
  node.appendChild(label);
  const text = document.createElement('span');
  text.textContent = suggestion;
  node.appendChild(text);

  const rect = el.getBoundingClientRect();
  // Position just below the field, clamped to viewport width.
  const maxLeft = Math.max(8, window.innerWidth - 500);
  const left = Math.min(Math.max(8, rect.left), maxLeft);
  const top = Math.min(rect.bottom + 6, window.innerHeight - 40);
  node.style.left = `${left}px`;
  node.style.top = `${top}px`;
  node.style.display = 'block';
  // Force reflow before transition
  void node.offsetHeight;
  node.style.opacity = '1';
  node.style.transform = 'translateY(0)';
}

function getInputText(el: HTMLElement): string {
  if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
    return el.value || '';
  }
  return el.textContent || '';
}

function applySuggestion(el: HTMLElement, suggestion: string): void {
  if (el instanceof HTMLTextAreaElement || el instanceof HTMLInputElement) {
    const start = el.selectionStart ?? el.value.length;
    el.value = el.value.slice(0, start) + suggestion + el.value.slice(start);
    const newPos = start + suggestion.length;
    try { el.setSelectionRange(newPos, newPos); } catch { /* noop for input types that don't support it */ }
    el.dispatchEvent(new Event('input', { bubbles: true }));
  } else if (el.isContentEditable) {
    const sel = window.getSelection();
    if (sel && sel.rangeCount > 0) {
      const range = sel.getRangeAt(0);
      const node = document.createTextNode(suggestion);
      range.insertNode(node);
      range.setStartAfter(node);
      range.setEndAfter(node);
      sel.removeAllRanges();
      sel.addRange(range);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }
  clearChip();
  state = null;
}

async function requestCompletion(el: HTMLElement, text: string): Promise<void> {
  if (!state) return;
  if (state.abortCtrl) state.abortCtrl.abort();
  const ctrl = new AbortController();
  state.abortCtrl = ctrl;

  try {
    const resp = await ext.runtime.sendMessage({
      type: 'GHOST_COMPLETE',
      text: text.slice(-MAX_TEXT_LEN),
      url: location.href,
      title: document.title,
    });
    if (!resp || typeof resp.continuation !== 'string') return;
    const s = String(resp.continuation || '').trim();
    if (!s || s.length > MAX_SUGGESTION_LEN) return;
    if (state && state.el === el && state.text === text) {
      state.suggestion = s;
      showChipFor(el, s);
    }
  } catch {
    /* silent — opportunistic */
  }
}

function scheduleCompletion(el: HTMLElement): void {
  if (!state || state.el !== el) return;
  if (state.debounceTimer) window.clearTimeout(state.debounceTimer);
  state.debounceTimer = window.setTimeout(() => {
    if (!state || state.el !== el) return;
    const text = getInputText(el);
    state.text = text;
    if (text.length < MIN_TEXT_LEN) return;
    requestCompletion(el, text);
  }, DEBOUNCE_MS);
}

function isEligibleTarget(el: EventTarget | null): el is HTMLElement {
  if (!el || !(el instanceof HTMLElement)) return false;
  if (el instanceof HTMLInputElement) {
    if (['password', 'email', 'tel', 'number', 'date', 'url'].includes(el.type)) return false;
    if (el.getAttribute('autocomplete') === 'off') return false;
    return true;
  }
  if (el instanceof HTMLTextAreaElement) {
    if (el.getAttribute('autocomplete') === 'off') return false;
    return true;
  }
  return el.isContentEditable;
}

function onInput(e: Event): void {
  if (!enabled || isDenylisted()) return;
  if (!isEligibleTarget(e.target)) return;
  const target = e.target as HTMLElement;
  state = {
    el: target,
    text: getInputText(target),
    suggestion: '',
    debounceTimer: null,
    abortCtrl: null,
  };
  clearChip();
  scheduleCompletion(target);
}

function onKeyDown(e: KeyboardEvent): void {
  if (!state) return;
  if (e.key === 'Tab' && !e.shiftKey && state.suggestion) {
    e.preventDefault();
    applySuggestion(state.el, state.suggestion);
    return;
  }
  // Any non-Tab key dismisses the current chip (but the new input event
  // will re-trigger the debounce timer).
  if (state.suggestion) clearChip();
}

function onScrollOrResize(): void {
  clearChip();
}

export function initGhostText(): void {
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return;

  // Respect ghostTextMode: only run chip mode when explicitly selected (or as
  // legacy default when the new inline renderer has disabled itself).
  // The new ghost-text-inline.ts sets the default to 'inline' and owns
  // rendering when that's set — we silently stand down in that case.
  try {
    ext.storage?.local?.get(['ghostTextMode'], (d: any) => {
      const m = d?.ghostTextMode;
      if (m === 'inline' || m === 'off') {
        enabled = false;
      }
    });
    ext.storage?.onChanged?.addListener((changes: any, area: string) => {
      if (area === 'local' && changes.ghostTextMode) {
        const v = changes.ghostTextMode.newValue;
        enabled = v === 'chip';
        if (!enabled) clearChip();
      }
    });
  } catch { /* silent */ }

  document.addEventListener('input', onInput, { capture: true, passive: true });
  document.addEventListener('keydown', onKeyDown, { capture: true });
  window.addEventListener('scroll', onScrollOrResize, { capture: true, passive: true });
  window.addEventListener('resize', onScrollOrResize, { passive: true });
  window.addEventListener('blur', clearChip);
}

// Host page can toggle at runtime.
window.addEventListener('aura-ghost-disable', () => { enabled = false; clearChip(); });
window.addEventListener('aura-ghost-enable', () => { enabled = true; });
