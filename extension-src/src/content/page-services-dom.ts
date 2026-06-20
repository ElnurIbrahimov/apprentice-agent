/**
 * DOM Serializer — extracted from page-services.ts
 * Provides DOM element serialization, selector generation, and action execution.
 */

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface SerializedElement {
  index: number;
  type: string;
  text: string;
  selector: string;
  href: string;
}

export interface ExecActionParams {
  action: 'click' | 'type' | 'scroll' | 'selectOption';
  selector?: string;
  text?: string;
  url?: string;
  amount?: number;
  value?: string;
}

export interface ExecActionResult {
  ok: boolean;
  error?: string;
}

// ── DOM Serializer ───────────────────────────────────────────────────────────

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

export function serializeDOM(): SerializedElement[] {
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

// Allowlist: only these action verbs can be dispatched to execAction. Any
// other value (including undefined) is refused before we reach the DOM.
const ALLOWED_EXEC_ACTIONS = new Set([
  'scroll', 'click', 'type', 'selectOption',
]);

export function execAction(action: ExecActionParams): ExecActionResult {
  const verb = action && typeof action.action === 'string' ? action.action : '';
  if (!ALLOWED_EXEC_ACTIONS.has(verb)) {
    return { ok: false, error: 'Unknown action: ' + verb };
  }
  if (verb === 'scroll') {
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
  if (verb === 'click') { (el as HTMLElement).click(); return { ok: true }; }
  if (verb === 'type') {
    (el as HTMLElement).focus();
    (el as HTMLInputElement).value = action.text || '';
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
    return { ok: true };
  }
  if (verb === 'selectOption') {
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
  // ALLOWED_EXEC_ACTIONS covers every branch above; unreachable unless the
  // set and the branches drift out of sync.
  return { ok: false, error: 'Unhandled action: ' + verb };
}