/**
 * Page Services module — Utility/service functions
 * Migrated from content.ts:
 *   - Lines ~1763-1838: DOM Serializer
 *   - Lines ~1839-1936: OCR Overlay
 *   - Lines ~1937-2250: Smart Content Extraction
 *   - Lines ~2251-2640: Quick Actions on Input Fields
 *   - Lines ~2641-2706: YouTube & Netflix Subtitle Relay
 *   - Lines ~2707-2986: Full Page Translation
 *   - Lines ~4132-4504: Full Page Extraction
 *   - Lines ~4608-5297: Google SERP AI Answer Card
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

export interface OcrOverlayResult {
  ok: boolean;
  x?: number;
  y?: number;
  w?: number;
  h?: number;
  dpr?: number;
}

export interface ExtractPageResponse {
  text: string;
  url: string;
  title: string;
  wordCount: number;
  isPdf?: boolean;
  isYouTube?: boolean;
  videoTitle?: string;
  transcript?: string;
}

interface QuickActionDef {
  label: string;
  icon: string;
  action: string;
  language?: string;
}

interface QuickActionResponse {
  ok: boolean;
  result?: string;
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

export function execAction(action: ExecActionParams): ExecActionResult {
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

// ── OCR Overlay ───────────────────────────────────────────────────────────────

export function showOcrOverlay(
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

// ── Smart Content Extraction ──────────────────────────────────────────────────

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
 * Walks a cleaned DOM tree and produces structured plain text.
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
 * YouTube-specific extraction.
 */
function extractYouTubeContent(): ExtractPageResponse {
  const url = window.location.href;

  // Video title
  const titleEl = document.querySelector<HTMLElement>(
    'h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1'
  );
  const videoTitle = titleEl?.textContent?.trim() || document.title.replace(/ - YouTube$/, '').trim();

  // Transcript segments
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
      if (i >= 10) return;
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

/**
 * Main extraction: finds content, cleans it, converts to structured text.
 * Falls back to raw innerText on any error.
 */
export function extractMainContent(): ExtractPageResponse {
  try {
    const url = window.location.href;
    const title = document.title;

    // PDF detection
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

    // YouTube detection
    if (url.includes('youtube.com/watch') || url.includes('youtu.be/')) {
      return extractYouTubeContent();
    }

    // General page extraction
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

// ── Quick Actions on Input Fields ─────────────────────────────────────────────

const QUICK_ACTIONS_DEF: QuickActionDef[] = [
  { label: 'Improve', icon: '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>', action: 'improve' },
  { label: 'Expand', icon: '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>', action: 'expand' },
  { label: 'Shorten', icon: '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>', action: 'shorten' },
  { label: 'Fix grammar', icon: '<polyline points="20 6 9 17 4 12"/>', action: 'fix_grammar' },
  { label: 'Translate', icon: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: 'translate' },
];

// Input types to skip
const SKIP_INPUT_TYPES = new Set(['password', 'hidden', 'file', 'checkbox', 'radio', 'range', 'color', 'date', 'datetime-local', 'month', 'week', 'time', 'submit', 'reset', 'button', 'image']);
const MIN_INPUT_WIDTH = 200;

export function initQuickActionsOnInputs(ext: typeof chrome, safeSend: (msg: any, cb?: (r: any) => void) => void, showToast: (msg: string, duration?: number) => void): void {
  // Shadow DOM host for all quick-action UI
  const qaHost: HTMLDivElement = document.createElement('div');
  qaHost.id = 'aura-quick-action-host';
  Object.assign(qaHost.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    zIndex: '2147483646',
    pointerEvents: 'none',
  });
  document.documentElement.appendChild(qaHost);
  const qaShadow: ShadowRoot = qaHost.attachShadow({ mode: 'closed' });

  const qaStyle: HTMLStyleElement = document.createElement('style');
  qaStyle.textContent = `
    @keyframes qa-icon-in {
      from { opacity: 0; transform: scale(0.7); }
      to   { opacity: 1; transform: scale(1); }
    }
    @keyframes qa-menu-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes qa-spin {
      to { transform: rotate(360deg); }
    }

    .qa-trigger {
      position: fixed;
      width: 20px;
      height: 20px;
      border-radius: 5px;
      background: rgba(10, 8, 24, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      pointer-events: auto;
      animation: qa-icon-in 0.2s ease forwards;
      transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
      padding: 0;
      box-sizing: border-box;
    }
    .qa-trigger:hover {
      border-color: rgba(124, 58, 237, 0.6);
      background: rgba(124, 58, 237, 0.18);
      box-shadow: 0 0 10px rgba(124, 58, 237, 0.25);
    }
    .qa-trigger svg {
      width: 12px;
      height: 12px;
      color: rgba(160, 148, 210, 0.8);
    }

    .qa-menu {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(20px) saturate(1.5);
      -webkit-backdrop-filter: blur(20px) saturate(1.5);
      border: 1px solid rgba(124, 58, 237, 0.25);
      border-radius: 10px;
      padding: 4px;
      pointer-events: auto;
      animation: qa-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
      min-width: 140px;
    }
    .qa-menu-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 7px 12px;
      border-radius: 7px;
      background: transparent;
      border: none;
      color: rgba(226, 232, 240, 0.9);
      font-size: 12px;
      font-weight: 500;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      cursor: pointer;
      white-space: nowrap;
      width: 100%;
      text-align: left;
      transition: background 0.12s, color 0.12s;
      line-height: 1;
      box-sizing: border-box;
    }
    .qa-menu-item:hover {
      background: rgba(124, 58, 237, 0.25);
      color: #fff;
    }
    .qa-menu-item:active {
      background: rgba(124, 58, 237, 0.4);
    }
    .qa-menu-item svg {
      width: 14px;
      height: 14px;
      flex-shrink: 0;
      color: rgba(160, 148, 210, 0.7);
    }
    .qa-menu-item:hover svg {
      color: rgba(200, 180, 255, 1);
    }
    .qa-menu-item.loading {
      opacity: 0.6;
      pointer-events: none;
    }
    .qa-menu-item .qa-spinner {
      width: 14px;
      height: 14px;
      border: 2px solid rgba(124, 58, 237, 0.3);
      border-top-color: rgba(160, 148, 210, 0.9);
      border-radius: 50%;
      animation: qa-spin 0.6s linear infinite;
      flex-shrink: 0;
    }

    .qa-translate-sub {
      padding: 2px 4px 4px 4px;
    }
    .qa-translate-sub .qa-menu-item {
      font-size: 11.5px;
      padding: 5px 10px 5px 22px;
    }
  `;
  qaShadow.appendChild(qaStyle);

  // Container inside the shadow for dynamic elements
  const qaContainer: HTMLDivElement = document.createElement('div');
  qaShadow.appendChild(qaContainer);

  // Track currently attached input and UI elements
  let _qaActiveInput: HTMLInputElement | HTMLTextAreaElement | null = null;
  let _qaTriggerEl: HTMLDivElement | null = null;
  let _qaMenuEl: HTMLDivElement | null = null;
  let _qaTranslateSub: HTMLDivElement | null = null;

  function isEligibleInput(el: Element): el is HTMLInputElement | HTMLTextAreaElement {
    if (el.tagName === 'TEXTAREA') return true;
    if (el.tagName === 'INPUT') {
      const inputEl = el as HTMLInputElement;
      const inputType = (inputEl.type || 'text').toLowerCase();
      if (SKIP_INPUT_TYPES.has(inputType)) return false;
      return true;
    }
    // contenteditable
    if ((el as HTMLElement).isContentEditable && el.getAttribute('contenteditable') === 'true') return true;
    return false;
  }

  function removeQaMenu(): void {
    if (_qaMenuEl) { _qaMenuEl.remove(); _qaMenuEl = null; }
    if (_qaTranslateSub) { _qaTranslateSub.remove(); _qaTranslateSub = null; }
  }

  function removeQaTrigger(): void {
    if (_qaTriggerEl) { _qaTriggerEl.remove(); _qaTriggerEl = null; }
    removeQaMenu();
    _qaActiveInput = null;
  }

  function positionTrigger(field: HTMLElement): void {
    const rect = field.getBoundingClientRect();
    if (rect.width < MIN_INPUT_WIDTH) { removeQaTrigger(); return; }

    // Hide trigger when the input is scrolled offscreen
    const offscreen = rect.bottom < 0 || rect.top > window.innerHeight ||
                      rect.right < 0 || rect.left > window.innerWidth;

    if (!_qaTriggerEl) {
      _qaTriggerEl = document.createElement('div');
      _qaTriggerEl.className = 'qa-trigger';
      _qaTriggerEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;
      _qaTriggerEl.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (_qaMenuEl) { removeQaMenu(); return; }
        showQaMenu();
      });
      qaContainer.appendChild(_qaTriggerEl);
    }

    // Toggle visibility based on whether the field is in the viewport
    _qaTriggerEl.style.display = offscreen ? 'none' : '';

    if (offscreen) {
      removeQaMenu(); // also hide the menu if open
      return;
    }

    // Position at the right edge of the field, vertically centered
    const trigSize = 20;
    const pad = 6;
    _qaTriggerEl.style.top = `${Math.round(rect.top + (rect.height - trigSize) / 2)}px`;
    _qaTriggerEl.style.left = `${Math.round(rect.right - trigSize - pad)}px`;
  }

  function getInputValue(el: HTMLInputElement | HTMLTextAreaElement): string {
    if ((el as HTMLElement).isContentEditable) {
      return (el as HTMLElement).innerText || '';
    }
    return el.value || '';
  }

  function setInputValue(el: HTMLInputElement | HTMLTextAreaElement, value: string): void {
    if ((el as HTMLElement).isContentEditable) {
      (el as HTMLElement).innerText = value;
    } else {
      el.value = value;
    }
    // Fire events so frameworks pick up the change
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function executeQuickAction(action: string, language?: string): void {
    if (!_qaActiveInput) return;
    const text = getInputValue(_qaActiveInput);
    if (!text.trim()) { removeQaMenu(); return; }

    // Show loading state on all menu items
    if (_qaMenuEl) {
      _qaMenuEl.querySelectorAll('.qa-menu-item').forEach((item) => {
        item.classList.add('loading');
      });
    }

    const targetField = _qaActiveInput;

    safeSend(
      { type: 'QUICK_ACTION', action, text, language },
      (response: QuickActionResponse) => {
        if (response && response.ok && response.result) {
          setInputValue(targetField, response.result);
          showToast('Text updated by AURA');
        } else {
          showToast(response?.error || 'Quick action failed', 3000);
        }
        removeQaMenu();
      }
    );
  }

  function toggleTranslateSub(anchor: HTMLButtonElement): void {
    if (_qaTranslateSub) { _qaTranslateSub.remove(); _qaTranslateSub = null; return; }

    const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Russian', 'Japanese', 'Arabic', 'Portuguese', 'Azerbaijani'];

    _qaTranslateSub = document.createElement('div');
    _qaTranslateSub.className = 'qa-translate-sub';

    LANGUAGES.forEach((lang) => {
      const item = document.createElement('button');
      item.className = 'qa-menu-item';
      item.textContent = lang;
      item.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        executeQuickAction('translate', lang);
      });
      _qaTranslateSub!.appendChild(item);
    });

    // Insert after the translate button inside the menu
    if (_qaMenuEl && anchor.parentNode === _qaMenuEl) {
      anchor.after(_qaTranslateSub);
    }
  }

  function showQaMenu(): void {
    if (!_qaTriggerEl || !_qaActiveInput) return;
    removeQaMenu();

    _qaMenuEl = document.createElement('div');
    _qaMenuEl.className = 'qa-menu';

    QUICK_ACTIONS_DEF.forEach((qa) => {
      const item = document.createElement('button');
      item.className = 'qa-menu-item';
      item.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${qa.icon}</svg><span>${qa.label}</span>`;
      item.addEventListener('click', (e: MouseEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (qa.action === 'translate') {
          toggleTranslateSub(item as HTMLButtonElement);
        } else {
          executeQuickAction(qa.action);
        }
      });
      _qaMenuEl!.appendChild(item);
    });

    qaContainer.appendChild(_qaMenuEl);

    // Position below the trigger
    const trigRect = _qaTriggerEl.getBoundingClientRect();
    const menuGap = 6;
    _qaMenuEl.style.top = `${Math.round(trigRect.bottom + menuGap)}px`;
    _qaMenuEl.style.left = `${Math.round(trigRect.right - 150)}px`; // right-align roughly

    // Clamp to viewport
    requestAnimationFrame(() => {
      if (!_qaMenuEl) return;
      const mRect = _qaMenuEl.getBoundingClientRect();
      if (mRect.right > window.innerWidth - 8) {
        _qaMenuEl.style.left = `${Math.round(window.innerWidth - mRect.width - 8)}px`;
      }
      if (mRect.left < 8) {
        _qaMenuEl.style.left = '8px';
      }
      if (mRect.bottom > window.innerHeight - 8) {
        // Show above trigger instead
        _qaMenuEl.style.top = `${Math.round(trigRect.top - mRect.height - menuGap)}px`;
      }
    });
  }

  // Attach trigger on focus
  function onFieldFocus(e: FocusEvent): void {
    const target = e.target as Element;
    if (!target || !isEligibleInput(target)) return;
    const rect = target.getBoundingClientRect();
    if (rect.width < MIN_INPUT_WIDTH) return;

    _qaActiveInput = target as HTMLInputElement | HTMLTextAreaElement;
    positionTrigger(target as HTMLElement);
  }

  // Remove trigger on blur (with small delay so menu clicks register)
  function onFieldBlur(_e: FocusEvent): void {
    setTimeout(() => {
      // If focus moved to our menu, don't remove
      if (_qaMenuEl || _qaTranslateSub) return;
      // Check if the new active element is the same input
      const active = document.activeElement;
      if (active && active === _qaActiveInput) return;
      removeQaTrigger();
    }, 200);
  }

  // Reposition trigger on scroll/resize
  function repositionQaTrigger(): void {
    if (_qaActiveInput) {
      positionTrigger(_qaActiveInput as HTMLElement);
    }
  }

  // Close menu when clicking outside
  document.addEventListener('mousedown', (e: MouseEvent) => {
    if (!_qaMenuEl && !_qaTriggerEl) return;
    // Check if click is inside our shadow DOM
    const path = e.composedPath();
    if (path.includes(qaHost)) return;
    removeQaMenu();
  }, true);

  // Listen for focus/blur on the document (delegation)
  document.addEventListener('focusin', onFieldFocus, true);
  document.addEventListener('focusout', onFieldBlur, true);
  window.addEventListener('scroll', repositionQaTrigger, { passive: true });
  window.addEventListener('resize', repositionQaTrigger, { passive: true });

  // MutationObserver: detect if input is removed while trigger is visible
  const qaObserver = new MutationObserver(() => {
    if (_qaActiveInput && !document.body.contains(_qaActiveInput)) {
      removeQaTrigger();
    }
  });
  qaObserver.observe(document.body, { childList: true, subtree: true });
}

// ── YouTube Subtitle Interception Relay ───────────────────────────────────────

export function initYoutubeRelay(safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  document.addEventListener('aura-yt-subtitles', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      safeSend({
        type: 'YT_SUBTITLES',
        videoId: d.videoId || '',
        lang: d.lang || '',
        segments: d.segments || [],
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);

  document.addEventListener('aura-yt-metadata', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      safeSend({
        type: 'YT_METADATA',
        videoId: d.videoId || '',
        title: d.title || '',
        duration: d.duration || 0,
        description: d.description || '',
        channelName: d.channelName || '',
        chapters: d.chapters || [],
        captionTracks: d.captionTracks || [],
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);
}

// ── Netflix Subtitle Interception Relay ───────────────────────────────────────

export function initNetflixRelay(safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  document.addEventListener('aura-netflix-subtitles', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      safeSend({
        type: 'NETFLIX_SUBTITLES',
        movieId: d.movieId || '',
        lang: d.lang || '',
        trackId: d.trackId || '',
        segments: d.segments || [],
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);

  document.addEventListener('aura-netflix-metadata', ((e: CustomEvent) => {
    try {
      const d = e.detail;
      safeSend({
        type: 'NETFLIX_METADATA',
        movieId: d.movieId || '',
        title: d.title || '',
        episodeTitle: d.episodeTitle || '',
        seasonNumber: d.seasonNumber || 0,
        episodeNumber: d.episodeNumber || 0,
        duration: d.duration || 0,
      });
    } catch { /* extension context may be invalidated */ }
  }) as EventListener);
}

// ── Full Page Translation ─────────────────────────────────────────────────────

const TRANSLATABLE_SELECTORS = 'p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption';
const AURA_TRANSLATE_ATTR = 'data-aura-translated';
const BATCH_SIZE = 10;
const MAX_CONCURRENT = 10;

interface TranslationState {
  mode: 'bilingual' | 'translated';
  targetLang: string;
  active: boolean;
  badge: HTMLDivElement | null;
  elements: { original: HTMLElement; translation: HTMLDivElement }[];
  activeCount: number;
}

export function initTranslation(ext: typeof chrome): {
  start(targetLang: string): Promise<void>;
  remove(): void;
  setMode(mode: 'bilingual' | 'translated'): void;
} {
  const state: TranslationState = {
    mode: 'bilingual',
    targetLang: 'English',
    active: false,
    badge: null,
    elements: [],
    activeCount: 0,
  };

  function getTranslatableElements(): HTMLElement[] {
    const all = document.querySelectorAll<HTMLElement>(TRANSLATABLE_SELECTORS);
    const results: HTMLElement[] = [];
    for (const el of all) {
      if (el.hasAttribute(AURA_TRANSLATE_ATTR)) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) continue;
      if (el.closest('#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge')) continue;
      if (el.tagName === 'SPAN' && (el.textContent || '').trim().length <= 20) continue;
      const text = (el.textContent || '').trim();
      if (text.length < 5) continue;
      results.push(el);
    }
    return results;
  }

  function createTranslationElement(originalEl: HTMLElement): HTMLDivElement {
    const translationDiv = document.createElement('div');
    translationDiv.className = 'aura-page-translation';
    translationDiv.setAttribute('data-aura-translation', 'true');
    Object.assign(translationDiv.style, {
      borderLeft: '2px solid rgba(124, 58, 237, 0.6)',
      background: 'rgba(124, 58, 237, 0.05)',
      padding: '6px 10px',
      marginTop: '4px',
      marginBottom: '4px',
      fontSize: '0.95em',
      color: 'inherit',
      opacity: '0',
      fontFamily: 'inherit',
      lineHeight: '1.5',
      borderRadius: '0 4px 4px 0',
      transition: 'opacity 0.3s ease',
      fontStyle: 'italic',
    });
    translationDiv.textContent = 'Translating...';
    translationDiv.style.color = 'rgba(124, 58, 237, 0.5)';

    originalEl.setAttribute(AURA_TRANSLATE_ATTR, 'true');
    originalEl.after(translationDiv);

    // Fade in the placeholder
    requestAnimationFrame(() => { translationDiv.style.opacity = '0.6'; });
    return translationDiv;
  }

  function fadeInTranslation(el: HTMLDivElement, text: string): void {
    el.style.opacity = '0';
    el.textContent = text;
    el.style.fontStyle = 'normal';
    el.style.color = 'inherit';
    requestAnimationFrame(() => { el.style.opacity = '0.85'; });
  }

  function translateBatchRequest(texts: string[], lang: string): Promise<string[]> {
    return new Promise((resolve) => {
      try {
        ext.runtime.sendMessage(
          { type: 'TRANSLATE_BATCH', texts, targetLang: lang },
          (response: { ok: boolean; translations?: string[]; error?: string }) => {
            if (ext.runtime.lastError) {
              resolve(texts.map(() => '[Translation failed]'));
              return;
            }
            if (response?.ok && response.translations) {
              resolve(response.translations);
            } else {
              resolve(texts.map(() => response?.error || '[Translation failed]'));
            }
          }
        );
      } catch {
        resolve(texts.map(() => '[Translation failed]'));
      }
    });
  }

  function updateBadgeText(): void {
    if (!state.badge) return;
    const modeBtn = state.badge.querySelector('[data-badge-mode]') as HTMLElement | null;
    if (modeBtn) {
      modeBtn.textContent = state.mode === 'bilingual' ? 'Bilingual' : 'Translated';
    }
  }

  function setMode(mode: 'bilingual' | 'translated'): void {
    state.mode = mode;
    for (const pair of state.elements) {
      if (mode === 'translated') {
        pair.original.style.display = 'none';
        pair.translation.style.marginTop = '0';
      } else {
        pair.original.style.display = '';
        pair.translation.style.marginTop = '4px';
      }
    }
    updateBadgeText();
  }

  function remove(): void {
    state.active = false;
    for (const pair of state.elements) {
      pair.translation.remove();
      pair.original.removeAttribute(AURA_TRANSLATE_ATTR);
      pair.original.style.display = '';
    }
    state.elements = [];
    if (state.badge) {
      state.badge.remove();
      state.badge = null;
    }
  }

  function showTranslateBadge(): void {
    if (state.badge) { state.badge.remove(); state.badge = null; }

    state.badge = document.createElement('div');
    state.badge.className = 'aura-translate-badge';
    Object.assign(state.badge.style, {
      position: 'fixed',
      bottom: '20px',
      right: '20px',
      zIndex: '2147483646',
      background: 'rgba(10, 8, 24, 0.92)',
      backdropFilter: 'blur(20px) saturate(1.5)',
      WebkitBackdropFilter: 'blur(20px) saturate(1.5)',
      border: '1px solid rgba(124, 58, 237, 0.35)',
      borderRadius: '12px',
      padding: '8px 12px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      boxShadow: '0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset',
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",
      fontSize: '12px',
      color: 'rgba(226, 232, 240, 0.9)',
    });

    // Purple status dot
    const dot = document.createElement('span');
    Object.assign(dot.style, {
      width: '6px', height: '6px', borderRadius: '50%',
      background: '#7c3aed', flexShrink: '0',
    });
    state.badge.appendChild(dot);

    // Label
    const label = document.createElement('span');
    label.style.color = 'rgba(160, 148, 210, 0.8)';
    label.textContent = 'Translation active';
    state.badge.appendChild(label);

    // Separator
    const sep1 = document.createElement('span');
    Object.assign(sep1.style, { width: '1px', height: '14px', background: 'rgba(255,255,255,0.1)', flexShrink: '0' });
    state.badge.appendChild(sep1);

    // Language display
    const langSpan = document.createElement('span');
    langSpan.setAttribute('data-badge-lang', '');
    langSpan.textContent = state.targetLang;
    langSpan.style.color = 'rgba(124, 58, 237, 0.9)';
    langSpan.style.fontWeight = '600';
    state.badge.appendChild(langSpan);

    const badgeBtnBase: Record<string, string> = {
      background: 'rgba(124, 58, 237, 0.15)',
      border: '1px solid rgba(124, 58, 237, 0.3)',
      borderRadius: '6px',
      color: 'rgba(226, 232, 240, 0.9)',
      padding: '3px 8px',
      cursor: 'pointer',
      fontSize: '11px',
      fontFamily: 'inherit',
      transition: 'background 0.15s, border-color 0.15s',
    };

    // Mode toggle button
    const modeBtn = document.createElement('button');
    modeBtn.setAttribute('data-badge-mode', '');
    modeBtn.textContent = 'Bilingual';
    Object.assign(modeBtn.style, badgeBtnBase);
    modeBtn.addEventListener('mouseenter', () => { modeBtn.style.background = 'rgba(124, 58, 237, 0.3)'; });
    modeBtn.addEventListener('mouseleave', () => { modeBtn.style.background = 'rgba(124, 58, 237, 0.15)'; });
    modeBtn.addEventListener('click', () => {
      setMode(state.mode === 'bilingual' ? 'translated' : 'bilingual');
    });
    state.badge.appendChild(modeBtn);

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.textContent = '\u2715';
    Object.assign(removeBtn.style, { ...badgeBtnBase, padding: '3px 6px', color: 'rgba(226, 232, 240, 0.6)' });
    removeBtn.title = 'Remove translation';
    removeBtn.addEventListener('mouseenter', () => {
      removeBtn.style.background = 'rgba(239, 68, 68, 0.2)';
      removeBtn.style.borderColor = 'rgba(239, 68, 68, 0.4)';
      removeBtn.style.color = 'rgba(239, 68, 68, 0.9)';
    });
    removeBtn.addEventListener('mouseleave', () => {
      removeBtn.style.background = 'rgba(124, 58, 237, 0.15)';
      removeBtn.style.borderColor = 'rgba(124, 58, 237, 0.3)';
      removeBtn.style.color = 'rgba(226, 232, 240, 0.6)';
    });
    removeBtn.addEventListener('click', () => { remove(); });
    state.badge.appendChild(removeBtn);

    document.body.appendChild(state.badge);
  }

  async function start(targetLang: string): Promise<void> {
    state.targetLang = targetLang;
    state.active = true;
    state.mode = 'bilingual';
    state.elements = [];
    state.activeCount = 0;

    showTranslateBadge();

    const elements = getTranslatableElements();
    if (elements.length === 0) return;

    // Create translation placeholders for all elements
    const pairs: { original: HTMLElement; translation: HTMLDivElement; text: string }[] = [];
    for (const el of elements) {
      const text = (el.textContent || '').trim();
      if (!text) continue;
      const translationDiv = createTranslationElement(el);
      state.elements.push({ original: el, translation: translationDiv });
      pairs.push({ original: el, translation: translationDiv, text });
    }

    // Split into batches
    const batches: typeof pairs[] = [];
    for (let i = 0; i < pairs.length; i += BATCH_SIZE) {
      batches.push(pairs.slice(i, i + BATCH_SIZE));
    }

    const processBatch = async (batch: typeof pairs): Promise<void> => {
      while (state.activeCount >= MAX_CONCURRENT) {
        await new Promise(r => setTimeout(r, 100));
      }
      if (!state.active) return;

      state.activeCount++;
      try {
        const texts = batch.map(p => p.text);
        const translations = await translateBatchRequest(texts, state.targetLang);
        if (!state.active) return;

        batch.forEach((pair, idx) => {
          if (!state.active) return;
          fadeInTranslation(pair.translation, translations[idx] || '[No translation]');
          if (state.mode === 'translated') {
            pair.original.style.display = 'none';
          }
        });
      } finally {
        state.activeCount--;
      }
    };

    const promises = batches.map(batch => processBatch(batch));
    await Promise.all(promises);
  }

  return { start, remove, setMode };
}

// ── Full Page Extraction ──────────────────────────────────────────────────────

// CSS properties for style extraction (shared with capture module)
const PAGE_CSS_PROPS: string[] = [
  'display', 'position', 'flex-direction', 'align-items', 'justify-content',
  'gap', 'flex-wrap', 'flex', 'flex-grow', 'flex-shrink',
  'width', 'height', 'min-width', 'min-height', 'max-width', 'max-height',
  'padding', 'padding-top', 'padding-right', 'padding-bottom', 'padding-left',
  'margin', 'margin-top', 'margin-right', 'margin-bottom', 'margin-left',
  'border', 'border-radius', 'border-color', 'border-width', 'border-style',
  'background', 'background-color', 'background-image', 'background-size',
  'color', 'font-size', 'font-weight', 'font-family', 'line-height',
  'letter-spacing', 'text-align', 'text-decoration', 'text-transform',
  'box-shadow', 'opacity', 'overflow', 'z-index',
  'grid-template-columns', 'grid-template-rows', 'grid-gap',
  'transform', 'transition',
];

function extractComputedStylesForPage(el: Element): Record<string, string> {
  const styles = window.getComputedStyle(el);
  const result: Record<string, string> = {};
  for (const prop of PAGE_CSS_PROPS) {
    const val = styles.getPropertyValue(prop);
    if (val && val !== 'none' && val !== 'normal' && val !== 'auto' && val !== '0px' && val !== 'rgba(0, 0, 0, 0)') {
      result[prop] = val;
    }
  }
  return result;
}

function buildPageCssSelector(el: Element): string {
  const tag = el.tagName.toLowerCase();
  const cls = el.className && typeof el.className === 'string'
    ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.')
    : '';
  return tag + cls;
}

export function extractFullPageData(): {
  html: string;
  css: string;
  css_map: Record<string, Record<string, string>>;
  colors: string[];
  fonts: string[];
  metadata: {
    title: string;
    description: string;
    og_image: string;
    og_title: string;
    og_description: string;
    og_type: string;
    og_site_name: string;
    favicon: string;
  };
  source_url: string;
  viewport: { width: number; height: number };
  asset_urls: { images: string[]; stylesheets: string[] };
  responsive_info: { viewport_width: number; media_queries: string[] };
  element_count: number;
} {
  // 1. Clean full page HTML
  const clone = document.documentElement.cloneNode(true) as HTMLElement;

  // Remove scripts, tracking pixels, ads, cookie banners, noscript
  const removeSelectors = [
    'script', 'noscript', 'iframe[src*="ads"]', 'iframe[src*="track"]',
    'iframe[src*="pixel"]', 'iframe[width="0"]', 'iframe[height="0"]',
    'img[src*="pixel"]', 'img[src*="track"]', 'img[width="1"]', 'img[height="1"]',
    '[id*="cookie"]', '[class*="cookie"]', '[id*="consent"]', '[class*="consent"]',
    '[id*="gdpr"]', '[class*="gdpr"]', '[id*="onetrust"]', '[class*="onetrust"]',
    '[id*="CybotCookiebot"]', '[data-testid*="cookie"]',
    '[id*="ad-"]', '[class*="ad-container"]', '[class*="ad-wrapper"]',
    'link[rel="preconnect"]', 'link[rel="dns-prefetch"]',
    'meta[http-equiv="Content-Security-Policy"]',
    'style[data-emotion]', // runtime CSS-in-JS noise
  ];
  for (const sel of removeSelectors) {
    try {
      clone.querySelectorAll(sel).forEach(el => el.remove());
    } catch (_e) { /* invalid selector, skip */ }
  }
  // Remove all inline event handlers
  clone.querySelectorAll('*').forEach(el => {
    const attrs = el.getAttributeNames();
    for (const attr of attrs) {
      if (attr.startsWith('on') || attr === 'data-analytics' || attr === 'data-tracking') {
        el.removeAttribute(attr);
      }
    }
  });
  const cleanHtml = clone.outerHTML;

  // 2. Extract computed styles for key elements (up to 200)
  const cssMap: Record<string, Record<string, string>> = {};
  const keySelectors = [
    'body', 'header', 'nav', 'main', 'footer', 'aside', 'section', 'article',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button', 'input', 'textarea',
    'ul', 'ol', 'li', 'img', 'form', 'table', 'th', 'td',
    '[class*="hero"]', '[class*="card"]', '[class*="btn"]', '[class*="nav"]',
    '[class*="header"]', '[class*="footer"]', '[class*="sidebar"]',
    '[class*="container"]', '[class*="wrapper"]', '[class*="grid"]',
    '[class*="flex"]', '[class*="modal"]', '[class*="banner"]',
  ];
  let styleCount = 0;
  for (const sel of keySelectors) {
    if (styleCount >= 200) break;
    try {
      const els = document.querySelectorAll(sel);
      for (const el of els) {
        if (styleCount >= 200) break;
        const styles = extractComputedStylesForPage(el);
        if (Object.keys(styles).length > 0) {
          const key = buildPageCssSelector(el);
          const finalKey = cssMap[key] ? `${key}:nth(${styleCount})` : key;
          cssMap[finalKey] = styles;
          styleCount++;
        }
      }
    } catch (_e) { /* invalid selector */ }
  }

  // Build CSS string
  const cssLines: string[] = [];
  for (const [selector, props] of Object.entries(cssMap)) {
    cssLines.push(`${selector} {`);
    for (const [prop, val] of Object.entries(props)) {
      cssLines.push(`  ${prop}: ${val};`);
    }
    cssLines.push('}');
    cssLines.push('');
  }
  const cssString = cssLines.join('\n');

  // 3. Extract color palette from computed styles
  const colorSet = new Set<string>();
  const colorProps = ['color', 'background-color', 'border-color', 'outline-color'];
  const sampleEls = document.querySelectorAll('*');
  let sampleCount = 0;
  for (const el of sampleEls) {
    if (sampleCount >= 500) break;
    const cs = window.getComputedStyle(el);
    for (const cp of colorProps) {
      const val = cs.getPropertyValue(cp);
      if (val && val !== 'rgba(0, 0, 0, 0)' && val !== 'transparent' && val !== 'inherit' && val !== 'initial') {
        colorSet.add(val);
      }
    }
    sampleCount++;
  }
  const colors = Array.from(colorSet).slice(0, 50);

  // 4. Extract font stack
  const fontSet = new Set<string>();
  for (const el of sampleEls) {
    if (fontSet.size >= 20) break;
    const cs = window.getComputedStyle(el);
    const ff = cs.getPropertyValue('font-family');
    if (ff) {
      // Extract individual font names
      const fonts = ff.split(',').map(f => f.trim().replace(/^["']|["']$/g, ''));
      for (const font of fonts) {
        if (font && !font.includes('inherit') && !font.includes('initial') && font.length < 50) {
          fontSet.add(font);
        }
      }
    }
  }
  const fonts = Array.from(fontSet).slice(0, 20);

  // 5. Page metadata
  const getMeta = (name: string): string => {
    const el = document.querySelector(`meta[property="${name}"], meta[name="${name}"]`);
    return el?.getAttribute('content') || '';
  };
  const faviconEl = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
  const metadata = {
    title: document.title || '',
    description: getMeta('description'),
    og_image: getMeta('og:image'),
    og_title: getMeta('og:title'),
    og_description: getMeta('og:description'),
    og_type: getMeta('og:type'),
    og_site_name: getMeta('og:site_name'),
    favicon: faviconEl?.getAttribute('href') || '',
  };

  // 6. Responsive info
  const viewport = {
    width: window.innerWidth,
    height: window.innerHeight,
  };

  // Try to detect media queries from stylesheets
  const mediaQueries: string[] = [];
  try {
    for (const sheet of document.styleSheets) {
      try {
        const rules = sheet.cssRules || sheet.rules;
        if (!rules) continue;
        for (const rule of rules) {
          if (rule instanceof CSSMediaRule && rule.conditionText) {
            if (!mediaQueries.includes(rule.conditionText)) {
              mediaQueries.push(rule.conditionText);
            }
            if (mediaQueries.length >= 20) break;
          }
        }
      } catch (_e) { /* cross-origin stylesheet */ }
      if (mediaQueries.length >= 20) break;
    }
  } catch (_e) { /* no access */ }

  // 7. Asset URLs
  const images: string[] = [];
  document.querySelectorAll('img[src]').forEach(img => {
    const src = img.getAttribute('src');
    if (src && !src.startsWith('data:') && images.length < 50) {
      try {
        images.push(new URL(src, location.href).href);
      } catch (_e) {
        images.push(src);
      }
    }
  });

  const stylesheets: string[] = [];
  document.querySelectorAll('link[rel="stylesheet"][href]').forEach(link => {
    const href = link.getAttribute('href');
    if (href && stylesheets.length < 20) {
      try {
        stylesheets.push(new URL(href, location.href).href);
      } catch (_e) {
        stylesheets.push(href);
      }
    }
  });

  // 8. Element count
  const elementCount = document.querySelectorAll('*').length;

  return {
    html: cleanHtml,
    css: cssString,
    css_map: cssMap,
    colors,
    fonts,
    metadata,
    source_url: location.href,
    viewport,
    asset_urls: { images, stylesheets },
    responsive_info: { viewport_width: viewport.width, media_queries: mediaQueries },
    element_count: elementCount,
  };
}

// ── Google SERP AI Answer Card ────────────────────────────────────────────────

export function initGoogleSerp(ext: typeof chrome, safeSend: (msg: any, cb?: (r: any) => void) => void): void {
  let SERP_BACKEND = 'https://aura-elnur.duckdns.org';
  let SERP_API_KEY = '';

  /** Read backend URL and API key from chrome.storage.local. */
  function loadSerpConfig(): Promise<void> {
    return new Promise((resolve) => {
      if (!ext?.storage?.local) {
        resolve();
        return;
      }
      ext.storage.local.get(['backendUrl', 'apiKey'], (d: any) => {
        if (d?.backendUrl?.trim()) SERP_BACKEND = d.backendUrl.trim().replace(/\/+$/, '');
        if (d?.apiKey?.trim()) SERP_API_KEY = d.apiKey.trim();
        resolve();
      });
    });
  }

  // Load config on init and listen for storage changes
  loadSerpConfig();
  if (ext?.storage?.onChanged) {
    ext.storage.onChanged.addListener((changes: any, area: string) => {
      if (area !== 'local') return;
      if (changes.backendUrl?.newValue) {
        SERP_BACKEND = changes.backendUrl.newValue.trim().replace(/\/+$/, '');
      }
      if (changes.apiKey?.newValue !== undefined) {
        SERP_API_KEY = changes.apiKey.newValue?.trim() || '';
      }
    });
  }

  function isGoogleSearchPage(): boolean {
    const hostname = window.location.hostname;
    const pathname = window.location.pathname;
    const params = new URLSearchParams(window.location.search);
    if (!hostname.match(/^(www\.)?google\./)) return false;
    if (pathname !== '/search') return false;
    if (!params.get('q')) return false;
    const tbm = params.get('tbm');
    if (tbm && ['isch', 'lcl', 'vid', 'shop', 'nws', 'bks', 'fin'].includes(tbm)) return false;
    const udm = params.get('udm');
    if (udm && ['2', '14'].includes(udm)) return false;
    return true;
  }

  function getSearchQuery(): string {
    const params = new URLSearchParams(window.location.search);
    const qParam = params.get('q') || '';
    if (qParam) return qParam;
    const input = document.querySelector<HTMLInputElement>('input[name="q"]');
    return input?.value || '';
  }

  function detectGoogleTheme(): 'dark' | 'light' {
    const bg = window.getComputedStyle(document.body).backgroundColor;
    if (!bg || bg === 'transparent') return 'light';
    const rgbMatch = bg.match(/\d+/g);
    if (rgbMatch && rgbMatch.length >= 3) {
      const [r, g, b] = rgbMatch.map(Number);
      const luminance = (0.299 * r + 0.587 * g + 0.114 * b);
      return luminance < 128 ? 'dark' : 'light';
    }
    return 'light';
  }

  function serpEscapeHtml(text: string): string {
    return text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /**
   * Minimal markdown-to-HTML renderer for SERP answers.
   * Security: input is escaped first, then only safe structural tags are
   * introduced by regex. Link hrefs are constrained to https?:// by the
   * regex pattern. Final output is sanitized to strip any unexpected tags
   * or attributes as defense-in-depth against backend compromise.
   */
  function serpRenderMarkdown(text: string): string {
    let html = serpEscapeHtml(text);

    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Inline code: `text`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Links: [text](url) — href constrained to https?:// only
    html = html.replace(
      /\[([^\]]+)\]\((https?:\/\/[^)"]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Unordered lists: lines starting with - or *
    html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, '<li>$1</li>');
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Ordered lists: lines starting with 1. 2. etc.
    html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, '<li>$1</li>');

    // Paragraphs: double newlines
    html = html.replace(/\n\n+/g, '</p><p>');
    html = '<p>' + html + '</p>';

    // Single newlines to line breaks
    html = html.replace(/\n/g, '<br>');

    // Clean empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    // Defense-in-depth: strip any tags that aren't in our safe list
    html = html.replace(/<\/?(?!(?:strong|em|code|a|li|ul|ol|p|br)\b)[^>]*>/gi, '');

    return html;
  }

  function serpAddCitations(bodyEl: HTMLElement, fullText: string): void {
    const citationRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
    const citations: Array<{ title: string; url: string }> = [];
    let citMatch: RegExpExecArray | null;
    while ((citMatch = citationRegex.exec(fullText)) !== null) {
      citations.push({ title: citMatch[1], url: citMatch[2] });
    }
    if (citations.length === 0) return;

    const citationsContainer = document.createElement('div');
    citationsContainer.className = 'serp-citations';
    const citLabel = document.createElement('div');
    citLabel.className = 'serp-citations-label';
    citLabel.textContent = 'Sources';
    citationsContainer.appendChild(citLabel);

    const citList = document.createElement('div');
    citList.className = 'serp-citation-list';
    citations.forEach((cit, idx) => {
      const chip = document.createElement('a');
      chip.className = 'serp-citation-chip';
      chip.href = cit.url;
      chip.target = '_blank';
      chip.rel = 'noopener noreferrer';
      const num = document.createElement('span');
      num.className = 'serp-citation-num';
      num.textContent = String(idx + 1);
      chip.appendChild(num);
      const chipText = document.createTextNode(' ' + cit.title);
      chip.appendChild(chipText);
      citList.appendChild(chip);
    });
    citationsContainer.appendChild(citList);
    bodyEl.appendChild(citationsContainer);
  }

  function serpAddFooter(cardEl: HTMLElement, query: string, fullText: string): void {
    const footer = document.createElement('div');
    footer.className = 'serp-footer';

    const followupBtn = document.createElement('button');
    followupBtn.className = 'serp-followup-btn';
    followupBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up`;
    followupBtn.addEventListener('click', () => {
      safeSend({
        type: 'OPEN_WITH_TEXT',
        action: 'ask',
        text: `I searched for "${query}" and got the following AI answer:\n\n${fullText}\n\nI have a follow-up question: `,
        url: window.location.href,
        title: document.title,
      });
    });

    const powered = document.createElement('span');
    powered.className = 'serp-powered';
    powered.textContent = 'Powered by AURA';

    footer.appendChild(followupBtn);
    footer.appendChild(powered);
    cardEl.appendChild(footer);
  }

  async function initGoogleSerpIntegration(): Promise<void> {
    if (!isGoogleSearchPage()) return;

    // Check user preference
    const stored = await new Promise<Record<string, any>>((resolve) => {
      ext.storage.local.get(['aura_serp_hidden'], resolve);
    });
    if (stored.aura_serp_hidden) return;

    const query = getSearchQuery();
    if (!query) return;

    // Create floating panel on the RIGHT side of the page (like Sider)
    const serpHost = document.createElement('div');
    serpHost.id = 'aura-serp-host';
    Object.assign(serpHost.style, {
      position: 'fixed',
      top: '80px',
      right: '16px',
      width: '340px',
      maxHeight: 'calc(100vh - 100px)',
      zIndex: '2147483640',
      pointerEvents: 'auto',
    });
    document.documentElement.appendChild(serpHost);

    const serpShadow = serpHost.attachShadow({ mode: 'closed' });
    const theme = detectGoogleTheme();
    const isDark = theme === 'dark';

    // Styles
    const serpStyle = document.createElement('style');
    serpStyle.textContent = `
      @keyframes serp-fade-in {
        from { opacity: 0; transform: translateY(-8px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes serp-pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
      }

      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      :host {
        display: block;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .serp-card {
        background: ${isDark
          ? 'rgba(30, 27, 48, 0.92)'
          : 'rgba(255, 255, 255, 0.95)'};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${isDark
          ? '0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)'
          : '0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)'};
        border: 1px solid ${isDark
          ? 'rgba(124, 58, 237, 0.2)'
          : 'rgba(124, 58, 237, 0.15)'};
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${isDark
          ? 'rgba(124, 58, 237, 0.35)'
          : 'rgba(124, 58, 237, 0.3)'};
      }

      .serp-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
      }
      .serp-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
      }
      .serp-logo {
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: ${isDark ? 'rgba(160, 148, 210, 0.9)' : 'rgba(124, 58, 237, 0.85)'};
        background: ${isDark
          ? 'rgba(124, 58, 237, 0.12)'
          : 'rgba(124, 58, 237, 0.08)'};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${isDark ? 'rgba(226, 232, 240, 0.9)' : 'rgba(30, 27, 48, 0.9)'};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.6)'};
        margin-left: 6px;
      }

      .serp-controls {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .serp-ctrl-btn {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        border: none;
        background: transparent;
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.5)'};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.15)' : 'rgba(124, 58, 237, 0.1)'};
        color: ${isDark ? 'rgba(224, 214, 255, 1)' : 'rgba(124, 58, 237, 0.9)'};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${isDark ? 'rgba(226, 232, 240, 0.85)' : 'rgba(30, 27, 48, 0.85)'};
        overflow: hidden;
        transition: max-height 0.3s ease;
      }
      .serp-body.collapsed {
        max-height: 0 !important;
        margin: 0;
        padding: 0;
      }

      .serp-loading {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0;
      }
      .serp-loading-dots {
        display: flex;
        gap: 4px;
      }
      .serp-loading-dots span {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: ${isDark ? 'rgba(124, 58, 237, 0.6)' : 'rgba(124, 58, 237, 0.5)'};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.6)' : 'rgba(100, 90, 140, 0.6)'};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${isDark ? 'rgba(226, 232, 240, 0.95)' : 'rgba(30, 27, 48, 0.95)'};
      }
      .serp-answer code {
        background: ${isDark ? 'rgba(124, 58, 237, 0.1)' : 'rgba(124, 58, 237, 0.06)'};
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 13px;
        font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
      }
      .serp-answer ul, .serp-answer ol {
        padding-left: 20px;
        margin-bottom: 8px;
      }
      .serp-answer li { margin-bottom: 4px; }
      .serp-answer a {
        color: ${isDark ? 'rgba(160, 148, 255, 0.9)' : 'rgba(100, 58, 237, 0.9)'};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.5)' : 'rgba(100, 90, 140, 0.5)'};
        margin-bottom: 6px;
      }
      .serp-citation-list {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .serp-citation-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: ${isDark ? 'rgba(124, 58, 237, 0.1)' : 'rgba(124, 58, 237, 0.06)'};
        border: 1px solid ${isDark ? 'rgba(124, 58, 237, 0.15)' : 'rgba(124, 58, 237, 0.1)'};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${isDark ? 'rgba(200, 180, 255, 0.8)' : 'rgba(100, 58, 237, 0.8)'};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.12)'};
        border-color: ${isDark ? 'rgba(124, 58, 237, 0.3)' : 'rgba(124, 58, 237, 0.2)'};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.1)'};
        font-size: 10px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
      }

      .serp-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 14px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${isDark ? 'rgba(124, 58, 237, 0.12)' : 'rgba(124, 58, 237, 0.08)'};
        border: 1px solid ${isDark ? 'rgba(124, 58, 237, 0.2)' : 'rgba(124, 58, 237, 0.15)'};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${isDark ? 'rgba(200, 180, 255, 0.9)' : 'rgba(100, 58, 237, 0.9)'};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${isDark ? 'rgba(124, 58, 237, 0.22)' : 'rgba(124, 58, 237, 0.15)'};
        border-color: ${isDark ? 'rgba(124, 58, 237, 0.35)' : 'rgba(124, 58, 237, 0.3)'};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${isDark ? 'rgba(160, 148, 210, 0.35)' : 'rgba(100, 90, 140, 0.35)'};
      }

      .serp-offline {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 0;
      }
      .serp-offline-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: rgba(239, 68, 68, 0.6);
        flex-shrink: 0;
      }
      .serp-offline-text {
        font-size: 13px;
        color: ${isDark ? 'rgba(226, 232, 240, 0.5)' : 'rgba(30, 27, 48, 0.5)'};
      }

      .serp-error {
        font-size: 13px;
        color: ${isDark ? 'rgba(239, 150, 150, 0.8)' : 'rgba(200, 50, 50, 0.7)'};
        padding: 4px 0;
      }
    `;
    serpShadow.appendChild(serpStyle);

    // Card container
    const card = document.createElement('div');
    card.className = 'serp-card';

    // Header
    const serpHeader = document.createElement('div');
    serpHeader.className = 'serp-header';

    const headerLeft = document.createElement('div');
    headerLeft.className = 'serp-header-left';

    const serpLogo = document.createElement('div');
    serpLogo.className = 'serp-logo';
    serpLogo.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;

    const titleWrap = document.createElement('div');
    const titleText = document.createElement('span');
    titleText.className = 'serp-title';
    titleText.textContent = 'AI Answer';
    const titleSub = document.createElement('span');
    titleSub.className = 'serp-title-sub';
    titleSub.textContent = 'by AURA';
    titleWrap.appendChild(titleText);
    titleWrap.appendChild(titleSub);

    headerLeft.appendChild(serpLogo);
    headerLeft.appendChild(titleWrap);

    const controls = document.createElement('div');
    controls.className = 'serp-controls';

    // Collapse/expand toggle
    const collapseBtn = document.createElement('button');
    collapseBtn.className = 'serp-ctrl-btn';
    collapseBtn.title = 'Collapse';
    collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;

    // Hide toggle
    const hideBtn = document.createElement('button');
    hideBtn.className = 'serp-ctrl-btn';
    hideBtn.title = 'Hide AURA answers';
    hideBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;

    controls.appendChild(collapseBtn);
    controls.appendChild(hideBtn);

    serpHeader.appendChild(headerLeft);
    serpHeader.appendChild(controls);
    card.appendChild(serpHeader);

    // Body
    const serpBody = document.createElement('div');
    serpBody.className = 'serp-body';

    // Loading state
    const serpLoading = document.createElement('div');
    serpLoading.className = 'serp-loading';
    const serpDots = document.createElement('div');
    serpDots.className = 'serp-loading-dots';
    serpDots.innerHTML = '<span></span><span></span><span></span>';
    const serpLoadingText = document.createElement('span');
    serpLoadingText.className = 'serp-loading-text';
    serpLoadingText.textContent = `Thinking about "${query.slice(0, 60)}${query.length > 60 ? '...' : ''}"`;
    serpLoading.appendChild(serpDots);
    serpLoading.appendChild(serpLoadingText);
    serpBody.appendChild(serpLoading);

    card.appendChild(serpBody);
    serpShadow.appendChild(card);

    // Collapse toggle logic
    let isSerpCollapsed = false;
    collapseBtn.addEventListener('click', () => {
      isSerpCollapsed = !isSerpCollapsed;
      if (isSerpCollapsed) {
        serpBody.classList.add('collapsed');
        collapseBtn.title = 'Expand';
        collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
      } else {
        serpBody.classList.remove('collapsed');
        collapseBtn.title = 'Collapse';
        collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
      }
    });

    // Hide toggle: persist and remove card
    hideBtn.addEventListener('click', () => {
      ext.storage.local.set({ aura_serp_hidden: true });
      serpHost.remove();
    });

    // Fetch AI answer via background script (avoids CORS)
    try {
      const fetchBody = JSON.stringify({
        message: query,
        conversation_id: '__serp_answer__',
        stream: false,
        system_context: `The user searched Google for: "${query}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`,
      });

      // Try background proxy first, fall back to direct fetch
      let proxyResult: any = null;
      (async () => {
        try {
          proxyResult = await new Promise((resolve, reject) => {
            ext.runtime.sendMessage(
              { type: 'SERP_FETCH', url: `${SERP_BACKEND}/api/chat`, body: fetchBody, apiKey: SERP_API_KEY },
              (response: any) => {
                if (ext.runtime.lastError) {
                  reject(new Error(ext.runtime.lastError.message));
                } else {
                  resolve(response);
                }
              }
            );
          });
        } catch {
          // Background unavailable — try direct fetch as fallback
          const serpHeaders: Record<string, string> = { 'Content-Type': 'application/json' };
          if (SERP_API_KEY) serpHeaders['X-API-Key'] = SERP_API_KEY;
          const directResp = await fetch(`${SERP_BACKEND}/api/chat`, {
            method: 'POST',
            headers: serpHeaders,
            body: fetchBody,
            signal: AbortSignal.timeout(30000),
          });
          if (!directResp.ok) throw new Error(`HTTP ${directResp.status}`);
          proxyResult = { ok: true, text: await directResp.text() };
        }

        if (!proxyResult?.ok) {
          throw new Error(proxyResult?.error || 'Backend unreachable');
        }

        // Clear loading
        serpLoading.remove();

        const answerEl = document.createElement('div');
        answerEl.className = 'serp-answer';
        serpBody.appendChild(answerEl);

        // Parse the response (may be NDJSON or plain JSON)
        let fullText = '';
        const responseText = proxyResult.text || '';
        const lines = responseText.split('\n').filter((l: string) => l.trim());
        for (const line of lines) {
          try {
            const parsed = JSON.parse(line);
            if (parsed.chunk) fullText += parsed.chunk;
            else if (parsed.response) fullText = parsed.response;
            else if (parsed.content) fullText = parsed.content;
          } catch {
            fullText += line;
          }
        }
        if (!fullText.trim() && responseText.trim()) {
          fullText = responseText;
        }

        // Render the response
        answerEl.innerHTML = serpRenderMarkdown(fullText);

        if (!fullText.trim()) {
          answerEl.innerHTML = '<span class="serp-error">No response from AI.</span>';
          return;
        }

        serpAddCitations(serpBody, fullText);
        serpAddFooter(card, query, fullText);
      })().catch((_err: unknown) => {
        serpLoading.remove();
        const offline = document.createElement('div');
        offline.className = 'serp-offline';
        const offDot = document.createElement('div');
        offDot.className = 'serp-offline-dot';
        const offText = document.createElement('span');
        offText.className = 'serp-offline-text';
        offText.textContent = `AURA is offline — backend did not respond (${(_err as Error)?.message || 'timeout'})`;
        offline.appendChild(offDot);
        offline.appendChild(offText);
        serpBody.appendChild(offline);
      });

    } catch (_err: unknown) {
      serpLoading.remove();
      const offline = document.createElement('div');
      offline.className = 'serp-offline';
      const offDot = document.createElement('div');
      offDot.className = 'serp-offline-dot';
      const offText = document.createElement('span');
      offText.className = 'serp-offline-text';
      offText.textContent = `AURA is offline — backend did not respond`;
      offline.appendChild(offDot);
      offline.appendChild(offText);
      serpBody.appendChild(offline);
    }
  }

  // Boot Google SERP integration
  initGoogleSerpIntegration();
}

// ── Message Listener Setup ────────────────────────────────────────────────────

export interface MessageHandlers {
  extractMainContent(): any;
  serializeDOM(): SerializedElement[];
  execAction(params: ExecActionParams): ExecActionResult;
  showOcrOverlay(dataUrl: string, sendResponse: (r: any) => void): void;
  startPageTranslation(targetLang: string): Promise<void>;
  removePageTranslation(): void;
  setTranslateMode(mode: 'bilingual' | 'translated'): void;
  scrollToHighlight(id: string): void;
  showDock(): void;
  startCaptureMode(): void;
  stopCaptureMode(): void;
  extractFullPageData(): any;
  translateActive: boolean;
}

export function setupMessageListener(ext: typeof chrome, handlers: MessageHandlers): void {
  ext.runtime.onMessage.addListener(
    (
      msg: any,
      _sender: chrome.runtime.MessageSender,
      sendResponse: (response: any) => void
    ): boolean | undefined => {
      if (msg.type === 'EXTRACT_PAGE') {
        sendResponse(handlers.extractMainContent());
        return false;
      }

      if (msg.type === 'GET_DOM') {
        sendResponse({ ok: true, dom: handlers.serializeDOM(), url: location.href, title: document.title });
        return false;
      }

      if (msg.type === 'EXEC_ACTION') {
        sendResponse(handlers.execAction(msg.action));
        return false;
      }

      if (msg.type === 'FILL_FORM') {
        const fields = msg.fields as Array<{ selector: string; value: string }>;
        let filled = 0;
        for (const field of fields || []) {
          const result = handlers.execAction({ action: 'type', selector: field.selector, text: field.value });
          if (result.ok) filled++;
        }
        sendResponse({ ok: true, filled, total: fields?.length || 0 });
        return false;
      }

      if (msg.type === 'SHOW_OCR_OVERLAY') {
        handlers.showOcrOverlay(msg.dataUrl, sendResponse);
        return true; // async
      }

      if (msg.type === 'PAGE_TRANSLATE') {
        if (handlers.translateActive) handlers.removePageTranslation();
        handlers.startPageTranslation(msg.targetLang).then(() => {
          sendResponse({ ok: true });
        }).catch((err: Error) => {
          sendResponse({ ok: false, error: err.message });
        });
        return true; // async
      }

      if (msg.type === 'TRANSLATE_TOGGLE_MODE') {
        handlers.setTranslateMode(msg.mode);
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'TRANSLATE_REMOVE') {
        handlers.removePageTranslation();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'TRANSLATE_CHANGE_LANG') {
        if (handlers.translateActive) {
          handlers.removePageTranslation();
          handlers.startPageTranslation(msg.targetLang).then(() => {
            sendResponse({ ok: true });
          }).catch((err: Error) => {
            sendResponse({ ok: false, error: err.message });
          });
          return true;
        }
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'SCROLL_TO_HIGHLIGHT') {
        handlers.scrollToHighlight(msg.id);
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'SHOW_DOCK') {
        handlers.showDock();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'START_CAPTURE_MODE') {
        handlers.startCaptureMode();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'STOP_CAPTURE_MODE') {
        handlers.stopCaptureMode();
        sendResponse({ ok: true });
        return false;
      }

      if (msg.type === 'EXTRACT_FULL_PAGE') {
        try {
          const data = handlers.extractFullPageData();
          sendResponse({ ok: true, data });
        } catch (err: any) {
          sendResponse({ ok: false, error: err.message || 'Extraction failed' });
        }
        return false;
      }

      return undefined;
    }
  );
}
