import type {
  ContentModule,
  ContextStore,
  QuickActionMessage,
} from './types';
import { ANIM, GHOST_BAR, GLASS, Z_TOP } from './tokens';
import { flow } from './animator';

// ── SVG icons ─────────────────────────────────────────────────────────────────

const ICONS: Record<string, string> = {
  ask: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1L10 6H15L11 9.5L12.5 14.5L8 11.5L3.5 14.5L5 9.5L1 6H6L8 1Z"/>
  </svg>`,
  copy: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M4 4h7a1 1 0 011 1v7a1 1 0 01-1 1H4a1 1 0 01-1-1V5a1 1 0 011-1zm0-2a3 3 0 00-3 3v7a3 3 0 003 3h7a3 3 0 003-3V5a3 3 0 00-3-3H4z"/>
    <path d="M7 1h5a3 3 0 013 3v5h-2V4a1 1 0 00-1-1H7V1z"/>
  </svg>`,
  explain: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1l1.5 3 3.5.5-2.5 2.5.6 3.5L8 9 4.9 10.5l.6-3.5L3 4.5 6.5 4z"/>
    <path d="M2 13h12v1.5H2z" opacity=".5"/>
  </svg>`,
  summarize: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 3h12v2H2V3zm0 4h12v2H2V7zm0 4h8v2H2v-2z"/>
  </svg>`,
  translate: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.5"/>
    <path d="M8 1C5.5 4 5.5 12 8 15M8 1c2.5 3 2.5 11 0 14M1 8h14M2 5h12M2 11h12"/>
  </svg>`,
  highlight: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M5 1h6v10l-3 3-3-3V1zm1 1v9l2 2 2-2V2H6z"/>
  </svg>`,
  more: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <circle cx="3" cy="8" r="1.5"/>
    <circle cx="8" cy="8" r="1.5"/>
    <circle cx="13" cy="8" r="1.5"/>
  </svg>`,
  // image actions
  describe: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 2h12a1 1 0 011 1v10a1 1 0 01-1 1H2a1 1 0 01-1-1V3a1 1 0 011-1zm1 2v8h10V4H3zm2 2a1 1 0 110 2 1 1 0 010-2zm7 4H4l2-3 1.5 2 2-3L12 10z"/>
  </svg>`,
  edit: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M11.5 1.5l3 3-8 8H3.5v-3l8-8zM10 3L13 6l-7 7H4v-2L10 3z"/>
  </svg>`,
  save: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 10L4.5 6.5l1-1L7 7V1h2v6l1.5-1.5 1 1L8 10zm-5 2h10v2H3v-2z"/>
  </svg>`,
  // extended actions
  rewrite: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 8a6 6 0 1110.76-3H10v2h5V2h-2v2.5A8 8 0 102 8h2z"/>
  </svg>`,
  grammar: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M1 2h14v2H1V2zm0 4h10v2H1V6zm0 4h14v2H1v-2zm0 4h6v2H1v-2z"/>
  </svg>`,
  define: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M7 1H2v14h12V6l-5-5H7zm0 1.5L11.5 7H7V2.5zM4 4h2v2H4V4zm0 3h8v2H4V7zm0 3h8v2H4v-2z"/>
  </svg>`,
  'read-aloud': `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M7 3v10L3.5 10H1V6h2.5L7 3zm2 2a4 4 0 010 6V9.5a2 2 0 000-3V5zm2-2a7 7 0 010 10V12.5a5 5 0 000-9V3z"/>
  </svg>`,
  // generic fallback
  review: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 2a6 6 0 100 12A6 6 0 008 2zm0 2a4 4 0 110 8A4 4 0 018 4zm0 2a2 2 0 100 4 2 2 0 000-4z"/>
  </svg>`,
  debug: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M8 1a4 4 0 00-4 4v1H2v2h2v1a4 4 0 004 4 4 4 0 004-4V8h2V6h-2V5a4 4 0 00-4-4zm0 2a2 2 0 012 2v6a2 2 0 01-4 0V5a2 2 0 012-2z"/>
  </svg>`,
  refactor: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
    <path d="M2 2h5v2H4v2h3v2H4v2h3v2H2V2zm7 0h5v12H9V2zm2 2v8h1V4h-1z"/>
  </svg>`,
};

function getIcon(action: string): string {
  return ICONS[action] ?? ICONS.more;
}

// ── Shared bar style ──────────────────────────────────────────────────────────

function applyBarStyle(
  el: HTMLElement,
  rect: DOMRect,
  top: number,
  height: number,
  bg: string,
): void {
  Object.assign(el.style, {
    position: 'fixed',
    left: `${rect.left}px`,
    top: `${top}px`,
    width: `${rect.width}px`,
    height: `${height}px`,
    background: bg,
    backdropFilter: GLASS.backdrop,
    WebkitBackdropFilter: GLASS.backdrop,
    border: `1px solid rgba(255,255,255,${GLASS.borderOpacity})`,
    boxShadow: GLASS.shadowBase,
    borderRadius: '6px',
    display: 'flex',
    alignItems: 'center',
    gap: '2px',
    padding: '0 6px',
    overflow: 'hidden',
    boxSizing: 'border-box',
    zIndex: String(Z_TOP),
    userSelect: 'none',
  });
}

function makeActionButton(action: string): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.className = 'gb-action';
  btn.dataset.action = action;
  btn.title = action;
  btn.innerHTML = getIcon(action);
  Object.assign(btn.style, {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    color: 'rgba(255,255,255,0.85)',
    padding: '3px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '4px',
    flexShrink: '0',
  });
  return btn;
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface AskPayload {
  type: 'text' | 'image';
  text: string;
  imageUrl: string;
  rect: DOMRect;
}

export interface GhostBarAPI {
  showTextBar(selectionRect: DOMRect, text: string): void;
  showImageBar(img: HTMLImageElement): void;
  hideBar(): Promise<void>;
  getBarRect(): DOMRect | null;
  onAskClicked(cb: (content: AskPayload) => void): void;
}

// ── createGhostBar ────────────────────────────────────────────────────────────

export function createGhostBar(): ContentModule & GhostBarAPI {
  let bar: HTMLElement | null = null;
  let barType: 'text' | 'image' | null = null;
  let currentText = '';
  let currentImageUrl = '';
  let currentRect: DOMRect | null = null;
  let containerEl: HTMLElement = document.body;
  let storeRef: ContextStore | null = null;
  let extRef: typeof chrome | null = null;
  let askCallback: ((payload: AskPayload) => void) | null = null;

  // Timers
  let selectionTimer: ReturnType<typeof setTimeout> | null = null;
  let imageHoverTimer: ReturnType<typeof setTimeout> | null = null;
  let dismissTimer: ReturnType<typeof setTimeout> | null = null;
  let currentHoveredImg: HTMLImageElement | null = null;

  // ── Remove existing bar immediately (sync, no animation) ──────────────────
  function removeBarSync(): void {
    if (bar) {
      bar.remove();
      bar = null;
      barType = null;
      currentRect = null;
    }
  }

  // ── hideBar (async, animated) ─────────────────────────────────────────────
  async function hideBar(): Promise<void> {
    if (!bar) return;
    const b = bar;
    const bt = barType;
    bar = null;
    barType = null;
    currentRect = null;
    try {
      const dir = bt === 'image' ? 'down' : 'up';
      await flow(b, { direction: dir, duration: ANIM.morphDuration, easing: ANIM.morphEasing });
    } catch {
      // animation may fail in test env — ignore
    }
    b.remove();
  }

  // ── Action click handler ──────────────────────────────────────────────────
  function handleActionClick(action: string): void {
    if (!extRef) return;

    if (action === 'ask') {
      if (askCallback) {
        const payload: AskPayload = {
          type: barType === 'image' ? 'image' : 'text',
          text: currentText,
          imageUrl: currentImageUrl,
          rect: bar ? bar.getBoundingClientRect() : (currentRect ?? new DOMRect()),
        };
        askCallback(payload);
      }
      return;
    }

    if (action === 'copy') {
      navigator.clipboard?.writeText(currentText).catch(() => {});
      return;
    }

    if (action === 'highlight') {
      extRef.runtime.sendMessage({ type: 'SAVE_KNOWLEDGE', text: currentText, url: location.href, title: document.title });
      return;
    }

    // image-specific
    if (action === 'describe') {
      extRef.runtime.sendMessage({ type: 'IMAGE_DESCRIBE', imageUrl: currentImageUrl });
      return;
    }
    if (action === 'edit') {
      extRef.runtime.sendMessage({ type: 'IMAGE_EDIT_OPEN', imageUrl: currentImageUrl });
      return;
    }
    if (action === 'save') {
      extRef.runtime.sendMessage({ type: 'IMAGE_SAVE', imageUrl: currentImageUrl });
      return;
    }

    // Generic quick action
    const msg: QuickActionMessage = { type: 'QUICK_ACTION', action, text: currentText };
    extRef.runtime.sendMessage(msg);
  }

  // ── Attach click listener to bar ──────────────────────────────────────────
  function attachClickListeners(barEl: HTMLElement): void {
    barEl.addEventListener('click', (e) => {
      const target = (e.target as HTMLElement).closest('.gb-action') as HTMLElement | null;
      if (!target) return;
      const action = target.dataset.action ?? '';

      if (action === 'more') {
        const extended = barEl.querySelector('.gb-extended') as HTMLElement | null;
        if (extended) {
          extended.style.display = extended.style.display === 'none' ? 'flex' : 'none';
        }
        return;
      }

      handleActionClick(action);
    });
  }

  // ── showTextBar ───────────────────────────────────────────────────────────
  function showTextBar(selectionRect: DOMRect, text: string): void {
    if (!storeRef) return;
    const signal = storeRef.get();
    if (signal.suppressGhostBars) return;

    removeBarSync();

    currentText = text;
    currentImageUrl = '';
    currentRect = selectionRect;
    barType = 'text';

    const barEl = document.createElement('div');
    barEl.className = 'ghost-bar ghost-bar-text';

    const top = selectionRect.bottom;
    applyBarStyle(barEl, selectionRect, top, GHOST_BAR.height, GLASS.bg);

    // Determine actions — up to maxActionsPerRow, last slot reserved for 'more'
    const actions = signal.actions.slice(0, GHOST_BAR.maxActionsPerRow - 1);

    for (const action of actions) {
      barEl.appendChild(makeActionButton(action));
    }

    // More button
    const moreBtn = makeActionButton('more');
    barEl.appendChild(moreBtn);

    // Extended row (hidden by default)
    const extended = document.createElement('div');
    extended.className = 'gb-extended';
    const extActions = ['rewrite', 'grammar', 'define', 'read-aloud'];
    for (const action of extActions) {
      extended.appendChild(makeActionButton(action));
    }
    Object.assign(extended.style, {
      display: 'none',
      position: 'absolute',
      top: `${GHOST_BAR.height}px`,
      left: '0',
      right: '0',
      background: GLASS.bg,
      borderRadius: '0 0 6px 6px',
      padding: '2px 6px',
      gap: '2px',
    });
    barEl.style.position = 'fixed';
    barEl.appendChild(extended);

    attachClickListeners(barEl);
    containerEl.appendChild(barEl);
    bar = barEl;

    flow(barEl, {
      direction: 'down',
      duration: ANIM.morphDuration,
      easing: ANIM.morphEasing,
    }).catch(() => {});
  }

  // ── showImageBar ──────────────────────────────────────────────────────────
  function showImageBar(img: HTMLImageElement): void {
    if (!storeRef) return;
    const signal = storeRef.get();
    if (signal.suppressGhostBars) return;

    removeBarSync();

    const imgRect = img.getBoundingClientRect();
    currentText = '';
    currentImageUrl = img.src ?? img.currentSrc ?? '';
    currentRect = imgRect;
    barType = 'image';

    const barEl = document.createElement('div');
    barEl.className = 'ghost-bar ghost-bar-image';

    const top = imgRect.bottom - GHOST_BAR.imageBarHeight;
    applyBarStyle(barEl, imgRect, top, GHOST_BAR.imageBarHeight, GLASS.bgHeavy);

    const imageActions = ['describe', 'edit', 'save', 'ask'];
    for (const action of imageActions) {
      const btn = makeActionButton(action);
      btn.style.width = `${GHOST_BAR.imageIconSize + 8}px`;
      btn.style.height = `${GHOST_BAR.imageIconSize + 8}px`;
      barEl.appendChild(btn);
    }

    attachClickListeners(barEl);
    containerEl.appendChild(barEl);
    bar = barEl;

    flow(barEl, {
      direction: 'down',
      duration: ANIM.morphDuration,
      easing: ANIM.morphEasing,
    }).catch(() => {});
  }

  // ── getBarRect ────────────────────────────────────────────────────────────
  function getBarRect(): DOMRect | null {
    return bar ? bar.getBoundingClientRect() : null;
  }

  // ── onAskClicked ──────────────────────────────────────────────────────────
  function onAskClicked(cb: (content: AskPayload) => void): void {
    askCallback = cb;
  }

  // ── init ──────────────────────────────────────────────────────────────────
  function init(container: HTMLElement, store: ContextStore, ext: typeof chrome): void {
    containerEl = container;
    storeRef = store;
    extRef = ext;

    // ── Selection tracking (debounced) ────────────────────────────────────
    const onSelectionChange = () => {
      if (selectionTimer) clearTimeout(selectionTimer);
      selectionTimer = setTimeout(() => {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0 || sel.toString().trim().length === 0) {
          hideBar().catch(() => {});
          return;
        }
        const range = sel.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) return;
        showTextBar(rect, sel.toString());
      }, ANIM.selectionDelay);
    };
    document.addEventListener('selectionchange', onSelectionChange);

    // ── Image hover tracking ──────────────────────────────────────────────
    const onMouseOver = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName !== 'IMG') return;
      const img = target as HTMLImageElement;
      const rect = img.getBoundingClientRect();
      if (rect.width < 80 || rect.height < 80) return;

      if (dismissTimer) {
        clearTimeout(dismissTimer);
        dismissTimer = null;
      }

      if (imageHoverTimer) clearTimeout(imageHoverTimer);
      currentHoveredImg = img;
      imageHoverTimer = setTimeout(() => {
        if (currentHoveredImg === img) {
          showImageBar(img);
        }
      }, ANIM.imageHoverDelay);
    };

    const onMouseOut = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const related = e.relatedTarget as HTMLElement | null;

      const leavingImg = target.tagName === 'IMG';
      const leavingBar = bar && (target === bar || bar.contains(target));

      if (!leavingImg && !leavingBar) return;

      // If moving into bar or image — cancel dismiss
      if (related && bar && (related === bar || bar.contains(related))) return;
      if (related && related.tagName === 'IMG' && related === currentHoveredImg) return;

      if (imageHoverTimer) {
        clearTimeout(imageHoverTimer);
        imageHoverTimer = null;
      }

      if (barType === 'image') {
        if (dismissTimer) clearTimeout(dismissTimer);
        dismissTimer = setTimeout(() => {
          hideBar().catch(() => {});
        }, ANIM.dismissDelay);
      }
    };

    document.addEventListener('mouseover', onMouseOver, true);
    document.addEventListener('mouseout', onMouseOut, true);

    // ── Scroll tracking: reposition or hide ──────────────────────────────
    const onScroll = () => {
      if (!bar || !currentRect) return;
      const viewH = window.innerHeight;
      const viewW = window.innerWidth;

      if (barType === 'text') {
        const sel = window.getSelection();
        if (!sel || sel.rangeCount === 0) {
          hideBar().catch(() => {});
          return;
        }
        const rect = sel.getRangeAt(0).getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > viewH || rect.right < 0 || rect.left > viewW) {
          hideBar().catch(() => {});
          return;
        }
        bar.style.top = `${rect.bottom}px`;
        bar.style.left = `${rect.left}px`;
        bar.style.width = `${rect.width}px`;
        currentRect = rect;
      } else if (barType === 'image' && currentHoveredImg) {
        const rect = currentHoveredImg.getBoundingClientRect();
        if (rect.bottom < 0 || rect.top > viewH || rect.right < 0 || rect.left > viewW) {
          hideBar().catch(() => {});
          return;
        }
        const top = rect.bottom - GHOST_BAR.imageBarHeight;
        bar.style.top = `${top}px`;
        bar.style.left = `${rect.left}px`;
        bar.style.width = `${rect.width}px`;
        currentRect = rect;
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  // ── destroy ───────────────────────────────────────────────────────────────
  function destroy(): void {
    if (selectionTimer) clearTimeout(selectionTimer);
    if (imageHoverTimer) clearTimeout(imageHoverTimer);
    if (dismissTimer) clearTimeout(dismissTimer);
    removeBarSync();
  }

  return {
    init,
    destroy,
    showTextBar,
    showImageBar,
    hideBar,
    getBarRect,
    onAskClicked,
  };
}
