import type { ContextSignal, ContextStore, ContextListener, PageContext, Cadence } from './types';
import { PALETTES, CONTEXT_ACTIONS } from './tokens';

// ── CadenceTracker ────────────────────────────────────────────────────────────

export interface CadenceTracker {
  getCadence(): Cadence;
  recordScroll(velocity: number): void;
  recordSelection(): void;
  recordInput(): void;
}

// ── SessionMemory ─────────────────────────────────────────────────────────────

export interface SessionMemory {
  recordAction(action: string, textHash: string): void;
  recordDismissal(): void;
  getExtraDelay(): number;
  shouldPromoteContinue(textHash: string): boolean;
  getSessionActions(): string[];
  recordHighlight(): void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeInitialSignal(): ContextSignal {
  const palette = PALETTES.general;
  return {
    type: 'general',
    cadence: 'engaged',
    suppressGhostBars: false,
    readingProgress: 0,
    actions: CONTEXT_ACTIONS.general,
    accent: palette.accent,
    glow: palette.glow,
    icon: palette.icon,
    sessionActions: [],
  };
}

// ── createContextStore ────────────────────────────────────────────────────────

export function createContextStore(): ContextStore {
  let signal: ContextSignal = makeInitialSignal();
  const listeners = new Set<ContextListener>();

  return {
    get() {
      return signal;
    },
    subscribe(fn: ContextListener): () => void {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    update(partial: Partial<ContextSignal>): void {
      signal = { ...signal, ...partial };
      for (const fn of listeners) fn(signal);
    },
  };
}

// ── detectPageType ────────────────────────────────────────────────────────────

export function detectPageType(url: string, doc: Document): PageContext {
  // Layer 1: URL pattern matching (highest priority)
  let hostname = '';
  try {
    hostname = new URL(url).hostname.replace(/^www\./, '');
  } catch {
    // malformed URL — fall through to DOM
  }

  if (hostname === 'github.com' || hostname === 'gitlab.com') return 'code';
  if (hostname === 'youtube.com' || hostname === 'netflix.com') return 'media';
  if (hostname === 'mail.google.com' || hostname === 'outlook.live.com') return 'email';

  const isShopping =
    hostname === 'amazon.com' ||
    hostname === 'ebay.com' ||
    hostname === 'etsy.com' ||
    url.includes('/product/') ||
    url.includes('/cart/');
  if (isShopping) return 'shopping';

  // Layer 2: DOM analysis (fallback)

  // article via element or role
  const articleEl = doc.querySelector('article, [role="article"]');
  if (articleEl) return 'article';

  // code via <pre>/<code>
  const preCount = doc.querySelectorAll('pre, code').length;
  if (preCount >= 3) return 'code';

  // media via <video>/<audio>
  if (doc.querySelector('video, audio')) return 'media';

  // shopping via ld+json
  const ldScripts = doc.querySelectorAll('script[type="application/ld+json"]');
  for (const s of ldScripts) {
    try {
      const data = JSON.parse(s.textContent ?? '');
      const type = (Array.isArray(data) ? data[0] : data)?.['@type'];
      if (typeof type === 'string' && type.toLowerCase().includes('product')) return 'shopping';
    } catch {
      // ignore parse errors
    }
  }

  return 'general';
}

// ── createCadenceTracker ──────────────────────────────────────────────────────

export function createCadenceTracker(): CadenceTracker {
  // Rolling windows: store timestamps + velocity for scroll, timestamps for selection/input
  const scrollWindow: Array<{ ts: number; velocity: number }> = [];
  const selectionWindow: number[] = [];

  const SCROLL_WINDOW_MS = 10_000;
  const SELECTION_WINDOW_MS = 30_000;
  const FAST_SCROLL_THRESHOLD = 300; // px/event — above this counts as "fast"
  const FAST_SCROLL_RATIO = 0.6;     // 60%+ fast events → passive
  const ACTIVE_SELECTION_COUNT = 3;  // 3+ selections in window → active
  const TRANSITION_COOLDOWN_MS = 3_000;

  let currentCadence: Cadence = 'engaged';
  let lastTransition = 0;

  function prune(now: number) {
    const scrollCutoff = now - SCROLL_WINDOW_MS;
    const selCutoff = now - SELECTION_WINDOW_MS;
    while (scrollWindow.length && scrollWindow[0].ts < scrollCutoff) scrollWindow.shift();
    while (selectionWindow.length && selectionWindow[0] < selCutoff) selectionWindow.shift();
  }

  function computeCadence(now: number): Cadence {
    // Selection/input activity wins over scrolling
    if (selectionWindow.length >= ACTIVE_SELECTION_COUNT) return 'active';

    if (scrollWindow.length >= 3) {
      const fastCount = scrollWindow.filter(e => e.velocity >= FAST_SCROLL_THRESHOLD).length;
      if (fastCount / scrollWindow.length >= FAST_SCROLL_RATIO) return 'passive';
    }

    return 'engaged';
  }

  function maybeTransition(now: number) {
    if (now - lastTransition < TRANSITION_COOLDOWN_MS) return;
    const next = computeCadence(now);
    if (next !== currentCadence) {
      currentCadence = next;
      lastTransition = now;
    }
  }

  return {
    getCadence() {
      return currentCadence;
    },
    recordScroll(velocity: number) {
      const now = Date.now();
      scrollWindow.push({ ts: now, velocity: Math.abs(velocity) });
      prune(now);
      maybeTransition(now);
    },
    recordSelection() {
      const now = Date.now();
      selectionWindow.push(now);
      prune(now);
      // Force immediate transition for active (no cooldown needed for engagement)
      const next = computeCadence(now);
      if (next === 'active') {
        currentCadence = 'active';
        lastTransition = now;
      } else {
        maybeTransition(now);
      }
    },
    recordInput() {
      // Treat input same as selection for cadence purposes
      const now = Date.now();
      selectionWindow.push(now);
      prune(now);
      const next = computeCadence(now);
      if (next === 'active') {
        currentCadence = 'active';
        lastTransition = now;
      } else {
        maybeTransition(now);
      }
    },
  };
}

// ── createSessionMemory ───────────────────────────────────────────────────────

export function createSessionMemory(): SessionMemory {
  let dismissals = 0;
  const actedHashes = new Set<string>();
  const actions: string[] = [];
  let highlights = 0;

  return {
    recordAction(action: string, textHash: string) {
      actedHashes.add(textHash);
      if (!actions.includes(action)) actions.push(action);
    },
    recordDismissal() {
      dismissals++;
    },
    getExtraDelay() {
      return Math.min(dismissals * 200, 2000);
    },
    shouldPromoteContinue(textHash: string) {
      return actedHashes.has(textHash);
    },
    getSessionActions() {
      const extra: string[] = [];
      if (highlights >= 2) extra.push('review-highlights');
      return extra;
    },
    recordHighlight() {
      highlights++;
    },
  };
}

// ── initContextEngine ─────────────────────────────────────────────────────────

export function initContextEngine(store: ContextStore, ext: typeof chrome): () => void {
  const cleanups: Array<() => void> = [];
  const cadence = createCadenceTracker();
  const session = createSessionMemory();

  // ── Detect current page and apply context ──
  function applyPageContext() {
    const url = location.href;
    const pageType = detectPageType(url, document);
    const palette = PALETTES[pageType];
    store.update({
      type: pageType,
      accent: palette.accent,
      glow: palette.glow,
      icon: palette.icon,
      actions: [...CONTEXT_ACTIONS[pageType], ...session.getSessionActions()],
      sessionActions: session.getSessionActions(),
    });
  }

  // Defer DOM analysis until idle
  const scheduleIdle = (fn: () => void) => {
    if (typeof requestIdleCallback !== 'undefined') {
      const id = requestIdleCallback(fn, { timeout: 2000 });
      cleanups.push(() => cancelIdleCallback(id));
    } else {
      const id = setTimeout(fn, 200);
      cleanups.push(() => clearTimeout(id));
    }
  };

  scheduleIdle(applyPageContext);

  // ── SPA navigation ──
  const onPopState = () => scheduleIdle(applyPageContext);
  window.addEventListener('popstate', onPopState);
  cleanups.push(() => window.removeEventListener('popstate', onPopState));

  // ── MutationObserver (debounced 2s) for content changes ──
  let mutationTimer: ReturnType<typeof setTimeout> | null = null;
  const observer = new MutationObserver(() => {
    if (mutationTimer) clearTimeout(mutationTimer);
    mutationTimer = setTimeout(() => scheduleIdle(applyPageContext), 2000);
  });
  observer.observe(document.body, { childList: true, subtree: false });
  cleanups.push(() => {
    observer.disconnect();
    if (mutationTimer) clearTimeout(mutationTimer);
  });

  // ── Scroll → cadence ──
  let lastScrollY = window.scrollY;
  let lastScrollTime = Date.now();
  const onScroll = () => {
    const now = Date.now();
    const dt = Math.max(now - lastScrollTime, 1);
    const dy = Math.abs(window.scrollY - lastScrollY);
    const velocity = (dy / dt) * 1000; // px/s
    cadence.recordScroll(velocity);
    lastScrollY = window.scrollY;
    lastScrollTime = now;
    store.update({ cadence: cadence.getCadence() });
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  cleanups.push(() => window.removeEventListener('scroll', onScroll));

  // ── Input focus/blur → suppressGhostBars ──
  const onFocusIn = (e: FocusEvent) => {
    const target = e.target as HTMLElement;
    if (target.matches('input, textarea, [contenteditable]')) {
      cadence.recordInput();
      store.update({ suppressGhostBars: true, cadence: cadence.getCadence() });
    }
  };
  const onFocusOut = (e: FocusEvent) => {
    const target = e.target as HTMLElement;
    if (target.matches('input, textarea, [contenteditable]')) {
      store.update({ suppressGhostBars: false });
    }
  };
  document.addEventListener('focusin', onFocusIn);
  document.addEventListener('focusout', onFocusOut);
  cleanups.push(() => {
    document.removeEventListener('focusin', onFocusIn);
    document.removeEventListener('focusout', onFocusOut);
  });

  // ── Selection → cadence ──
  const onSelectionChange = () => {
    const sel = window.getSelection();
    if (sel && sel.toString().length > 0) {
      cadence.recordSelection();
      store.update({ cadence: cadence.getCadence() });
    }
  };
  document.addEventListener('selectionchange', onSelectionChange);
  cleanups.push(() => document.removeEventListener('selectionchange', onSelectionChange));

  // ── IntersectionObserver → readingProgress ──
  const contentEl =
    document.querySelector('article') ??
    document.querySelector('main') ??
    document.querySelector('[role="main"]');

  if (contentEl) {
    let topY = 0;
    let bottomY = 0;

    const updateBounds = () => {
      const rect = contentEl.getBoundingClientRect();
      topY = rect.top + window.scrollY;
      bottomY = rect.bottom + window.scrollY;
    };
    scheduleIdle(updateBounds);

    const onProgressScroll = () => {
      const viewBottom = window.scrollY + window.innerHeight;
      const total = bottomY - topY;
      if (total <= 0) return;
      const progress = Math.min(Math.max((viewBottom - topY) / total, 0), 1);
      store.update({ readingProgress: progress });
    };
    window.addEventListener('scroll', onProgressScroll, { passive: true });
    cleanups.push(() => window.removeEventListener('scroll', onProgressScroll));
  }

  // ── Layer 5: chrome.storage.session cross-tab state ──
  try {
    const sessionStorage = (ext.storage as any).session as chrome.storage.StorageArea | undefined;
    if (sessionStorage) {
      sessionStorage.get(['contextType'], (result) => {
        if (result?.contextType) {
          // Seed initial type from session if available
          const saved = result.contextType as PageContext;
          const palette = PALETTES[saved];
          store.update({
            type: saved,
            accent: palette.accent,
            glow: palette.glow,
            icon: palette.icon,
            actions: CONTEXT_ACTIONS[saved],
          });
        }
      });

      // Persist type changes across tabs
      const unsub = store.subscribe((sig) => {
        sessionStorage.set({ contextType: sig.type });
      });
      cleanups.push(unsub);
    }
  } catch {
    // chrome.storage.session not available in all contexts — ignore
  }

  return () => {
    for (const fn of cleanups) fn();
  };
}
