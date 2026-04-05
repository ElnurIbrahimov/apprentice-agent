(function() {
  "use strict";
  const PALETTES = {
    general: {
      accent: "#7c3aed",
      glow: "rgba(124, 58, 237, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 1L10 6H15L11 9.5L12.5 14.5L8 11.5L3.5 14.5L5 9.5L1 6H6L8 1Z"/>
    </svg>`
    },
    article: {
      accent: "#3b82f6",
      glow: "rgba(59, 130, 246, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`
    },
    media: {
      accent: "#f59e0b",
      glow: "rgba(245, 158, 11, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 4h1v8H2V4zm2 2h1v4H4V6zm2-1h1v6H6V5zm2 2h1v2H8V7zm2-2h1v6h-1V5zm2 1h1v4h-1V6z"/>
    </svg>`
    },
    code: {
      accent: "#10b981",
      glow: "rgba(16, 185, 129, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M5.5 3.5L1 8l4.5 4.5 1-1L3 8l3.5-3.5-1-1zm5 0l-1 1L13 8l-3.5 3.5 1 1L15 8l-4.5-4.5z"/>
    </svg>`
    },
    email: {
      accent: "#6366f1",
      glow: "rgba(99, 102, 241, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 9.5c-.3 0-.6-.1-.8-.3L2 4.5V12h12V4.5l-5.2 4.7c-.2.2-.5.3-.8.3zM2 3h12l-6 5.4L2 3z"/>
    </svg>`
    },
    shopping: {
      accent: "#ec4899",
      glow: "rgba(236, 72, 153, 0.35)",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6 1l-1.5 4H1l3 2.5-1 4L6 9l2 1.5L10 9l3 3-1-4L15 5h-3.5L10 1H6zm0 1.5h4l1 2.5h2l-2 1.5.7 2.8L10 8l-2 1.5L6 8 4.3 9.3 5 6.5 3 5h2L6 2.5z"/>
    </svg>`
    }
  };
  const CONTEXT_ACTIONS = {
    general: ["ask", "summarize", "explain", "translate", "save", "copy"],
    article: ["ask", "summarize", "highlight", "translate", "save", "explain", "copy"],
    media: ["ask", "describe", "transcript", "summarize", "translate", "save"],
    code: ["ask", "explain", "review", "debug", "refactor", "copy", "save"],
    email: ["ask", "summarize", "reply", "translate", "save", "action-items"],
    shopping: ["ask", "compare", "summarize", "pros-cons", "save", "price-history"]
  };
  const ANIM = {
    morphDuration: 350,
    morphEasing: "cubic-bezier(0.4, 0, 0.0, 1)",
    flowDuration: 500,
    glowPulse: 3e3,
    sequentialStagger: 40,
    dismissDelay: 400,
    crossFadeDuration: 400,
    selectionDelay: 300,
    imageHoverDelay: 800
  };
  const GLASS = {
    bg: "rgba(10, 8, 24, 0.88)",
    bgHeavy: "rgba(10, 8, 24, 0.75)",
    backdrop: "blur(20px) saturate(1.5)",
    borderOpacity: 0.25,
    shadowBase: "0 8px 32px rgba(0,0,0,0.4)"
  };
  const GHOST_BAR = {
    height: 28,
    iconSize: 15,
    imageIconSize: 16,
    imageBarHeight: 32,
    maxActionsPerRow: 7
  };
  const MODAL = {
    maxWidth: 520,
    maxHeight: 480,
    previewMaxLines: 6,
    previewMaxChars: 2e3,
    imagePreviewMaxHeight: 200
  };
  const FAB = {
    pillPadding: "6px 10px",
    glowIntensityMin: 0.15,
    glowIntensityMax: 0.35,
    logoSize: 20,
    expandDuration: 220,
    dragThreshold: 4,
    edgeMargin: 12
  };
  const FONT_STACK = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  const Z_TOP = 2147483647;
  const Z_MID = 2147483645;
  function makeInitialSignal() {
    const palette = PALETTES.general;
    return {
      type: "general",
      cadence: "engaged",
      suppressGhostBars: false,
      readingProgress: 0,
      actions: CONTEXT_ACTIONS.general,
      accent: palette.accent,
      glow: palette.glow,
      icon: palette.icon,
      sessionActions: []
    };
  }
  function createContextStore() {
    let signal = makeInitialSignal();
    const listeners = /* @__PURE__ */ new Set();
    return {
      get() {
        return signal;
      },
      subscribe(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
      },
      update(partial) {
        signal = { ...signal, ...partial };
        for (const fn of listeners) fn(signal);
      }
    };
  }
  function detectPageType(url, doc) {
    var _a;
    let hostname = "";
    try {
      hostname = new URL(url).hostname.replace(/^www\./, "");
    } catch {
    }
    if (hostname === "github.com" || hostname === "gitlab.com") return "code";
    if (hostname === "youtube.com" || hostname === "netflix.com") return "media";
    if (hostname === "mail.google.com" || hostname === "outlook.live.com") return "email";
    const isShopping = hostname === "amazon.com" || hostname === "ebay.com" || hostname === "etsy.com" || url.includes("/product/") || url.includes("/cart/");
    if (isShopping) return "shopping";
    const articleEl = doc.querySelector('article, [role="article"]');
    if (articleEl) return "article";
    const preCount = doc.querySelectorAll("pre, code").length;
    if (preCount >= 3) return "code";
    if (doc.querySelector("video, audio")) return "media";
    const ldScripts = doc.querySelectorAll('script[type="application/ld+json"]');
    for (const s of ldScripts) {
      try {
        const data = JSON.parse(s.textContent ?? "");
        const type = (_a = Array.isArray(data) ? data[0] : data) == null ? void 0 : _a["@type"];
        if (typeof type === "string" && type.toLowerCase().includes("product")) return "shopping";
      } catch {
      }
    }
    return "general";
  }
  function createCadenceTracker() {
    const scrollWindow = [];
    const selectionWindow = [];
    const SCROLL_WINDOW_MS = 1e4;
    const SELECTION_WINDOW_MS = 3e4;
    const FAST_SCROLL_THRESHOLD = 300;
    const FAST_SCROLL_RATIO = 0.6;
    const ACTIVE_SELECTION_COUNT = 3;
    const TRANSITION_COOLDOWN_MS = 3e3;
    let currentCadence = "engaged";
    let lastTransition = 0;
    function prune(now) {
      const scrollCutoff = now - SCROLL_WINDOW_MS;
      const selCutoff = now - SELECTION_WINDOW_MS;
      while (scrollWindow.length && scrollWindow[0].ts < scrollCutoff) scrollWindow.shift();
      while (selectionWindow.length && selectionWindow[0] < selCutoff) selectionWindow.shift();
    }
    function computeCadence(now) {
      if (selectionWindow.length >= ACTIVE_SELECTION_COUNT) return "active";
      if (scrollWindow.length >= 3) {
        const fastCount = scrollWindow.filter((e) => e.velocity >= FAST_SCROLL_THRESHOLD).length;
        if (fastCount / scrollWindow.length >= FAST_SCROLL_RATIO) return "passive";
      }
      return "engaged";
    }
    function maybeTransition(now) {
      if (now - lastTransition < TRANSITION_COOLDOWN_MS) return;
      const next = computeCadence();
      if (next !== currentCadence) {
        currentCadence = next;
        lastTransition = now;
      }
    }
    return {
      getCadence() {
        return currentCadence;
      },
      recordScroll(velocity) {
        const now = Date.now();
        scrollWindow.push({ ts: now, velocity: Math.abs(velocity) });
        prune(now);
        maybeTransition(now);
      },
      recordSelection() {
        const now = Date.now();
        selectionWindow.push(now);
        prune(now);
        const next = computeCadence();
        if (next === "active") {
          currentCadence = "active";
          lastTransition = now;
        } else {
          maybeTransition(now);
        }
      },
      recordInput() {
        const now = Date.now();
        selectionWindow.push(now);
        prune(now);
        const next = computeCadence();
        if (next === "active") {
          currentCadence = "active";
          lastTransition = now;
        } else {
          maybeTransition(now);
        }
      }
    };
  }
  function createSessionMemory() {
    let dismissals = 0;
    const actedHashes = /* @__PURE__ */ new Set();
    let highlights = 0;
    return {
      recordAction(action, textHash) {
        actedHashes.add(textHash);
      },
      recordDismissal() {
        dismissals++;
      },
      getExtraDelay() {
        return Math.min(dismissals * 200, 2e3);
      },
      shouldPromoteContinue(textHash) {
        return actedHashes.has(textHash);
      },
      getSessionActions() {
        const extra = [];
        if (highlights >= 2) extra.push("review-highlights");
        return extra;
      },
      recordHighlight() {
        highlights++;
      }
    };
  }
  function initContextEngine(store, ext2) {
    const cleanups = [];
    const cadence = createCadenceTracker();
    const session = createSessionMemory();
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
        sessionActions: session.getSessionActions()
      });
    }
    const scheduleIdle = (fn) => {
      if (typeof requestIdleCallback !== "undefined") {
        const id = requestIdleCallback(fn, { timeout: 2e3 });
        cleanups.push(() => cancelIdleCallback(id));
      } else {
        const id = setTimeout(fn, 200);
        cleanups.push(() => clearTimeout(id));
      }
    };
    scheduleIdle(applyPageContext);
    const onNavChange = () => scheduleIdle(applyPageContext);
    window.addEventListener("popstate", onNavChange);
    cleanups.push(() => window.removeEventListener("popstate", onNavChange));
    const origPushState = history.pushState.bind(history);
    const origReplaceState = history.replaceState.bind(history);
    history.pushState = function(...args) {
      origPushState(...args);
      onNavChange();
    };
    history.replaceState = function(...args) {
      origReplaceState(...args);
      onNavChange();
    };
    cleanups.push(() => {
      history.pushState = origPushState;
      history.replaceState = origReplaceState;
    });
    let mutationTimer = null;
    const observer = new MutationObserver(() => {
      if (mutationTimer) clearTimeout(mutationTimer);
      mutationTimer = setTimeout(() => scheduleIdle(applyPageContext), 2e3);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    cleanups.push(() => {
      observer.disconnect();
      if (mutationTimer) clearTimeout(mutationTimer);
    });
    let lastScrollY = window.scrollY;
    let lastScrollTime = Date.now();
    const onScroll = () => {
      const now = Date.now();
      const dt = Math.max(now - lastScrollTime, 1);
      const dy = Math.abs(window.scrollY - lastScrollY);
      const velocity = dy / dt * 1e3;
      cadence.recordScroll(velocity);
      lastScrollY = window.scrollY;
      lastScrollTime = now;
      store.update({ cadence: cadence.getCadence() });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    cleanups.push(() => window.removeEventListener("scroll", onScroll));
    const onFocusIn = (e) => {
      const target = e.target;
      if (target.matches("input, textarea, [contenteditable]")) {
        cadence.recordInput();
        store.update({ suppressGhostBars: true, cadence: cadence.getCadence() });
      }
    };
    const onFocusOut = (e) => {
      const target = e.target;
      if (target.matches("input, textarea, [contenteditable]")) {
        store.update({ suppressGhostBars: false });
      }
    };
    document.addEventListener("focusin", onFocusIn);
    document.addEventListener("focusout", onFocusOut);
    cleanups.push(() => {
      document.removeEventListener("focusin", onFocusIn);
      document.removeEventListener("focusout", onFocusOut);
    });
    const onSelectionChange = () => {
      const sel = window.getSelection();
      if (sel && sel.toString().length > 0) {
        cadence.recordSelection();
        store.update({ cadence: cadence.getCadence() });
      }
    };
    document.addEventListener("selectionchange", onSelectionChange);
    cleanups.push(() => document.removeEventListener("selectionchange", onSelectionChange));
    const contentEl = document.querySelector("article") ?? document.querySelector("main") ?? document.querySelector('[role="main"]');
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
      window.addEventListener("scroll", onProgressScroll, { passive: true });
      cleanups.push(() => window.removeEventListener("scroll", onProgressScroll));
    }
    try {
      const sessionStorage = ext2.storage.session;
      if (sessionStorage) {
        sessionStorage.get(["contextType"], (result) => {
          if ((result == null ? void 0 : result.contextType) && store.get().type === "general") {
            const saved = result.contextType;
            const palette = PALETTES[saved];
            store.update({
              type: saved,
              accent: palette.accent,
              glow: palette.glow,
              icon: palette.icon,
              actions: CONTEXT_ACTIONS[saved]
            });
          }
        });
        let lastWrittenType = null;
        const unsub = store.subscribe((sig) => {
          if (sig.type !== lastWrittenType) {
            lastWrittenType = sig.type;
            sessionStorage.set({ contextType: sig.type });
          }
        });
        cleanups.push(unsub);
      }
    } catch {
    }
    return () => {
      for (const fn of cleanups) fn();
    };
  }
  function buildStylesheet() {
    return `
/* ── Host / Root ─────────────────────────────────────────────────────────── */
:host {
  --aura-accent: #7c3aed;
  --aura-glow: rgba(124, 58, 237, 0.3);
  all: initial;
  pointer-events: none;
  font-family: ${FONT_STACK};
}

/* ── FAB ──────────────────────────────────────────────────────────────────── */
.aura-fab {
  position: fixed;
  right: 0;
  bottom: 30px;
  z-index: ${Z_TOP};
  transform: translateX(100%);
  transition: transform ${ANIM.morphDuration}ms ${ANIM.morphEasing};
  pointer-events: none;
  will-change: transform, opacity;
}

.aura-fab.show {
  transform: translateX(0);
}

.aura-fab.left {
  left: 0;
  right: auto;
  transform: translateX(-100%);
}

.aura-fab.left.show {
  transform: translateX(0);
}

.fab-pill {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: ${FAB.pillPadding};
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 20px;
  box-shadow: ${GLASS.shadowBase};
  cursor: pointer;
  pointer-events: auto;
  transition: padding ${FAB.expandDuration}ms ${ANIM.morphEasing},
              border-radius ${FAB.expandDuration}ms ${ANIM.morphEasing};
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
  user-select: none;
}

.fab-pill.hover {
  padding: ${FAB.pillPadding};
}

.fab-pill.dragging {
  border-radius: 50%;
  cursor: move;
  padding: ${FAB.pillPadding};
}

.fab-glow {
  position: absolute;
  inset: -6px;
  border-radius: inherit;
  background: var(--aura-glow);
  filter: blur(10px);
  animation: aura-glow-pulse ${ANIM.glowPulse}ms ease-in-out infinite;
  pointer-events: none;
  z-index: -1;
  will-change: transform, opacity;
}

.fab-logo {
  width: ${FAB.logoSize}px;
  height: ${FAB.logoSize}px;
  pointer-events: none;
  flex-shrink: 0;
}

.fab-close {
  position: absolute;
  bottom: -18px;
  left: 50%;
  transform: translateX(-50%);
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.2s ease;
  pointer-events: auto;
  font-size: 9px;
  color: rgba(255, 255, 255, 0.7);
}

.fab-pill:hover .fab-close,
.aura-fab:hover .fab-close {
  opacity: 1;
}

.fab-popout {
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  min-width: 160px;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 12px;
  box-shadow: ${GLASS.shadowBase};
  padding: 6px;
  opacity: 0;
  pointer-events: none;
  transform: translateY(4px);
  transition: opacity 0.2s ease, transform 0.2s ease;
  will-change: transform, opacity;
}

.fab-popout.visible {
  opacity: 1;
  pointer-events: auto;
  transform: translateY(0);
}

.aura-fab.left .fab-popout {
  right: auto;
  left: 0;
}

.fab-action-btn {
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: transparent;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.8);
  transition: background 0.15s ease, color 0.15s ease;
  pointer-events: auto;
  font-family: ${FONT_STACK};
}

.fab-action-btn:hover {
  background: var(--aura-accent);
  color: #fff;
}

/* ── Ghost Bar ────────────────────────────────────────────────────────────── */
.ghost-bar {
  position: fixed;
  z-index: ${Z_MID};
  display: flex;
  align-items: center;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  box-shadow: ${GLASS.shadowBase};
  pointer-events: auto;
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
}

.ghost-bar-text {
  height: ${GHOST_BAR.height}px;
  padding: 0 8px;
  border-radius: 0 0 8px 8px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.ghost-bar-image {
  height: ${GHOST_BAR.imageBarHeight}px;
  padding: 0 10px;
  background: ${GLASS.bgHeavy};
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.gb-action {
  width: ${GHOST_BAR.iconSize}px;
  height: ${GHOST_BAR.iconSize}px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: rgba(255, 255, 255, 0.75);
  border-radius: 4px;
  transition: color 0.15s ease, background 0.15s ease;
  pointer-events: auto;
  background: transparent;
  border: none;
  padding: 0;
}

.gb-action:hover {
  color: var(--aura-accent);
  background: rgba(255, 255, 255, 0.08);
}

.gb-separator {
  width: 1px;
  height: 12px;
  background: rgba(255, 255, 255, 0.2);
  flex-shrink: 0;
  margin: 0 2px;
}

.gb-extended {
  display: none;
  align-items: center;
  gap: 2px;
  padding-top: 4px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  margin-top: 4px;
}

.gb-extended.visible {
  display: flex;
}

/* ── Modal ────────────────────────────────────────────────────────────────── */
.aura-modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.3);
  z-index: ${Z_MID};
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: auto;
}

.aura-modal {
  max-width: ${MODAL.maxWidth}px;
  max-height: ${MODAL.maxHeight}px;
  width: 90vw;
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 16px;
  box-shadow: ${GLASS.shadowBase};
  overflow: hidden;
  display: flex;
  flex-direction: column;
  will-change: transform, opacity;
  font-family: ${FONT_STACK};
  color: #fff;
}

.modal-preview {
  padding: 12px 16px;
  max-height: calc(${MODAL.previewMaxLines} * 1.5em + 24px);
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.75);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-input {
  width: 100%;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 10px 12px;
  color: #fff;
  font-family: ${FONT_STACK};
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s ease;
  box-sizing: border-box;
}

.modal-input:focus {
  border-color: var(--aura-accent);
}

.modal-action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  background: transparent;
  color: rgba(255, 255, 255, 0.85);
  font-family: ${FONT_STACK};
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.modal-action-btn:hover {
  background: var(--aura-accent);
  border-color: var(--aura-accent);
  color: #fff;
}

.modal-model-select {
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 8px;
  padding: 6px 10px;
  color: #fff;
  font-family: ${FONT_STACK};
  font-size: 13px;
  outline: none;
  cursor: pointer;
  transition: border-color 0.2s ease;
}

.modal-model-select:focus {
  border-color: var(--aura-accent);
}

/* ── Highlights ───────────────────────────────────────────────────────────── */
.hl-tooltip {
  position: fixed;
  z-index: ${Z_TOP};
  background: ${GLASS.bg};
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 8px;
  box-shadow: ${GLASS.shadowBase};
  padding: 6px 10px;
  font-family: ${FONT_STACK};
  font-size: 12px;
  color: rgba(255, 255, 255, 0.9);
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Toast ────────────────────────────────────────────────────────────────── */
.aura-toast {
  position: fixed;
  bottom: 80px;
  left: 50%;
  transform: translateX(-50%);
  z-index: ${Z_TOP};
  background: rgba(16, 185, 129, 0.9);
  backdrop-filter: ${GLASS.backdrop};
  -webkit-backdrop-filter: ${GLASS.backdrop};
  border: 1px solid rgba(255, 255, 255, ${GLASS.borderOpacity});
  border-radius: 20px;
  box-shadow: ${GLASS.shadowBase};
  padding: 8px 18px;
  font-family: ${FONT_STACK};
  font-size: 13px;
  color: #fff;
  pointer-events: none;
  will-change: transform, opacity;
}

/* ── Keyframes ────────────────────────────────────────────────────────────── */
@keyframes aura-glow-pulse {
  0%   { opacity: ${FAB.glowIntensityMin}; }
  50%  { opacity: ${FAB.glowIntensityMax}; }
  100% { opacity: ${FAB.glowIntensityMin}; }
}
`.trim();
  }
  const FILL = "forwards";
  function commitAndCancel(anim) {
    try {
      anim.commitStyles();
    } catch {
    }
    anim.cancel();
  }
  async function flow(el, opts) {
    const h = el.offsetHeight || 0;
    const appear = [
      { height: "0px", opacity: 0 },
      { height: `${h}px`, opacity: 1 }
    ];
    const keyframes = opts.direction === "down" ? appear : [...appear].reverse();
    const anim = el.animate(keyframes, {
      duration: opts.duration,
      easing: opts.easing,
      delay: opts.delay ?? 0,
      fill: FILL
    });
    await anim.finished;
    commitAndCancel(anim);
  }
  async function dissolve(el, opts) {
    const anim = el.animate(
      [{ opacity: 1 }, { opacity: 0 }],
      { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL }
    );
    await anim.finished;
    commitAndCancel(anim);
  }
  async function fadeIn(el, opts) {
    const anim = el.animate(
      [{ opacity: 0 }, { opacity: 1 }],
      { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL }
    );
    await anim.finished;
    commitAndCancel(anim);
  }
  async function crossFade(oldEl, newEl, opts) {
    const timing = { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL };
    const outAnim = oldEl.animate([{ opacity: 1 }, { opacity: 0 }], timing);
    const inAnim = newEl.animate([{ opacity: 0 }, { opacity: 1 }], timing);
    await Promise.all([outAnim.finished, inAnim.finished]);
    commitAndCancel(outAnim);
    commitAndCancel(inAnim);
  }
  async function morph(el, from, to, opts) {
    const anim = el.animate(
      [
        {
          width: `${from.width}px`,
          height: `${from.height}px`,
          transform: `translate(${from.left}px, ${from.top}px)`
        },
        {
          width: `${to.width}px`,
          height: `${to.height}px`,
          transform: `translate(${to.left}px, ${to.top}px)`
        }
      ],
      { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL }
    );
    await anim.finished;
    commitAndCancel(anim);
  }
  const DEFAULT_ACTIONS = [
    {
      action: "chat",
      tip: "Chat",
      svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 2h12a1 1 0 011 1v8a1 1 0 01-1 1H5l-3 3V3a1 1 0 011-1z"/>
    </svg>`
    },
    {
      action: "search",
      tip: "Search",
      svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6.5 1a5.5 5.5 0 014.23 9.02l3.12 3.12-1.06 1.06-3.12-3.12A5.5 5.5 0 116.5 1zm0 1.5a4 4 0 100 8 4 4 0 000-8z"/>
    </svg>`
    },
    {
      action: "page",
      tip: "This Page",
      svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`
    },
    {
      action: "translate",
      tip: "Translate",
      svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M1 2h7v1.5H5.5v1H8v1.5H5.5c-.2 1-.7 2-1.5 2.7l1.7 1.8-1.1 1-1.6-1.8C2.5 10 2 10.2 1.5 10.3L1 8.8c.5-.1.9-.3 1.3-.5L1 6.8l1.1-1 1.2 1.4c.5-.5.9-1.1 1.1-1.7H1V2zm10 3l3 8h-1.5l-.6-1.7h-2.8L8.5 13H7l3-8h1zm-.5 2.5l-1 2.8h2l-1-2.8z"/>
    </svg>`
    },
    {
      action: "save",
      tip: "Save to Memory",
      svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 1h10a1 1 0 011 1v13l-6-3-6 3V2a1 1 0 011-1z"/>
    </svg>`
    }
  ];
  function createFab() {
    let _root = null;
    let _pill = null;
    let _glow = null;
    let _logo = null;
    let _popout = null;
    let _closeBtn = null;
    let _unsub = null;
    let _ext = null;
    let _side = "right";
    let _offset = 40;
    let _dragStartX = 0;
    let _dragStartY = 0;
    let _isDragging = false;
    let _totalMove = 0;
    let _hoverTimer = null;
    let _morphing = false;
    function buildGlow() {
      const glow = document.createElement("div");
      glow.className = "fab-glow";
      Object.assign(glow.style, {
        position: "absolute",
        inset: "-8px",
        borderRadius: "50px",
        background: "var(--aura-glow)",
        filter: "blur(12px)",
        animation: "aura-glow-pulse 3s ease-in-out infinite",
        pointerEvents: "none",
        zIndex: "-1"
      });
      return glow;
    }
    function buildLogo(iconSvg) {
      const logo = document.createElement("div");
      logo.className = "fab-logo";
      Object.assign(logo.style, {
        width: `${FAB.logoSize}px`,
        height: `${FAB.logoSize}px`,
        color: "var(--aura-accent)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexShrink: "0",
        transition: "color 0.3s ease"
      });
      logo.innerHTML = iconSvg;
      const svg = logo.querySelector("svg");
      if (svg) {
        svg.style.width = "100%";
        svg.style.height = "100%";
      }
      return logo;
    }
    function buildCloseBtn() {
      const btn = document.createElement("button");
      btn.className = "fab-close";
      btn.setAttribute("aria-label", "Close Aura");
      btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 12 12" fill="currentColor" width="10" height="10">
      <path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>`;
      Object.assign(btn.style, {
        position: "absolute",
        top: "-6px",
        right: "-6px",
        width: "16px",
        height: "16px",
        borderRadius: "50%",
        background: "rgba(10,8,24,0.9)",
        border: "1px solid rgba(255,255,255,0.15)",
        color: "rgba(255,255,255,0.6)",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "0",
        opacity: "0",
        transition: "opacity 0.2s",
        pointerEvents: "all"
      });
      return btn;
    }
    function buildActionBtn(def) {
      const btn = document.createElement("button");
      btn.className = "fab-action-btn";
      btn.dataset.action = def.action;
      btn.setAttribute("aria-label", def.tip);
      btn.innerHTML = def.svg;
      Object.assign(btn.style, {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "4px",
        background: "transparent",
        border: "none",
        color: "rgba(255,255,255,0.75)",
        cursor: "pointer",
        padding: "6px 8px",
        borderRadius: "8px",
        fontSize: "10px",
        fontFamily: FONT_STACK,
        transition: "background 0.15s, color 0.15s"
      });
      const svg = btn.querySelector("svg");
      if (svg) {
        svg.setAttribute("width", "16");
        svg.setAttribute("height", "16");
      }
      const tip = document.createElement("span");
      tip.textContent = def.tip;
      tip.style.fontSize = "10px";
      btn.appendChild(tip);
      btn.addEventListener("mouseenter", () => {
        btn.style.background = "rgba(255,255,255,0.08)";
        btn.style.color = "var(--aura-accent)";
      });
      btn.addEventListener("mouseleave", () => {
        btn.style.background = "transparent";
        btn.style.color = "rgba(255,255,255,0.75)";
      });
      return btn;
    }
    function buildPopout() {
      const popout = document.createElement("div");
      popout.className = "fab-popout hidden";
      Object.assign(popout.style, {
        position: "absolute",
        display: "flex",
        flexDirection: "row",
        gap: "4px",
        padding: "8px",
        background: "rgba(10,8,24,0.92)",
        backdropFilter: "blur(20px) saturate(1.5)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "14px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
        zIndex: String(Z_TOP),
        transition: "opacity 0.2s, transform 0.2s",
        opacity: "0",
        pointerEvents: "none"
      });
      for (const def of DEFAULT_ACTIONS) {
        popout.appendChild(buildActionBtn(def));
      }
      return popout;
    }
    function buildPill(signal) {
      const pill = document.createElement("div");
      pill.className = "fab-pill";
      Object.assign(pill.style, {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: FAB.pillPadding,
        background: "rgba(10,8,24,0.88)",
        backdropFilter: "blur(20px) saturate(1.5)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "50px",
        cursor: "pointer",
        position: "relative",
        boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
        transition: `padding ${FAB.expandDuration}ms ease, border-radius ${FAB.expandDuration}ms ease`,
        userSelect: "none",
        touchAction: "none"
      });
      _glow = buildGlow();
      pill.appendChild(_glow);
      _logo = buildLogo(signal.icon);
      pill.appendChild(_logo);
      _closeBtn = buildCloseBtn();
      pill.appendChild(_closeBtn);
      return pill;
    }
    function applyPosition() {
      if (!_root || !_pill) return;
      const margin = FAB.edgeMargin;
      Object.assign(_root.style, {
        position: "fixed",
        top: `${_offset}%`,
        [_side === "right" ? "right" : "left"]: `${margin}px`,
        [_side === "right" ? "left" : "right"]: "auto",
        zIndex: String(Z_TOP),
        transform: ""
      });
      positionPopout();
    }
    function positionPopout() {
      if (!_popout || !_root) return;
      const isRight = _side === "right";
      Object.assign(_popout.style, {
        top: "50%",
        transform: "translateY(-50%)",
        [isRight ? "right" : "left"]: `calc(100% + 8px)`,
        [isRight ? "left" : "right"]: "auto"
      });
    }
    function showPopout() {
      if (!_popout || !_pill) return;
      if (_hoverTimer) {
        clearTimeout(_hoverTimer);
        _hoverTimer = null;
      }
      _popout.classList.remove("hidden");
      _popout.style.opacity = "1";
      _popout.style.pointerEvents = "all";
      _pill.style.borderBottomRightRadius = _side === "right" ? "50px" : "50px";
      _closeBtn && (_closeBtn.style.opacity = "1");
    }
    function hidePopout() {
      if (!_popout || !_pill) return;
      _popout.style.opacity = "0";
      _popout.style.pointerEvents = "none";
      _hoverTimer = setTimeout(() => {
        _popout.classList.add("hidden");
        _closeBtn && (_closeBtn.style.opacity = "0");
      }, 200);
    }
    function setupHover(pill, popout) {
      let insidePill = false;
      let insidePopout = false;
      function maybeHide() {
        setTimeout(() => {
          if (!insidePill && !insidePopout) hidePopout();
        }, 0);
      }
      pill.addEventListener("mouseenter", () => {
        insidePill = true;
        showPopout();
      });
      pill.addEventListener("mouseleave", () => {
        insidePill = false;
        maybeHide();
      });
      popout.addEventListener("mouseenter", () => {
        insidePopout = true;
        showPopout();
      });
      popout.addEventListener("mouseleave", () => {
        insidePopout = false;
        maybeHide();
      });
    }
    function setupDrag(pill) {
      pill.addEventListener("pointerdown", (e) => {
        if (e.target.closest(".fab-close")) return;
        _dragStartX = e.clientX;
        _dragStartY = e.clientY;
        _isDragging = false;
        _totalMove = 0;
        pill.setPointerCapture(e.pointerId);
      });
      pill.addEventListener("pointermove", (e) => {
        if (!pill.hasPointerCapture(e.pointerId)) return;
        const dx = e.clientX - _dragStartX;
        const dy = e.clientY - _dragStartY;
        _totalMove = Math.sqrt(dx * dx + dy * dy);
        if (_totalMove > FAB.dragThreshold) {
          _isDragging = true;
          pill.classList.add("dragging");
          pill.style.borderRadius = "50%";
          const vpHeight = window.innerHeight;
          const newTopPx = e.clientY;
          const clampedPct = Math.min(Math.max(newTopPx / vpHeight * 100, 5), 90);
          if (_root) {
            _root.style.top = `${clampedPct}%`;
          }
        }
      });
      pill.addEventListener("pointerup", (e) => {
        if (!_isDragging) return;
        pill.classList.remove("dragging");
        pill.style.borderRadius = "50px";
        const vpWidth = window.innerWidth;
        _side = e.clientX > vpWidth / 2 ? "right" : "left";
        const vpHeight = window.innerHeight;
        _offset = Math.min(Math.max(e.clientY / vpHeight * 100, 5), 90);
        _isDragging = false;
        applyPosition();
        savePersistence();
      });
    }
    function setupClicks(pill, popout) {
      pill.addEventListener("click", (e) => {
        if (e.target.closest(".fab-close")) return;
        if (_isDragging || _totalMove > FAB.dragThreshold) return;
        if (!_ext) return;
        _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "chat" });
      });
      _closeBtn == null ? void 0 : _closeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (_root) _root.style.display = "none";
      });
      popout.addEventListener("click", (e) => {
        const btn = e.target.closest(".fab-action-btn");
        if (!btn || !_ext) return;
        const action = btn.dataset.action ?? "";
        dispatchAction(action);
      });
    }
    function getSelectionText() {
      const sel = window.getSelection();
      return sel ? sel.toString().trim() : "";
    }
    function dispatchAction(action) {
      if (!_ext) return;
      const url = location.href;
      const title = document.title;
      switch (action) {
        case "chat":
          _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "chat" });
          break;
        case "search":
          _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "search" });
          break;
        case "page":
          _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "ask" });
          break;
        case "translate":
          _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "translate" });
          break;
        case "save": {
          const selText = getSelectionText();
          const textToSave = selText || `${title}
${url}`;
          _ext.runtime.sendMessage(
            { type: "SAVE_KNOWLEDGE", text: textToSave, url, title },
            (response) => {
            }
          );
          break;
        }
      }
    }
    function handleContextChange(signal) {
      var _a;
      if (!_root || !_logo) return;
      _root.style.setProperty("--aura-accent", signal.accent);
      _root.style.setProperty("--aura-glow", signal.glow);
      if (!_morphing && _logo) {
        _morphing = true;
        const oldLogo = _logo;
        const newLogo = buildLogo(signal.icon);
        newLogo.style.position = "absolute";
        newLogo.style.inset = "0";
        newLogo.style.opacity = "0";
        (_a = oldLogo.parentElement) == null ? void 0 : _a.appendChild(newLogo);
        crossFade(oldLogo, newLogo, {
          duration: ANIM.crossFadeDuration,
          easing: "ease"
        }).then(() => {
          oldLogo.remove();
          newLogo.style.position = "";
          newLogo.style.inset = "";
          newLogo.style.opacity = "1";
          _logo = newLogo;
          _morphing = false;
        });
      }
    }
    function loadPersistence() {
      if (!_ext) return;
      _ext.storage.local.get(["auraFabSide", "auraFabOffset"], (result) => {
        if (result.auraFabSide === "left" || result.auraFabSide === "right") {
          _side = result.auraFabSide;
        }
        if (typeof result.auraFabOffset === "number") {
          _offset = result.auraFabOffset;
        }
        applyPosition();
      });
    }
    function savePersistence() {
      if (!_ext) return;
      _ext.storage.local.set({ auraFabSide: _side, auraFabOffset: _offset });
    }
    return {
      init(container, store, ext2) {
        _ext = ext2;
        const signal = store.get();
        const root = document.createElement("div");
        root.className = "aura-fab";
        Object.assign(root.style, {
          position: "fixed",
          zIndex: String(Z_TOP),
          fontFamily: FONT_STACK,
          // CSS custom props for theming
          "--aura-accent": signal.accent,
          "--aura-glow": signal.glow
        });
        _root = root;
        const pill = buildPill(signal);
        _pill = pill;
        root.appendChild(pill);
        const popout = buildPopout();
        _popout = popout;
        root.appendChild(popout);
        container.appendChild(root);
        setupHover(pill, popout);
        setupDrag(pill);
        setupClicks(pill, popout);
        loadPersistence();
        _unsub = store.subscribe(handleContextChange);
      },
      destroy() {
        if (_unsub) {
          _unsub();
          _unsub = null;
        }
        if (_hoverTimer) {
          clearTimeout(_hoverTimer);
          _hoverTimer = null;
        }
        _root == null ? void 0 : _root.remove();
        _root = null;
        _pill = null;
        _glow = null;
        _logo = null;
        _popout = null;
        _closeBtn = null;
        _ext = null;
      },
      showDock() {
        if (_root) _root.style.display = "";
      }
    };
  }
  const ICONS = {
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
    "read-aloud": `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" width="14" height="14">
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
  </svg>`
  };
  function getIcon(action) {
    return ICONS[action] ?? ICONS.more;
  }
  function applyBarStyle(el, rect, top, height, bg) {
    Object.assign(el.style, {
      position: "fixed",
      left: `${rect.left}px`,
      top: `${top}px`,
      width: `${rect.width}px`,
      height: `${height}px`,
      background: bg,
      backdropFilter: GLASS.backdrop,
      WebkitBackdropFilter: GLASS.backdrop,
      border: `1px solid rgba(255,255,255,${GLASS.borderOpacity})`,
      boxShadow: GLASS.shadowBase,
      borderRadius: "6px",
      display: "flex",
      alignItems: "center",
      gap: "2px",
      padding: "0 6px",
      overflow: "hidden",
      boxSizing: "border-box",
      zIndex: String(Z_TOP),
      userSelect: "none"
    });
  }
  function makeActionButton(action) {
    const btn = document.createElement("button");
    btn.className = "gb-action";
    btn.dataset.action = action;
    btn.title = action;
    btn.innerHTML = getIcon(action);
    Object.assign(btn.style, {
      background: "none",
      border: "none",
      cursor: "pointer",
      color: "rgba(255,255,255,0.85)",
      padding: "3px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      borderRadius: "4px",
      flexShrink: "0"
    });
    return btn;
  }
  function createGhostBar() {
    let bar = null;
    let barType = null;
    let currentText = "";
    let currentImageUrl = "";
    let currentRect = null;
    let containerEl = document.body;
    let storeRef = null;
    let extRef = null;
    let askCallback = null;
    let selectionTimer = null;
    let imageHoverTimer = null;
    let dismissTimer = null;
    let currentHoveredImg = null;
    const cleanups = [];
    function removeBarSync() {
      if (bar) {
        bar.remove();
        bar = null;
        barType = null;
        currentRect = null;
      }
    }
    async function hideBar() {
      if (!bar) return;
      const b = bar;
      bar = null;
      barType = null;
      currentRect = null;
      try {
        await flow(b, { direction: "up", duration: ANIM.morphDuration, easing: ANIM.morphEasing });
      } catch {
      }
      b.remove();
    }
    function handleActionClick(action) {
      var _a;
      if (!extRef) return;
      if (action === "ask") {
        if (askCallback) {
          const payload = {
            type: barType === "image" ? "image" : "text",
            text: currentText,
            imageUrl: currentImageUrl,
            rect: bar ? bar.getBoundingClientRect() : currentRect ?? new DOMRect()
          };
          askCallback(payload);
        }
        return;
      }
      if (action === "copy") {
        (_a = navigator.clipboard) == null ? void 0 : _a.writeText(currentText).catch(() => {
        });
        return;
      }
      if (action === "highlight") {
        extRef.runtime.sendMessage({ type: "SAVE_KNOWLEDGE", text: currentText, url: location.href, title: document.title });
        return;
      }
      if (action === "describe") {
        extRef.runtime.sendMessage({ type: "IMAGE_DESCRIBE", imageUrl: currentImageUrl });
        return;
      }
      if (action === "edit") {
        extRef.runtime.sendMessage({ type: "IMAGE_EDIT_OPEN", imageUrl: currentImageUrl });
        return;
      }
      if (action === "save") {
        extRef.runtime.sendMessage({ type: "IMAGE_SAVE", imageUrl: currentImageUrl });
        return;
      }
      const msg = { type: "QUICK_ACTION", action, text: currentText };
      extRef.runtime.sendMessage(msg);
    }
    function attachClickListeners(barEl) {
      barEl.addEventListener("click", (e) => {
        const target = e.target.closest(".gb-action");
        if (!target) return;
        const action = target.dataset.action ?? "";
        if (action === "more") {
          const extended = barEl.querySelector(".gb-extended");
          if (extended) {
            extended.style.display = extended.style.display === "none" ? "flex" : "none";
          }
          return;
        }
        handleActionClick(action);
      });
    }
    function detectSelectionType(text) {
      const t = text.trim();
      if (/^(https?:\/\/|www\.)\S+$/.test(t)) return "url";
      if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(t)) return "email";
      const codeScore = [
        /[{}\[\]();]/.test(t),
        /\b(function|const|let|var|def|class|import|return|if|for|while)\b/.test(t),
        /=>|->|::/.test(t),
        /^\s{2,}/m.test(t)
      ].filter(Boolean).length;
      if (codeScore >= 2) return "code";
      return "text";
    }
    function showTextBar(selectionRect, text) {
      if (!storeRef) return;
      const signal = storeRef.get();
      if (signal.suppressGhostBars) return;
      removeBarSync();
      currentText = text;
      currentImageUrl = "";
      currentRect = selectionRect;
      barType = "text";
      const barEl = document.createElement("div");
      barEl.className = "ghost-bar ghost-bar-text";
      const top = selectionRect.bottom;
      applyBarStyle(barEl, selectionRect, top, GHOST_BAR.height, GLASS.bg);
      const selType = detectSelectionType(text);
      const SMART_ACTIONS = {
        code: ["explain", "ask", "copy"],
        url: ["ask", "summarize", "copy"],
        email: ["ask", "copy"],
        text: []
      };
      const baseActions = SMART_ACTIONS[selType].length ? SMART_ACTIONS[selType] : signal.actions;
      const actions = baseActions.slice(0, GHOST_BAR.maxActionsPerRow - 1);
      for (const action of actions) {
        barEl.appendChild(makeActionButton(action));
      }
      const moreBtn = makeActionButton("more");
      barEl.appendChild(moreBtn);
      const extended = document.createElement("div");
      extended.className = "gb-extended";
      const extActions = ["rewrite", "grammar", "define", "read-aloud"];
      for (const action of extActions) {
        extended.appendChild(makeActionButton(action));
      }
      Object.assign(extended.style, {
        display: "none",
        position: "absolute",
        top: `${GHOST_BAR.height}px`,
        left: "0",
        right: "0",
        background: GLASS.bg,
        borderRadius: "0 0 6px 6px",
        padding: "2px 6px",
        gap: "2px"
      });
      barEl.style.position = "fixed";
      barEl.appendChild(extended);
      attachClickListeners(barEl);
      containerEl.appendChild(barEl);
      bar = barEl;
      flow(barEl, {
        direction: "down",
        duration: ANIM.morphDuration,
        easing: ANIM.morphEasing
      }).catch(() => {
      });
    }
    function showImageBar(img) {
      if (!storeRef) return;
      const signal = storeRef.get();
      if (signal.suppressGhostBars) return;
      removeBarSync();
      const imgRect = img.getBoundingClientRect();
      currentText = "";
      currentImageUrl = img.src ?? img.currentSrc ?? "";
      currentRect = imgRect;
      barType = "image";
      const barEl = document.createElement("div");
      barEl.className = "ghost-bar ghost-bar-image";
      const top = imgRect.bottom - GHOST_BAR.imageBarHeight;
      applyBarStyle(barEl, imgRect, top, GHOST_BAR.imageBarHeight, GLASS.bgHeavy);
      const imageActions = ["describe", "edit", "save", "ask"];
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
        direction: "down",
        duration: ANIM.morphDuration,
        easing: ANIM.morphEasing
      }).catch(() => {
      });
    }
    function getBarRect() {
      return bar ? bar.getBoundingClientRect() : null;
    }
    function onAskClicked(cb) {
      askCallback = cb;
    }
    function init2(container, store, ext2) {
      containerEl = container;
      storeRef = store;
      extRef = ext2;
      const onSelectionChange = () => {
        if (selectionTimer) clearTimeout(selectionTimer);
        selectionTimer = setTimeout(() => {
          const sel = window.getSelection();
          if (!sel || sel.rangeCount === 0 || sel.toString().trim().length === 0) {
            hideBar().catch(() => {
            });
            return;
          }
          const range = sel.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          if (rect.width === 0 && rect.height === 0) return;
          showTextBar(rect, sel.toString());
        }, ANIM.selectionDelay);
      };
      document.addEventListener("selectionchange", onSelectionChange);
      const onMouseOver = (e) => {
        const target = e.target;
        if (target.tagName !== "IMG") return;
        const img = target;
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
      const onMouseOut = (e) => {
        const target = e.target;
        const related = e.relatedTarget;
        const leavingImg = target.tagName === "IMG";
        const leavingBar = bar && (target === bar || bar.contains(target));
        if (!leavingImg && !leavingBar) return;
        if (related && bar && (related === bar || bar.contains(related))) return;
        if (related && related.tagName === "IMG" && related === currentHoveredImg) return;
        if (imageHoverTimer) {
          clearTimeout(imageHoverTimer);
          imageHoverTimer = null;
        }
        if (barType === "image") {
          if (dismissTimer) clearTimeout(dismissTimer);
          dismissTimer = setTimeout(() => {
            hideBar().catch(() => {
            });
          }, ANIM.dismissDelay);
        }
      };
      document.addEventListener("mouseover", onMouseOver, true);
      document.addEventListener("mouseout", onMouseOut, true);
      cleanups.push(
        () => document.removeEventListener("selectionchange", onSelectionChange),
        () => document.removeEventListener("mouseover", onMouseOver, true),
        () => document.removeEventListener("mouseout", onMouseOut, true)
      );
      const onScroll = () => {
        if (!bar || !currentRect) return;
        const viewH = window.innerHeight;
        const viewW = window.innerWidth;
        if (barType === "text") {
          const sel = window.getSelection();
          if (!sel || sel.rangeCount === 0) {
            hideBar().catch(() => {
            });
            return;
          }
          const rect = sel.getRangeAt(0).getBoundingClientRect();
          if (rect.bottom < 0 || rect.top > viewH || rect.right < 0 || rect.left > viewW) {
            hideBar().catch(() => {
            });
            return;
          }
          bar.style.top = `${rect.bottom}px`;
          bar.style.left = `${rect.left}px`;
          bar.style.width = `${rect.width}px`;
          currentRect = rect;
        } else if (barType === "image" && currentHoveredImg) {
          const rect = currentHoveredImg.getBoundingClientRect();
          if (rect.bottom < 0 || rect.top > viewH || rect.right < 0 || rect.left > viewW) {
            hideBar().catch(() => {
            });
            return;
          }
          const top = rect.bottom - GHOST_BAR.imageBarHeight;
          bar.style.top = `${top}px`;
          bar.style.left = `${rect.left}px`;
          bar.style.width = `${rect.width}px`;
          currentRect = rect;
        }
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      cleanups.push(() => window.removeEventListener("scroll", onScroll));
    }
    function destroy() {
      if (selectionTimer) clearTimeout(selectionTimer);
      if (imageHoverTimer) clearTimeout(imageHoverTimer);
      if (dismissTimer) clearTimeout(dismissTimer);
      for (const fn of cleanups) fn();
      cleanups.length = 0;
      removeBarSync();
    }
    return {
      init: init2,
      destroy,
      showTextBar,
      showImageBar,
      hideBar,
      getBarRect,
      onAskClicked
    };
  }
  function truncateText(text) {
    if (text.length <= MODAL.previewMaxChars) return text;
    const remaining = text.length - MODAL.previewMaxChars;
    return text.slice(0, MODAL.previewMaxChars) + `... (${remaining} more chars)`;
  }
  function getPlaceholder(storeType) {
    switch (storeType) {
      case "article":
        return "Ask about this article...";
      case "code":
        return "Ask about this code...";
      default:
        return "Ask anything about this text...";
    }
  }
  function centeredRect() {
    const w = Math.min(MODAL.maxWidth, window.innerWidth - 32);
    const h = Math.min(MODAL.maxHeight, window.innerHeight - 32);
    const left = (window.innerWidth - w) / 2;
    const top = (window.innerHeight - h) / 2;
    return {
      left,
      top,
      right: left + w,
      bottom: top + h,
      width: w,
      height: h,
      x: left,
      y: top,
      toJSON: () => ({})
    };
  }
  function applyGlassStyle(el, rect) {
    Object.assign(el.style, {
      position: "fixed",
      left: "0",
      top: "0",
      width: `${rect.width}px`,
      height: `${rect.height}px`,
      transform: `translate(${rect.left}px, ${rect.top}px)`,
      background: GLASS.bg,
      backdropFilter: GLASS.backdrop,
      WebkitBackdropFilter: GLASS.backdrop,
      border: `1px solid rgba(255,255,255,${GLASS.borderOpacity})`,
      borderRadius: "16px",
      boxShadow: GLASS.shadowBase,
      fontFamily: FONT_STACK,
      color: "#e5e7eb",
      overflow: "hidden",
      zIndex: String(Z_TOP),
      boxSizing: "border-box"
    });
  }
  function buildTextContent(text, placeholder) {
    const wrap = document.createElement("div");
    wrap.className = "modal-content-wrap";
    Object.assign(wrap.style, {
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      padding: "16px",
      height: "100%",
      boxSizing: "border-box",
      opacity: "0"
    });
    const preview = document.createElement("div");
    preview.className = "modal-preview";
    Object.assign(preview.style, {
      fontSize: "13px",
      lineHeight: "1.5",
      color: "rgba(229,231,235,0.75)",
      overflow: "hidden",
      display: "-webkit-box",
      WebkitLineClamp: String(MODAL.previewMaxLines),
      WebkitBoxOrient: "vertical",
      maxHeight: `${MODAL.previewMaxLines * 20}px`,
      flexShrink: "0"
    });
    preview.textContent = truncateText(text);
    const input = document.createElement("input");
    input.type = "text";
    input.className = "modal-input";
    input.placeholder = placeholder;
    Object.assign(input.style, {
      background: "rgba(255,255,255,0.07)",
      border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: "8px",
      padding: "8px 12px",
      color: "#e5e7eb",
      fontSize: "14px",
      fontFamily: FONT_STACK,
      outline: "none",
      flexShrink: "0"
    });
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    Object.assign(actions.style, {
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
      flexShrink: "0"
    });
    const actionDefs = [
      { label: "Explain", value: "explain" },
      { label: "Summarize", value: "summarize" },
      { label: "Chat with AURA", value: "chat" },
      { label: "Save to Memory", value: "save" },
      { label: "Translate", value: "translate" }
    ];
    for (const def of actionDefs) {
      const btn = document.createElement("button");
      btn.className = "modal-action-btn";
      btn.textContent = def.label;
      btn.dataset.action = def.value;
      Object.assign(btn.style, {
        background: "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "6px",
        padding: "5px 10px",
        color: "#e5e7eb",
        fontSize: "12px",
        fontFamily: FONT_STACK,
        cursor: "pointer"
      });
      actions.appendChild(btn);
    }
    const modelRow = document.createElement("div");
    modelRow.className = "modal-model-row";
    Object.assign(modelRow.style, {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      marginTop: "auto",
      flexShrink: "0"
    });
    const modelLabel = document.createElement("span");
    modelLabel.textContent = "Model";
    Object.assign(modelLabel.style, { fontSize: "12px", color: "rgba(229,231,235,0.5)" });
    const select = document.createElement("select");
    select.className = "modal-model-select";
    Object.assign(select.style, {
      background: "rgba(255,255,255,0.07)",
      border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: "6px",
      padding: "4px 8px",
      color: "#e5e7eb",
      fontSize: "12px",
      fontFamily: FONT_STACK,
      cursor: "pointer"
    });
    const modelOptions = [
      { label: "Auto", value: "auto" },
      { label: "Fast", value: "fast" },
      { label: "Balanced", value: "balanced" },
      { label: "Powerful", value: "powerful" }
    ];
    for (const opt of modelOptions) {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      select.appendChild(option);
    }
    modelRow.appendChild(modelLabel);
    modelRow.appendChild(select);
    wrap.appendChild(preview);
    wrap.appendChild(input);
    wrap.appendChild(actions);
    wrap.appendChild(modelRow);
    return wrap;
  }
  function buildImageContent(imageUrl) {
    const wrap = document.createElement("div");
    wrap.className = "modal-content-wrap";
    Object.assign(wrap.style, {
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      padding: "16px",
      height: "100%",
      boxSizing: "border-box",
      opacity: "0"
    });
    const preview = document.createElement("div");
    preview.className = "modal-preview";
    Object.assign(preview.style, {
      flexShrink: "0",
      overflow: "hidden",
      borderRadius: "8px"
    });
    const img = document.createElement("img");
    img.src = imageUrl;
    Object.assign(img.style, {
      maxWidth: "100%",
      maxHeight: `${MODAL.imagePreviewMaxHeight}px`,
      objectFit: "contain",
      display: "block"
    });
    preview.appendChild(img);
    const actions = document.createElement("div");
    actions.className = "modal-actions";
    Object.assign(actions.style, {
      display: "flex",
      flexWrap: "wrap",
      gap: "6px",
      flexShrink: "0"
    });
    const actionDefs = [
      { label: "Describe", value: "describe" },
      { label: "Summarize", value: "summarize" },
      { label: "Chat with AURA", value: "chat" },
      { label: "Save to Memory", value: "save" },
      { label: "Translate", value: "translate" }
    ];
    for (const def of actionDefs) {
      const btn = document.createElement("button");
      btn.className = "modal-action-btn";
      btn.textContent = def.label;
      btn.dataset.action = def.value;
      Object.assign(btn.style, {
        background: "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "6px",
        padding: "5px 10px",
        color: "#e5e7eb",
        fontSize: "12px",
        fontFamily: FONT_STACK,
        cursor: "pointer"
      });
      actions.appendChild(btn);
    }
    const input = document.createElement("input");
    input.type = "text";
    input.className = "modal-input";
    input.placeholder = "Ask about this image...";
    Object.assign(input.style, {
      background: "rgba(255,255,255,0.07)",
      border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: "8px",
      padding: "8px 12px",
      color: "#e5e7eb",
      fontSize: "14px",
      fontFamily: FONT_STACK,
      outline: "none",
      flexShrink: "0"
    });
    const modelRow = document.createElement("div");
    modelRow.className = "modal-model-row";
    Object.assign(modelRow.style, {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      marginTop: "auto",
      flexShrink: "0"
    });
    const modelLabel = document.createElement("span");
    modelLabel.textContent = "Model";
    Object.assign(modelLabel.style, { fontSize: "12px", color: "rgba(229,231,235,0.5)" });
    const select = document.createElement("select");
    select.className = "modal-model-select";
    Object.assign(select.style, {
      background: "rgba(255,255,255,0.07)",
      border: "1px solid rgba(255,255,255,0.15)",
      borderRadius: "6px",
      padding: "4px 8px",
      color: "#e5e7eb",
      fontSize: "12px",
      fontFamily: FONT_STACK,
      cursor: "pointer"
    });
    for (const opt of [
      { label: "Auto", value: "auto" },
      { label: "Fast", value: "fast" },
      { label: "Balanced", value: "balanced" },
      { label: "Powerful", value: "powerful" }
    ]) {
      const option = document.createElement("option");
      option.value = opt.value;
      option.textContent = opt.label;
      select.appendChild(option);
    }
    modelRow.appendChild(modelLabel);
    modelRow.appendChild(select);
    wrap.appendChild(preview);
    wrap.appendChild(actions);
    wrap.appendChild(input);
    wrap.appendChild(modelRow);
    return wrap;
  }
  function createModal() {
    let internals = {
      overlay: null,
      modal: null,
      originRect: null,
      content: "",
      isOpen: false,
      closing: false,
      opening: false
    };
    let store_ = null;
    let actionCallback = null;
    const onKeyDown = (e) => {
      if (e.key === "Escape" && internals.isOpen) {
        close();
      }
    };
    async function open(contentEl, originRect, content) {
      if (internals.opening) return;
      if (internals.isOpen) await close();
      internals.opening = true;
      internals.originRect = originRect;
      internals.content = content;
      internals.isOpen = true;
      const overlay = document.createElement("div");
      overlay.className = "aura-modal-overlay";
      Object.assign(overlay.style, {
        position: "fixed",
        inset: "0",
        background: "rgba(0,0,0,0.3)",
        zIndex: String(Z_TOP - 1),
        opacity: "0"
      });
      document.body.appendChild(overlay);
      internals.overlay = overlay;
      const modal = document.createElement("div");
      modal.className = "aura-modal";
      applyGlassStyle(modal, originRect);
      document.body.appendChild(modal);
      internals.modal = modal;
      fadeIn(overlay, {
        duration: ANIM.flowDuration,
        easing: "ease-out"
      }).then(() => {
        overlay.style.opacity = "1";
      });
      const target = centeredRect();
      await morph(modal, originRect, target, {
        duration: ANIM.morphDuration,
        easing: ANIM.morphEasing
      });
      applyGlassStyle(modal, target);
      modal.appendChild(contentEl);
      fadeIn(contentEl, {
        duration: ANIM.crossFadeDuration,
        easing: "ease-out"
      }).then(() => {
        contentEl.style.opacity = "1";
      });
      modal.querySelectorAll(".modal-action-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
          const action = btn.dataset.action ?? "ask";
          const select = modal.querySelector(".modal-model-select");
          const model = (select == null ? void 0 : select.value) ?? "auto";
          actionCallback == null ? void 0 : actionCallback(action, internals.content, model);
        });
      });
      const inputEl = modal.querySelector(".modal-input");
      if (inputEl) {
        inputEl.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            const select = modal.querySelector(".modal-model-select");
            const model = (select == null ? void 0 : select.value) ?? "auto";
            actionCallback == null ? void 0 : actionCallback("ask", inputEl.value, model);
          }
        });
      }
      overlay.addEventListener("click", () => close());
      document.addEventListener("keydown", onKeyDown);
      internals.opening = false;
    }
    async function close() {
      if (!internals.isOpen || internals.closing) return;
      internals.closing = true;
      const { modal, overlay, originRect } = internals;
      document.removeEventListener("keydown", onKeyDown);
      const animations = [];
      if (modal && originRect) {
        const currentRect = centeredRect();
        animations.push(
          morph(modal, currentRect, originRect, {
            duration: ANIM.morphDuration,
            easing: ANIM.morphEasing
          }).catch(() => {
          })
        );
      }
      if (overlay) {
        animations.push(
          dissolve(overlay, {
            duration: ANIM.morphDuration,
            easing: "ease-in"
          }).catch(() => {
          })
        );
      }
      await Promise.all(animations);
      modal == null ? void 0 : modal.remove();
      overlay == null ? void 0 : overlay.remove();
      internals = {
        overlay: null,
        modal: null,
        originRect: null,
        content: "",
        isOpen: false,
        closing: false,
        opening: false
      };
    }
    return {
      // ── ContentModule ──
      init(_container, store, _ext) {
        store_ = store;
      },
      destroy() {
        close();
      },
      // ── ModalAPI ──
      openWithText(text, originRect) {
        const placeholder = getPlaceholder((store_ == null ? void 0 : store_.get().type) ?? "general");
        const contentEl = buildTextContent(text, placeholder);
        open(contentEl, originRect, text);
      },
      openWithImage(imageUrl, originRect) {
        const contentEl = buildImageContent(imageUrl);
        open(contentEl, originRect, imageUrl);
      },
      close,
      onAction(cb) {
        actionCallback = cb;
      }
    };
  }
  function createHighlights() {
    let _ext;
    let _showToast = () => {
    };
    function safeSend2(msg, cb) {
      try {
        if (cb) {
          _ext.runtime.sendMessage(msg, cb);
        } else {
          _ext.runtime.sendMessage(msg);
        }
      } catch (_e) {
      }
    }
    const hlHost = document.createElement("div");
    hlHost.id = "aura-highlight-host";
    Object.assign(hlHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483646",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(hlHost);
    const hlShadow = hlHost.attachShadow({ mode: "closed" });
    const hlStyle = document.createElement("style");
    hlStyle.textContent = `
    @keyframes hl-tooltip-in {
      from { opacity: 0; transform: translateY(4px) scale(0.95); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }
    .hl-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.3);
      border-radius: 8px;
      padding: 5px 10px;
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: auto;
      animation: hl-tooltip-in 0.15s ease forwards;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
    }
    .hl-tooltip-text {
      color: rgba(226, 232, 240, 0.9);
      font-size: 11px;
      font-weight: 500;
      white-space: nowrap;
    }
    .hl-tooltip-delete {
      width: 18px; height: 18px; border-radius: 4px;
      background: transparent; border: none;
      color: rgba(226, 232, 240, 0.5);
      cursor: pointer; display: flex; align-items: center; justify-content: center;
      padding: 0; transition: background 0.12s, color 0.12s;
    }
    .hl-tooltip-delete:hover {
      background: rgba(239, 68, 68, 0.25);
      color: rgba(239, 68, 68, 1);
    }
  `;
    hlShadow.appendChild(hlStyle);
    const hlContainer = document.createElement("div");
    hlShadow.appendChild(hlContainer);
    const pageHlStyle = document.createElement("style");
    pageHlStyle.textContent = `
    mark[data-aura-hl] {
      background: rgba(124, 58, 237, 0.15);
      border-bottom: 2px solid rgba(124, 58, 237, 0.5);
      border-radius: 2px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    mark[data-aura-hl]:hover {
      background: rgba(124, 58, 237, 0.28);
    }
    mark[data-aura-hl].aura-hl-stale {
      background: rgba(124, 58, 237, 0.08);
      border-bottom: 2px dashed rgba(124, 58, 237, 0.35);
    }
    mark[data-aura-hl].aura-hl-flash {
      background: rgba(124, 58, 237, 0.45) !important;
      transition: background 0.3s ease;
    }
  `;
    document.head.appendChild(pageHlStyle);
    let _hlTooltipEl = null;
    let _hlTooltipTimer = null;
    function removeHlTooltip() {
      if (_hlTooltipTimer) {
        clearTimeout(_hlTooltipTimer);
        _hlTooltipTimer = null;
      }
      if (_hlTooltipEl) {
        _hlTooltipEl.remove();
        _hlTooltipEl = null;
      }
    }
    function showHlTooltip(mark, highlightId) {
      removeHlTooltip();
      const rect = mark.getBoundingClientRect();
      _hlTooltipEl = document.createElement("div");
      _hlTooltipEl.className = "hl-tooltip";
      const label = document.createElement("span");
      label.className = "hl-tooltip-text";
      label.textContent = "Saved to AURA";
      _hlTooltipEl.appendChild(label);
      const delBtn = document.createElement("button");
      delBtn.className = "hl-tooltip-delete";
      delBtn.title = "Remove highlight";
      delBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deleteHighlightFromPage(highlightId);
        removeHlTooltip();
      });
      _hlTooltipEl.appendChild(delBtn);
      _hlTooltipEl.style.top = `${Math.round(rect.top - 34)}px`;
      _hlTooltipEl.style.left = `${Math.round(rect.left + rect.width / 2 - 60)}px`;
      hlContainer.appendChild(_hlTooltipEl);
    }
    function getXPath(node) {
      if (node.nodeType === Node.DOCUMENT_NODE) return "/";
      const parts = [];
      let current = node;
      while (current && current !== document) {
        if (current.nodeType === Node.ELEMENT_NODE) {
          const el = current;
          let tag = el.tagName.toLowerCase();
          const parent = el.parentNode;
          if (parent) {
            const siblings = Array.from(parent.childNodes).filter(
              (n) => n.nodeType === Node.ELEMENT_NODE && n.tagName === el.tagName
            );
            if (siblings.length > 1) {
              const idx = siblings.indexOf(el) + 1;
              tag += `[${idx}]`;
            }
          }
          parts.unshift(tag);
        } else if (current.nodeType === Node.TEXT_NODE) {
          const parent = current.parentNode;
          if (parent) {
            const textNodes = Array.from(parent.childNodes).filter(
              (n) => n.nodeType === Node.TEXT_NODE
            );
            if (textNodes.length > 1) {
              const idx = textNodes.indexOf(current) + 1;
              parts.unshift(`text()[${idx}]`);
            } else {
              parts.unshift("text()");
            }
          }
        }
        current = current.parentNode;
      }
      return "/" + parts.join("/");
    }
    function getHighlightContext(range) {
      const container = range.commonAncestorContainer;
      const fullText = container.nodeType === Node.TEXT_NODE ? container.textContent || "" : container.textContent || "";
      const selectedText = range.toString();
      const idx = fullText.indexOf(selectedText);
      if (idx === -1) return "";
      const before = fullText.slice(Math.max(0, idx - 50), idx);
      const after = fullText.slice(idx + selectedText.length, idx + selectedText.length + 50);
      return before + "|||" + after;
    }
    function generateHighlightId() {
      return "hl_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8);
    }
    function attachMarkListeners(mark) {
      const hlId = mark.getAttribute("data-aura-hl") || "";
      mark.addEventListener("mouseenter", () => showHlTooltip(mark, hlId));
      mark.addEventListener("mouseleave", () => {
        _hlTooltipTimer = setTimeout(removeHlTooltip, 300);
      });
    }
    function wrapSelectionWithMark(highlightId) {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0) return null;
      const range = sel.getRangeAt(0);
      if (range.collapsed) return null;
      try {
        const mark = document.createElement("mark");
        mark.setAttribute("data-aura-hl", highlightId);
        range.surroundContents(mark);
        sel.removeAllRanges();
        attachMarkListeners(mark);
        return mark;
      } catch (_e) {
        try {
          const frag = range.cloneContents();
          const textContent = frag.textContent || "";
          if (!textContent.trim()) return null;
          range.deleteContents();
          const mark = document.createElement("mark");
          mark.setAttribute("data-aura-hl", highlightId);
          mark.textContent = textContent;
          range.insertNode(mark);
          sel.removeAllRanges();
          attachMarkListeners(mark);
          return mark;
        } catch (_e2) {
          return null;
        }
      }
    }
    function saveHighlight() {
      const sel = window.getSelection();
      if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return false;
      const range = sel.getRangeAt(0);
      const text = range.toString().trim();
      if (!text) return false;
      const highlightId = generateHighlightId();
      const xpath = getXPath(range.startContainer);
      const context = getHighlightContext(range);
      const mark = wrapSelectionWithMark(highlightId);
      if (!mark) return false;
      const highlight = {
        id: highlightId,
        url: window.location.href,
        text,
        xpath,
        context,
        timestamp: Date.now(),
        color: "purple",
        pageTitle: document.title
      };
      safeSend2(
        { type: "SAVE_HIGHLIGHT", highlight },
        (response) => {
          if (response && response.ok) {
            _showToast("Highlight saved to AURA");
          } else {
            _showToast((response == null ? void 0 : response.error) || "Failed to save highlight", 3e3);
          }
        }
      );
      return true;
    }
    function deleteHighlightFromPage(highlightId) {
      const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
      if (mark) {
        const parent = mark.parentNode;
        while (mark.firstChild) parent == null ? void 0 : parent.insertBefore(mark.firstChild, mark);
        mark.remove();
        parent == null ? void 0 : parent.normalize();
      }
      safeSend2(
        { type: "DELETE_HIGHLIGHT", id: highlightId, url: window.location.href },
        (_response) => {
          _showToast("Highlight removed");
        }
      );
    }
    function findTextNode(xpath, text, context) {
      try {
        const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
        const node = result.singleNodeValue;
        if (node && node.textContent && node.textContent.includes(text)) {
          const range = document.createRange();
          const idx = node.textContent.indexOf(text);
          if (idx >= 0) {
            range.setStart(node, idx);
            range.setEnd(node, idx + text.length);
            return range;
          }
        }
      } catch (_e) {
      }
      const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
      const [contextBefore, contextAfter] = context.split("|||");
      let bestNode = null;
      let bestOffset = -1;
      let bestScore = 0;
      while (walker.nextNode()) {
        const tNode = walker.currentNode;
        const nodeText = tNode.textContent || "";
        const idx = nodeText.indexOf(text);
        if (idx === -1) continue;
        let score = 1;
        if (contextBefore) {
          const before = nodeText.slice(Math.max(0, idx - 50), idx);
          if (before.includes(contextBefore.slice(-20))) score += 2;
        }
        if (contextAfter) {
          const after = nodeText.slice(idx + text.length, idx + text.length + 50);
          if (after.includes(contextAfter.slice(0, 20))) score += 2;
        }
        if (score > bestScore) {
          bestScore = score;
          bestNode = tNode;
          bestOffset = idx;
        }
      }
      if (bestNode && bestOffset >= 0) {
        const range = document.createRange();
        range.setStart(bestNode, bestOffset);
        range.setEnd(bestNode, bestOffset + text.length);
        return range;
      }
      return null;
    }
    function restoreHighlight(hl) {
      if (document.querySelector(`mark[data-aura-hl="${hl.id}"]`)) return true;
      const range = findTextNode(hl.xpath, hl.text, hl.context);
      if (!range) return false;
      try {
        const mark = document.createElement("mark");
        mark.setAttribute("data-aura-hl", hl.id);
        if (hl.stale) mark.classList.add("aura-hl-stale");
        range.surroundContents(mark);
        attachMarkListeners(mark);
        return true;
      } catch (_e) {
        try {
          const text = range.toString();
          range.deleteContents();
          const mark = document.createElement("mark");
          mark.setAttribute("data-aura-hl", hl.id);
          if (hl.stale) mark.classList.add("aura-hl-stale");
          mark.textContent = text;
          range.insertNode(mark);
          attachMarkListeners(mark);
          return true;
        } catch (_e2) {
          return false;
        }
      }
    }
    function restoreAllHighlights() {
      safeSend2(
        { type: "GET_HIGHLIGHTS", url: window.location.href },
        (response) => {
          if (!response || !response.ok || !response.highlights) return;
          for (const hl of response.highlights) {
            const success = restoreHighlight(hl);
            if (!success) {
              hl.stale = true;
              restoreHighlight(hl);
            }
          }
        }
      );
    }
    function scrollTo(highlightId) {
      const mark = document.querySelector(`mark[data-aura-hl="${highlightId}"]`);
      if (mark) {
        mark.scrollIntoView({ behavior: "smooth", block: "center" });
        mark.classList.add("aura-hl-flash");
        setTimeout(() => mark.classList.remove("aura-hl-flash"), 1500);
      }
    }
    return {
      init(_container, _store, ext2) {
        _ext = ext2;
        setTimeout(restoreAllHighlights, 1500);
      },
      destroy() {
        hlHost.remove();
        pageHlStyle.remove();
      },
      scrollTo,
      saveHighlight,
      setShowToast(fn) {
        _showToast = fn;
      }
    };
  }
  const GMAIL_HOST = "mail.google.com";
  function createGmail() {
    let _ext;
    let _safeSend = () => {
    };
    const _gmailTrackedComposes = /* @__PURE__ */ new Map();
    function isGmailPage() {
      return window.location.hostname === GMAIL_HOST;
    }
    function extractGmailThreadText() {
      const bodies = document.querySelectorAll(".a3s.aiL");
      if (bodies.length === 0) return "";
      const parts = [];
      bodies.forEach((body) => {
        var _a;
        const text = (_a = body.innerText) == null ? void 0 : _a.trim();
        if (text) parts.push(text);
      });
      return parts.join("\n\n---\n\n").slice(0, 2e4);
    }
    function getComposeBody(composeEl) {
      const ariaLabels = [
        "Message Body",
        // English
        "Nachrichtentext",
        // German
        "Corps du message",
        // French
        "Cuerpo del mensaje",
        // Spanish
        "Corpo da mensagem",
        // Portuguese
        "Corpo del messaggio",
        // Italian
        "Текст сообщения",
        // Russian
        "Mesaj Metni",
        // Turkish
        "メッセージ本文",
        // Japanese
        "메시지 본문",
        // Korean
        "邮件正文",
        // Chinese Simplified
        "نص الرسالة",
        // Arabic
        "Berichttekst",
        // Dutch
        "Treść wiadomości",
        // Polish
        "संदेश का मुख्य भाग",
        // Hindi
        "Mesaj mətni"
        // Azerbaijani
      ];
      const ariaSelector = ariaLabels.map((l) => `div[aria-label="${l}"]`).join(", ");
      const result = composeEl.querySelector(
        ariaSelector + ', div[g_editable="true"][contenteditable="true"], div.editable[contenteditable="true"]'
      );
      if (result) return result;
      return composeEl.querySelector('div[contenteditable="true"][role="textbox"]');
    }
    function getComposeText(composeEl) {
      var _a;
      const body = getComposeBody(composeEl);
      if (!body) return "";
      return ((_a = body.innerText) == null ? void 0 : _a.trim()) || "";
    }
    function escapeHtml(s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
    function setComposeText(composeEl, text) {
      const body = getComposeBody(composeEl);
      if (!body) return;
      body.focus();
      const sel = window.getSelection();
      if (sel) {
        const range = document.createRange();
        range.selectNodeContents(body);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      const success = document.execCommand("insertText", false, text);
      if (!success) {
        body.innerHTML = text.split("\n").map(
          (line) => `<div>${escapeHtml(line) || "<br>"}</div>`
        ).join("");
      }
      body.dispatchEvent(new Event("input", { bubbles: true }));
      body.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function injectGmailAiButton(composeEl) {
      if (_gmailTrackedComposes.has(composeEl)) return;
      const sendBtn = composeEl.querySelector(
        'div[aria-label*="Send"], div[data-tooltip*="Send"], div[aria-label*="Enviar"], div[aria-label*="Envoyer"], div[aria-label*="Senden"], div[aria-label*="Отправить"]'
      );
      const toolbarRow = composeEl.querySelector(
        ".btC, .bAK, tr.btC, .IZ"
      );
      const insertTarget = (sendBtn == null ? void 0 : sendBtn.parentElement) || toolbarRow;
      if (!insertTarget) return;
      const buttonHost = document.createElement("div");
      buttonHost.className = "aura-gmail-ai-host";
      Object.assign(buttonHost.style, {
        display: "inline-flex",
        alignItems: "center",
        verticalAlign: "middle",
        marginLeft: "8px",
        position: "relative",
        zIndex: "1"
      });
      const gmailShadow = buttonHost.attachShadow({ mode: "closed" });
      const gmailStyle = document.createElement("style");
      gmailStyle.textContent = `
      @keyframes gmail-aura-in {
        from { opacity: 0; transform: scale(0.85); }
        to   { opacity: 1; transform: scale(1); }
      }
      @keyframes gmail-aura-spin {
        to { transform: rotate(360deg); }
      }
      @keyframes gmail-aura-menu-in {
        from { opacity: 0; transform: translateY(4px) scale(0.95); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
      }

      :host {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      }

      .gmail-ai-btn {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 12px;
        border-radius: 18px;
        border: 1px solid rgba(124, 58, 237, 0.35);
        background: rgba(124, 58, 237, 0.08);
        color: #7c3aed;
        font-size: 12px;
        font-weight: 600;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        transition: all 0.15s ease;
        animation: gmail-aura-in 0.25s ease forwards;
        line-height: 1.4;
        letter-spacing: 0.01em;
      }
      .gmail-ai-btn:hover {
        background: rgba(124, 58, 237, 0.15);
        border-color: rgba(124, 58, 237, 0.5);
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.15);
      }
      .gmail-ai-btn:active {
        transform: scale(0.97);
      }
      .gmail-ai-btn .sparkle {
        font-size: 13px;
        line-height: 1;
      }

      .gmail-ai-menu {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 0;
        background: rgba(10, 8, 24, 0.94);
        backdrop-filter: blur(20px) saturate(1.5);
        -webkit-backdrop-filter: blur(20px) saturate(1.5);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-radius: 10px;
        padding: 4px;
        min-width: 180px;
        box-shadow: 0 -8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.05) inset;
        animation: gmail-aura-menu-in 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        z-index: 10000;
      }

      .gmail-ai-menu-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 7px;
        background: transparent;
        border: none;
        color: rgba(226, 232, 240, 0.92);
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        cursor: pointer;
        white-space: nowrap;
        width: 100%;
        text-align: left;
        transition: background 0.12s, color 0.12s;
        line-height: 1;
        box-sizing: border-box;
      }
      .gmail-ai-menu-item:hover {
        background: rgba(124, 58, 237, 0.25);
        color: #fff;
      }
      .gmail-ai-menu-item:active {
        background: rgba(124, 58, 237, 0.4);
      }
      .gmail-ai-menu-item .item-icon {
        font-size: 14px;
        width: 18px;
        text-align: center;
        flex-shrink: 0;
      }
      .gmail-ai-menu-item.loading {
        opacity: 0.55;
        pointer-events: none;
      }

      .gmail-ai-sep {
        height: 1px;
        background: rgba(255,255,255,0.08);
        margin: 3px 8px;
      }

      .gmail-ai-sub {
        padding: 2px 0 2px 4px;
      }
      .gmail-ai-sub .gmail-ai-menu-item {
        font-size: 11.5px;
        padding: 6px 12px 6px 26px;
      }

      .gmail-ai-spinner {
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 2px solid rgba(124, 58, 237, 0.3);
        border-top-color: rgba(160, 148, 210, 0.9);
        border-radius: 50%;
        animation: gmail-aura-spin 0.6s linear infinite;
        flex-shrink: 0;
      }

      .gmail-ai-toast {
        position: absolute;
        bottom: calc(100% + 6px);
        left: 50%;
        transform: translateX(-50%);
        background: rgba(5, 150, 105, 0.92);
        backdrop-filter: blur(12px);
        color: #fff;
        font-size: 11.5px;
        font-weight: 500;
        font-family: inherit;
        padding: 5px 12px;
        border-radius: 6px;
        white-space: nowrap;
        pointer-events: none;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: gmail-aura-menu-in 0.15s ease forwards;
        z-index: 10001;
      }
    `;
      gmailShadow.appendChild(gmailStyle);
      const gmailContainer = document.createElement("div");
      gmailContainer.style.position = "relative";
      gmailContainer.style.display = "inline-flex";
      gmailContainer.style.alignItems = "center";
      gmailShadow.appendChild(gmailContainer);
      const aiBtn = document.createElement("button");
      aiBtn.className = "gmail-ai-btn";
      aiBtn.innerHTML = `<span class="sparkle">✦</span> AI`;
      gmailContainer.appendChild(aiBtn);
      let gmailMenuEl = null;
      let gmailTranslateSubEl = null;
      let gmailToastEl = null;
      let gmailToastTimer = null;
      function showGmailToast(msg, duration = 2500) {
        if (gmailToastEl) gmailToastEl.remove();
        if (gmailToastTimer) clearTimeout(gmailToastTimer);
        gmailToastEl = document.createElement("div");
        gmailToastEl.className = "gmail-ai-toast";
        gmailToastEl.textContent = msg;
        gmailContainer.appendChild(gmailToastEl);
        gmailToastTimer = setTimeout(() => {
          if (gmailToastEl) {
            gmailToastEl.remove();
            gmailToastEl = null;
          }
          gmailToastTimer = null;
        }, duration);
      }
      function removeGmailMenu() {
        if (gmailMenuEl) {
          gmailMenuEl.remove();
          gmailMenuEl = null;
        }
        gmailTranslateSubEl = null;
      }
      function setGmailMenuLoading(loading) {
        if (!gmailMenuEl) return;
        gmailMenuEl.querySelectorAll(".gmail-ai-menu-item").forEach((item) => {
          item.classList.add("loading");
        });
      }
      function executeGmailAction(action, language) {
        const composeText = getComposeText(composeEl);
        const threadText = extractGmailThreadText();
        if (action === "draft_reply" && !composeText && !threadText) {
          showGmailToast("No email thread found", 3e3);
          removeGmailMenu();
          return;
        }
        if (action !== "draft_reply" && !composeText) {
          showGmailToast("Compose body is empty", 3e3);
          removeGmailMenu();
          return;
        }
        setGmailMenuLoading();
        const outMsg = {
          type: "QUICK_ACTION",
          action,
          text: composeText || "(empty — draft a new reply)",
          ...action === "draft_reply" ? { threadContext: threadText } : {},
          ...language ? { language } : {}
        };
        _safeSend(outMsg, (response) => {
          if (response && response.ok && response.result) {
            setComposeText(composeEl, response.result);
            showGmailToast("Updated by AURA");
          } else {
            showGmailToast((response == null ? void 0 : response.error) || "Action failed", 3e3);
          }
          removeGmailMenu();
        });
      }
      const GMAIL_ACTIONS = [
        { icon: "✍️", label: "Draft reply", action: "draft_reply" },
        { icon: "✨", label: "Improve", action: "improve" },
        { icon: "🏢", label: "Make formal", action: "make_formal", separator: true },
        { icon: "😊", label: "Make casual", action: "make_casual" },
        { icon: "✂️", label: "Shorten", action: "shorten" },
        { icon: "🌐", label: "Translate to...", action: "translate_menu", separator: true }
      ];
      const GMAIL_TRANSLATE_LANGS = ["English", "Spanish", "French", "German", "Chinese"];
      function showGmailMenu() {
        removeGmailMenu();
        gmailMenuEl = document.createElement("div");
        gmailMenuEl.className = "gmail-ai-menu";
        GMAIL_ACTIONS.forEach((item) => {
          if (item.separator) {
            const sep = document.createElement("div");
            sep.className = "gmail-ai-sep";
            gmailMenuEl.appendChild(sep);
          }
          const btn = document.createElement("button");
          btn.className = "gmail-ai-menu-item";
          btn.innerHTML = `<span class="item-icon">${item.icon}</span><span>${item.label}</span>`;
          btn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (item.action === "translate_menu") {
              toggleGmailTranslateSub(btn);
            } else {
              executeGmailAction(item.action);
            }
          });
          gmailMenuEl.appendChild(btn);
        });
        gmailContainer.appendChild(gmailMenuEl);
      }
      function toggleGmailTranslateSub(anchor) {
        if (gmailTranslateSubEl) {
          gmailTranslateSubEl.remove();
          gmailTranslateSubEl = null;
          return;
        }
        gmailTranslateSubEl = document.createElement("div");
        gmailTranslateSubEl.className = "gmail-ai-sub";
        GMAIL_TRANSLATE_LANGS.forEach((lang) => {
          const item = document.createElement("button");
          item.className = "gmail-ai-menu-item";
          item.textContent = lang;
          item.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            executeGmailAction("gmail_translate", lang);
          });
          gmailTranslateSubEl.appendChild(item);
        });
        if (gmailMenuEl && anchor.parentNode === gmailMenuEl) {
          anchor.after(gmailTranslateSubEl);
        }
      }
      aiBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (gmailMenuEl) {
          removeGmailMenu();
        } else {
          showGmailMenu();
        }
      });
      const outsideClickHandler = (e) => {
        if (!gmailMenuEl) return;
        const path = e.composedPath();
        if (!path.includes(buttonHost)) {
          removeGmailMenu();
        }
      };
      document.addEventListener("mousedown", outsideClickHandler, true);
      if (sendBtn == null ? void 0 : sendBtn.parentElement) {
        sendBtn.parentElement.insertBefore(buttonHost, sendBtn.nextSibling);
      } else if (toolbarRow) {
        toolbarRow.appendChild(buttonHost);
      }
      const composeObserver = new MutationObserver(() => {
        if (!document.body.contains(composeEl)) {
          composeObserver.disconnect();
          document.removeEventListener("mousedown", outsideClickHandler, true);
          buttonHost.remove();
          _gmailTrackedComposes.delete(composeEl);
        }
      });
      composeObserver.observe(document.body, { childList: true, subtree: true });
      _gmailTrackedComposes.set(composeEl, {
        composeEl,
        buttonHost,
        shadow: gmailShadow,
        observer: composeObserver,
        outsideHandler: outsideClickHandler
      });
    }
    function scanGmailComposeWindows() {
      const composeSelectors = [
        'div[role="dialog"]',
        // Popup compose / reply
        "div.ip.iq",
        // Inline reply
        "div.nH.nn"
        // Another compose variant
      ];
      composeSelectors.forEach((sel) => {
        document.querySelectorAll(sel).forEach((el) => {
          const body = getComposeBody(el);
          if (!body) return;
          if (_gmailTrackedComposes.has(el)) return;
          injectGmailAiButton(el);
        });
      });
    }
    function initGmailIntegration() {
      if (!isGmailPage()) return;
      scanGmailComposeWindows();
      const gmailObserver = new MutationObserver((mutations) => {
        var _a, _b, _c;
        let shouldScan = false;
        for (const mutation of mutations) {
          if (mutation.addedNodes.length > 0) {
            for (const node of mutation.addedNodes) {
              if (node.nodeType !== Node.ELEMENT_NODE) continue;
              const el = node;
              if (((_a = el.matches) == null ? void 0 : _a.call(el, 'div[role="dialog"]')) || ((_b = el.querySelector) == null ? void 0 : _b.call(el, 'div[role="dialog"]')) || ((_c = el.querySelector) == null ? void 0 : _c.call(el, 'div[contenteditable="true"]'))) {
                shouldScan = true;
                break;
              }
            }
          }
          if (shouldScan) break;
        }
        if (shouldScan) {
          setTimeout(scanGmailComposeWindows, 300);
        }
      });
      const gmailRoot = document.querySelector('div[role="main"]') || document.body;
      gmailObserver.observe(gmailRoot, { childList: true, subtree: true });
      let _gmailScanInterval = null;
      _gmailScanInterval = setInterval(() => {
        if (!isGmailPage()) {
          if (_gmailScanInterval) clearInterval(_gmailScanInterval);
          return;
        }
        scanGmailComposeWindows();
      }, 3e3);
    }
    return {
      init(_container, _store, ext2) {
        _ext = ext2;
        _safeSend = (msg, cb) => {
          try {
            if (cb) {
              _ext.runtime.sendMessage(msg, cb);
            } else {
              _ext.runtime.sendMessage(msg);
            }
          } catch (_e) {
          }
        };
        initGmailIntegration();
      },
      destroy() {
        for (const tracker of _gmailTrackedComposes.values()) {
          tracker.observer.disconnect();
          document.removeEventListener("mousedown", tracker.outsideHandler, true);
          tracker.buttonHost.remove();
        }
        _gmailTrackedComposes.clear();
      }
    };
  }
  const CAPTURE_CSS_PROPS = [
    "display",
    "position",
    "flex-direction",
    "align-items",
    "justify-content",
    "gap",
    "flex-wrap",
    "flex",
    "flex-grow",
    "flex-shrink",
    "width",
    "height",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "border",
    "border-radius",
    "border-color",
    "border-width",
    "border-style",
    "background",
    "background-color",
    "background-image",
    "background-size",
    "color",
    "font-size",
    "font-weight",
    "font-family",
    "line-height",
    "letter-spacing",
    "text-align",
    "text-decoration",
    "text-transform",
    "box-shadow",
    "opacity",
    "overflow",
    "z-index",
    "grid-template-columns",
    "grid-template-rows",
    "grid-gap",
    "transform",
    "transition"
  ];
  function createCapture() {
    let _ext;
    const captureHost = document.createElement("div");
    captureHost.id = "aura-capture-host";
    Object.assign(captureHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "0",
      height: "0",
      zIndex: "2147483647",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(captureHost);
    const captureShadow = captureHost.attachShadow({ mode: "closed" });
    const captureStyle = document.createElement("style");
    captureStyle.textContent = `
    @keyframes capture-pulse {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }
    .capture-overlay {
      position: fixed;
      pointer-events: none;
      border: 2px solid rgba(124, 58, 237, 0.8);
      background: rgba(124, 58, 237, 0.08);
      border-radius: 3px;
      transition: top 0.05s ease, left 0.05s ease, width 0.05s ease, height 0.05s ease;
      box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.2),
                  0 0 20px rgba(124, 58, 237, 0.15),
                  inset 0 0 20px rgba(124, 58, 237, 0.05);
      z-index: 2147483647;
    }
    .capture-tooltip {
      position: fixed;
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.35);
      border-radius: 6px;
      padding: 4px 10px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 11px;
      color: rgba(226, 232, 240, 0.9);
      white-space: nowrap;
      pointer-events: none;
      z-index: 2147483647;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .capture-tooltip .tag {
      color: #a78bfa;
      font-weight: 600;
    }
    .capture-tooltip .cls {
      color: rgba(167, 139, 250, 0.6);
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .capture-tooltip .dims {
      color: rgba(226, 232, 240, 0.5);
      font-size: 10px;
    }
    .capture-banner {
      position: fixed;
      top: 8px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(10, 8, 24, 0.92);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid rgba(124, 58, 237, 0.4);
      border-radius: 10px;
      padding: 8px 16px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif;
      font-size: 12px;
      color: rgba(226, 232, 240, 0.9);
      z-index: 2147483647;
      pointer-events: none;
      box-shadow: 0 8px 32px rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      gap: 8px;
      animation: capture-pulse 2s ease-in-out infinite;
    }
    .capture-banner .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #a78bfa;
      box-shadow: 0 0 8px rgba(124, 58, 237, 0.6);
    }
  `;
    captureShadow.appendChild(captureStyle);
    const captureContainer = document.createElement("div");
    captureShadow.appendChild(captureContainer);
    let _captureActive = false;
    let _captureOverlay = null;
    let _captureTooltip = null;
    let _captureBanner = null;
    let _captureHoveredEl = null;
    function extractComputedStyles(el) {
      const styles = window.getComputedStyle(el);
      const result = {};
      for (const prop of CAPTURE_CSS_PROPS) {
        const val = styles.getPropertyValue(prop);
        if (val && val !== "none" && val !== "normal" && val !== "auto" && val !== "0px" && val !== "rgba(0, 0, 0, 0)") {
          result[prop] = val;
        }
      }
      return result;
    }
    function buildCssSelector(el) {
      const tag = el.tagName.toLowerCase();
      const cls = el.className && typeof el.className === "string" ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
      return tag + cls;
    }
    function captureElementData(el) {
      const rect = el.getBoundingClientRect();
      const styles = window.getComputedStyle(el);
      const html = el.outerHTML;
      const cssMap = {};
      cssMap[buildCssSelector(el)] = extractComputedStyles(el);
      const children = el.querySelectorAll("*");
      let count = 0;
      for (const child of children) {
        if (count >= 50) break;
        const childStyles = extractComputedStyles(child);
        if (Object.keys(childStyles).length > 0) {
          const selector = buildCssSelector(child);
          const key = cssMap[selector] ? `${selector}:nth(${count})` : selector;
          cssMap[key] = childStyles;
        }
        count++;
      }
      return {
        html,
        css: cssMap,
        dimensions: {
          width: rect.width,
          height: rect.height,
          padding: `${styles.paddingTop} ${styles.paddingRight} ${styles.paddingBottom} ${styles.paddingLeft}`,
          margin: `${styles.marginTop} ${styles.marginRight} ${styles.marginBottom} ${styles.marginLeft}`
        },
        textContent: (el.textContent || "").slice(0, 2e3).trim(),
        tagName: el.tagName.toLowerCase(),
        className: (typeof el.className === "string" ? el.className : "").trim()
      };
    }
    function start() {
      if (_captureActive) return;
      _captureActive = true;
      _captureBanner = document.createElement("div");
      _captureBanner.className = "capture-banner";
      _captureBanner.innerHTML = '<span class="dot"></span> AURA Capture Mode — Click any element • Esc to exit';
      captureContainer.appendChild(_captureBanner);
      _captureOverlay = document.createElement("div");
      _captureOverlay.className = "capture-overlay";
      _captureOverlay.style.display = "none";
      captureContainer.appendChild(_captureOverlay);
      _captureTooltip = document.createElement("div");
      _captureTooltip.className = "capture-tooltip";
      _captureTooltip.style.display = "none";
      captureContainer.appendChild(_captureTooltip);
      captureHost.style.width = "100vw";
      captureHost.style.height = "100vh";
      document.addEventListener("mousemove", onCaptureMouseMove, true);
      document.addEventListener("click", onCaptureClick, true);
      document.addEventListener("keydown", onCaptureKeydown, true);
    }
    function stop() {
      if (!_captureActive) return;
      _captureActive = false;
      _captureHoveredEl = null;
      if (_captureOverlay) {
        _captureOverlay.remove();
        _captureOverlay = null;
      }
      if (_captureTooltip) {
        _captureTooltip.remove();
        _captureTooltip = null;
      }
      if (_captureBanner) {
        _captureBanner.remove();
        _captureBanner = null;
      }
      captureHost.style.width = "0";
      captureHost.style.height = "0";
      document.removeEventListener("mousemove", onCaptureMouseMove, true);
      document.removeEventListener("click", onCaptureClick, true);
      document.removeEventListener("keydown", onCaptureKeydown, true);
      try {
        _ext.runtime.sendMessage({ type: "OPEN_PANEL", panel: "capture" });
      } catch (_e) {
      }
    }
    function onCaptureMouseMove(e) {
      if (!_captureActive) return;
      const elements = document.elementsFromPoint(e.clientX, e.clientY);
      let target = null;
      for (const el of elements) {
        if (el === captureHost || captureHost.contains(el)) continue;
        if (el.id === "aura-host" || el.id === "aura-dock-shadow" || el.id === "aura-quick-action-host" || el.id === "aura-highlight-host" || el.id === "aura-img-toolbar-host" || el.id === "aura-capture-host") continue;
        if (el === document.documentElement || el === document.body) continue;
        target = el;
        break;
      }
      if (!target) {
        if (_captureOverlay) _captureOverlay.style.display = "none";
        if (_captureTooltip) _captureTooltip.style.display = "none";
        _captureHoveredEl = null;
        return;
      }
      _captureHoveredEl = target;
      const rect = target.getBoundingClientRect();
      if (_captureOverlay) {
        _captureOverlay.style.display = "block";
        _captureOverlay.style.top = rect.top + "px";
        _captureOverlay.style.left = rect.left + "px";
        _captureOverlay.style.width = rect.width + "px";
        _captureOverlay.style.height = rect.height + "px";
      }
      if (_captureTooltip) {
        const tag = target.tagName.toLowerCase();
        const cls = target.className && typeof target.className === "string" ? target.className.trim().split(/\s+/).slice(0, 3).join(" ") : "";
        const w = Math.round(rect.width);
        const h = Math.round(rect.height);
        _captureTooltip.textContent = "";
        const tagSpan = document.createElement("span");
        tagSpan.className = "tag";
        tagSpan.textContent = `<${tag}>`;
        _captureTooltip.appendChild(tagSpan);
        if (cls) {
          const clsSpan = document.createElement("span");
          clsSpan.className = "cls";
          clsSpan.textContent = "." + cls.split(" ").join(".");
          _captureTooltip.appendChild(clsSpan);
        }
        const dimsSpan = document.createElement("span");
        dimsSpan.className = "dims";
        dimsSpan.textContent = `${w}x${h}`;
        _captureTooltip.appendChild(dimsSpan);
        let tooltipTop = rect.top - 30;
        if (tooltipTop < 4) tooltipTop = rect.bottom + 6;
        let tooltipLeft = rect.left;
        if (tooltipLeft < 4) tooltipLeft = 4;
        _captureTooltip.style.display = "flex";
        _captureTooltip.style.top = tooltipTop + "px";
        _captureTooltip.style.left = tooltipLeft + "px";
      }
    }
    function onCaptureClick(e) {
      if (!_captureActive || !_captureHoveredEl) return;
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      const el = _captureHoveredEl;
      const rect = el.getBoundingClientRect();
      const data = captureElementData(el);
      try {
        _ext.runtime.sendMessage(
          {
            type: "CAPTURE_ELEMENT",
            rect: {
              x: Math.round(rect.left),
              y: Math.round(rect.top),
              w: Math.round(rect.width),
              h: Math.round(rect.height)
            },
            elementData: data
          },
          (_response) => {
          }
        );
      } catch (_e) {
      }
      stop();
    }
    function onCaptureKeydown(e) {
      if (e.key === "Escape" && _captureActive) {
        e.preventDefault();
        e.stopPropagation();
        stop();
        try {
          _ext.runtime.sendMessage({ type: "CAPTURE_MODE_EXITED" }).catch(() => {
          });
        } catch (_e) {
        }
      }
    }
    return {
      init(_container, _store, ext2) {
        _ext = ext2;
      },
      destroy() {
        if (_captureActive) stop();
        captureHost.remove();
      },
      start,
      stop
    };
  }
  const LP_CACHE_MAX = 50;
  function createLinkPreview() {
    let _ext;
    const _linkPreviewCache = /* @__PURE__ */ new Map();
    function lpCacheSet(cacheUrl, cacheData) {
      if (_linkPreviewCache.size >= LP_CACHE_MAX) {
        const oldest = _linkPreviewCache.keys().next().value;
        if (oldest) _linkPreviewCache.delete(oldest);
      }
      _linkPreviewCache.set(cacheUrl, cacheData);
    }
    function lpCacheGet(cacheUrl) {
      const d = _linkPreviewCache.get(cacheUrl);
      if (d) {
        _linkPreviewCache.delete(cacheUrl);
        _linkPreviewCache.set(cacheUrl, d);
      }
      return d;
    }
    const lpHost = document.createElement("div");
    lpHost.id = "aura-link-preview-host";
    Object.assign(lpHost.style, { position: "fixed", top: "0", left: "0", zIndex: "2147483646", pointerEvents: "none" });
    document.documentElement.appendChild(lpHost);
    const lpShadow = lpHost.attachShadow({ mode: "closed" });
    const lpCss = document.createElement("style");
    lpCss.textContent = [
      "@keyframes lp-in { from { opacity:0; transform:translateY(4px) scale(0.96); } to { opacity:1; transform:translateY(0) scale(1); } }",
      "@keyframes lp-shimmer { 0% { background-position:-200px 0; } 100% { background-position:200px 0; } }",
      '.lp-popup { position:fixed; width:320px; max-height:280px; background:rgba(10,8,24,0.92); backdrop-filter:blur(20px) saturate(1.5); -webkit-backdrop-filter:blur(20px) saturate(1.5); border:1px solid rgba(124,58,237,0.25); border-radius:12px; padding:14px 16px 12px; pointer-events:auto; animation:lp-in 0.2s cubic-bezier(0.16,1,0.3,1) forwards; box-shadow:0 8px 32px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.05) inset; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Inter",system-ui,sans-serif; box-sizing:border-box; overflow:hidden; display:flex; flex-direction:column; gap:8px; }',
      ".lp-domain { display:inline-block; background:rgba(124,58,237,0.15); border:1px solid rgba(124,58,237,0.25); border-radius:4px; padding:2px 7px; font-size:10.5px; font-weight:600; color:rgba(160,148,210,0.9); letter-spacing:0.3px; max-width:fit-content; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }",
      ".lp-title { font-size:13px; font-weight:600; color:rgba(226,232,240,0.95); line-height:1.35; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",
      ".lp-description { font-size:12px; font-weight:400; color:rgba(226,232,240,0.65); line-height:1.45; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; margin:0; }",
      ".lp-shimmer { height:12px; border-radius:4px; background:linear-gradient(90deg,rgba(124,58,237,0.08) 25%,rgba(124,58,237,0.18) 50%,rgba(124,58,237,0.08) 75%); background-size:400px 100%; animation:lp-shimmer 1.5s infinite linear; }",
      ".lp-shimmer.short { width:60%; } .lp-shimmer.long { width:90%; } .lp-shimmer+.lp-shimmer { margin-top:6px; }",
      ".lp-loading-label { font-size:11px; color:rgba(160,148,210,0.5); margin-bottom:4px; }",
      ".lp-actions { display:flex; gap:6px; margin-top:4px; padding-top:8px; border-top:1px solid rgba(255,255,255,0.06); }",
      ".lp-btn { background:rgba(124,58,237,0.12); border:1px solid rgba(124,58,237,0.2); border-radius:6px; padding:4px 10px; font-size:11px; font-weight:500; font-family:inherit; color:rgba(200,180,255,0.9); cursor:pointer; transition:background 0.15s,border-color 0.15s,color 0.15s; white-space:nowrap; }",
      ".lp-btn:hover { background:rgba(124,58,237,0.25); border-color:rgba(124,58,237,0.4); color:#fff; }",
      ".lp-btn:active { background:rgba(124,58,237,0.35); }"
    ].join("\n");
    lpShadow.appendChild(lpCss);
    const lpBox = document.createElement("div");
    lpShadow.appendChild(lpBox);
    let _lpPopup = null;
    let _lpHoverTmr = null;
    let _lpDismissTmr = null;
    let _lpCurLink = null;
    let _lpMouseIsDown = false;
    const onMouseDown = () => {
      _lpMouseIsDown = true;
    };
    const onMouseUp = () => {
      _lpMouseIsDown = false;
    };
    function lpIsExternal(a) {
      try {
        return new URL(a.href, location.href).hostname !== location.hostname;
      } catch {
        return false;
      }
    }
    function lpShouldShow(a) {
      const h = a.href || "";
      if (!h.startsWith("http://") && !h.startsWith("https://")) return false;
      try {
        const u = new URL(h, location.href);
        if (u.hostname === location.hostname && u.pathname === location.pathname && u.hash) return false;
      } catch {
        return false;
      }
      if ((a.textContent || "").trim().length < 10) return false;
      return lpIsExternal(a);
    }
    function lpRemove() {
      if (_lpPopup) {
        _lpPopup.remove();
        _lpPopup = null;
      }
      _lpCurLink = null;
    }
    function lpCancelTimers() {
      if (_lpHoverTmr) {
        clearTimeout(_lpHoverTmr);
        _lpHoverTmr = null;
      }
      if (_lpDismissTmr) {
        clearTimeout(_lpDismissTmr);
        _lpDismissTmr = null;
      }
    }
    function lpStartDismiss() {
      if (_lpDismissTmr) clearTimeout(_lpDismissTmr);
      _lpDismissTmr = setTimeout(() => {
        lpRemove();
        _lpDismissTmr = null;
      }, 300);
    }
    function lpCancelDismiss() {
      if (_lpDismissTmr) {
        clearTimeout(_lpDismissTmr);
        _lpDismissTmr = null;
      }
    }
    function lpPosition(a) {
      if (!_lpPopup) return;
      const r = a.getBoundingClientRect();
      _lpPopup.style.visibility = "hidden";
      _lpPopup.style.display = "flex";
      const ph = _lpPopup.offsetHeight || 180;
      _lpPopup.style.visibility = "";
      let l = r.left + r.width / 2 - 160;
      if (l < 8) l = 8;
      if (l + 320 > window.innerWidth - 8) l = window.innerWidth - 328;
      let t = r.bottom + 8;
      if (t + ph > window.innerHeight - 8) {
        t = r.top - ph - 8;
        if (t < 8) t = 8;
      }
      _lpPopup.style.top = Math.round(t) + "px";
      _lpPopup.style.left = Math.round(l) + "px";
    }
    function lpUpdate(lw, te, d) {
      lw.innerHTML = "";
      lw.style.display = "none";
      if (d.title && d.title !== te.textContent) te.textContent = d.title;
      if (d.description) {
        const de = document.createElement("div");
        de.className = "lp-description";
        de.textContent = d.description;
        te.after(de);
      }
      if (_lpPopup && _lpCurLink) lpPosition(_lpCurLink);
    }
    function lpBuild(a, href) {
      lpRemove();
      _lpCurLink = a;
      let dom = "";
      try {
        dom = new URL(href).hostname;
      } catch {
        dom = href;
      }
      const txt = (a.textContent || "").trim();
      _lpPopup = document.createElement("div");
      _lpPopup.className = "lp-popup";
      const dEl = document.createElement("div");
      dEl.className = "lp-domain";
      dEl.textContent = dom;
      _lpPopup.appendChild(dEl);
      const tEl = document.createElement("div");
      tEl.className = "lp-title";
      tEl.textContent = txt;
      _lpPopup.appendChild(tEl);
      const lw = document.createElement("div");
      const ll = document.createElement("div");
      ll.className = "lp-loading-label";
      ll.textContent = "Loading preview…";
      const s1 = document.createElement("div");
      s1.className = "lp-shimmer long";
      const s2 = document.createElement("div");
      s2.className = "lp-shimmer short";
      lw.appendChild(ll);
      lw.appendChild(s1);
      lw.appendChild(s2);
      _lpPopup.appendChild(lw);
      const acts = document.createElement("div");
      acts.className = "lp-actions";
      const ob = document.createElement("button");
      ob.className = "lp-btn";
      ob.textContent = "Open";
      ob.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        window.open(href, "_blank", "noopener");
        lpRemove();
      });
      const sb = document.createElement("button");
      sb.className = "lp-btn";
      sb.textContent = "Summarize in AURA";
      sb.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        try {
          _ext.runtime.sendMessage({ type: "OPEN_WITH_TEXT", action: "summarize", text: "Summarize this page: " + href, url: href, title: txt });
        } catch (_e) {
        }
        lpRemove();
      });
      acts.appendChild(ob);
      acts.appendChild(sb);
      _lpPopup.appendChild(acts);
      _lpPopup.addEventListener("mouseenter", lpCancelDismiss);
      _lpPopup.addEventListener("mouseleave", lpStartDismiss);
      lpBox.appendChild(_lpPopup);
      lpPosition(a);
      const c = lpCacheGet(href);
      if (c) {
        lpUpdate(lw, tEl, c);
        return;
      }
      try {
        _ext.runtime.sendMessage({ type: "LINK_PREVIEW", url: href }, (rsp) => {
          if (_ext.runtime.lastError || !rsp) return;
          if (!_lpPopup || _lpCurLink !== a) return;
          const pd = { title: rsp.title || txt, description: rsp.description || "", domain: rsp.domain || dom };
          lpCacheSet(href, pd);
          lpUpdate(lw, tEl, pd);
        });
      } catch {
      }
    }
    const onMouseOver = (me) => {
      if (_lpMouseIsDown) return;
      const a = me.target.closest("a");
      if (!a || !lpShouldShow(a)) return;
      if (_lpCurLink === a && _lpPopup) {
        lpCancelDismiss();
        return;
      }
      lpCancelTimers();
      _lpHoverTmr = setTimeout(() => {
        if (_lpMouseIsDown) return;
        lpBuild(a, a.href);
        _lpHoverTmr = null;
      }, 800);
    };
    const onMouseOut = (me) => {
      const a = me.target.closest("a");
      if (a && a === _lpCurLink) {
        const rel = me.relatedTarget;
        if (rel && lpHost.contains(rel)) return;
        lpStartDismiss();
      }
      if (a && _lpHoverTmr) lpCancelTimers();
    };
    const onScroll = () => {
      if (_lpPopup && _lpCurLink) {
        const r = _lpCurLink.getBoundingClientRect();
        if (r.bottom < 0 || r.top > window.innerHeight) {
          lpCancelTimers();
          lpRemove();
        } else {
          lpPosition(_lpCurLink);
        }
      }
    };
    return {
      init(_container, _store, ext2) {
        _ext = ext2;
        document.addEventListener("mousedown", onMouseDown, true);
        document.addEventListener("mouseup", onMouseUp, true);
        document.addEventListener("mouseover", onMouseOver, true);
        document.addEventListener("mouseout", onMouseOut, true);
        window.addEventListener("scroll", onScroll, { passive: true });
      },
      destroy() {
        document.removeEventListener("mousedown", onMouseDown, true);
        document.removeEventListener("mouseup", onMouseUp, true);
        document.removeEventListener("mouseover", onMouseOver, true);
        document.removeEventListener("mouseout", onMouseOut, true);
        window.removeEventListener("scroll", onScroll);
        lpCancelTimers();
        lpRemove();
        lpHost.remove();
      }
    };
  }
  function bestSelector(el) {
    var _a;
    if (el.id) return "#" + CSS.escape(el.id);
    const al = el.getAttribute("aria-label");
    if (al) return `[aria-label="${al}"]`;
    const path = [];
    let cur = el;
    for (let i = 0; i < 4 && cur && cur !== document.body; i++, cur = cur.parentElement) {
      const s = cur.tagName.toLowerCase();
      if (cur.id) {
        path.unshift("#" + CSS.escape(cur.id));
        break;
      }
      const siblings = [...((_a = cur.parentElement) == null ? void 0 : _a.children) || []];
      const idx = siblings.indexOf(cur) + 1;
      path.unshift(s + ":nth-child(" + idx + ")");
    }
    return path.join(">");
  }
  function serializeDOM() {
    const els = [];
    const nodes = document.querySelectorAll(
      'a,button,input,textarea,select,[role="button"],[onclick]'
    );
    let idx = 0;
    for (const el of nodes) {
      if (els.length >= 80) break;
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      const htmlEl = el;
      const inputEl = el;
      els.push({
        index: idx++,
        type: el.tagName.toLowerCase(),
        text: (htmlEl.innerText || inputEl.value || inputEl.placeholder || htmlEl.title || "").slice(0, 80).trim(),
        selector: bestSelector(htmlEl),
        href: el.href || ""
      });
    }
    return els;
  }
  function execAction(action) {
    if (action.action === "scroll") {
      window.scrollBy(0, action.amount || 300);
      return { ok: true };
    }
    let el;
    try {
      el = document.querySelector(action.selector);
    } catch (e) {
      return { ok: false, error: "Invalid selector: " + action.selector };
    }
    if (!el) return { ok: false, error: "Element not found: " + action.selector };
    if (action.action === "click") {
      el.click();
      return { ok: true };
    }
    if (action.action === "type") {
      el.focus();
      el.value = action.text || "";
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return { ok: true };
    }
    if (action.action === "selectOption") {
      if (el.tagName.toLowerCase() !== "select") return { ok: false, error: "Element is not a <select>" };
      const selectEl = el;
      const opt = [...selectEl.options].find(
        (o) => o.value === action.value || o.text === action.value
      );
      if (!opt) return { ok: false, error: "Option not found: " + action.value };
      selectEl.value = opt.value;
      selectEl.dispatchEvent(new Event("change", { bubbles: true }));
      return { ok: true };
    }
    return { ok: false, error: "Unknown action: " + action.action };
  }
  function showOcrOverlay(dataUrl, sendResponse) {
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "100vw",
      height: "100vh",
      zIndex: "2147483646",
      cursor: "crosshair",
      background: "rgba(0,0,0,0.4)"
    });
    const img = new Image();
    img.src = dataUrl;
    img.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;opacity:0.7;pointer-events:none;";
    overlay.appendChild(img);
    const canvas = document.createElement("canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    Object.assign(canvas.style, {
      position: "absolute",
      top: "0",
      left: "0",
      width: "100%",
      height: "100%"
    });
    overlay.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const hint = document.createElement("div");
    Object.assign(hint.style, {
      position: "fixed",
      top: "12px",
      left: "50%",
      transform: "translateX(-50%)",
      background: "rgba(0,0,0,0.75)",
      color: "#fff",
      padding: "6px 14px",
      borderRadius: "6px",
      fontSize: "13px",
      pointerEvents: "none"
    });
    hint.textContent = "Drag to select region • Press Esc to cancel";
    overlay.appendChild(hint);
    document.body.appendChild(overlay);
    let startX = 0;
    let startY = 0;
    let dragging = false;
    const dpr = window.devicePixelRatio || 1;
    function drawRect(x, y, w, h) {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#7c3aed";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = "rgba(124,58,237,0.12)";
      ctx.fillRect(x, y, w, h);
    }
    overlay.addEventListener("mousedown", (e) => {
      startX = e.clientX;
      startY = e.clientY;
      dragging = true;
    });
    overlay.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      drawRect(startX, startY, e.clientX - startX, e.clientY - startY);
    });
    function onEsc(e) {
      if (e.key === "Escape") {
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
        document.removeEventListener("keydown", onEsc);
        sendResponse({ ok: false });
      }
    }
    overlay.addEventListener("mouseup", (e) => {
      dragging = false;
      const x = Math.min(startX, e.clientX);
      const y = Math.min(startY, e.clientY);
      const w = Math.abs(e.clientX - startX);
      const h = Math.abs(e.clientY - startY);
      document.removeEventListener("keydown", onEsc);
      if (document.body.contains(overlay)) document.body.removeChild(overlay);
      if (w < 5 || h < 5) {
        sendResponse({ ok: false });
        return;
      }
      sendResponse({ ok: true, x, y, w, h, dpr });
    });
    document.addEventListener("keydown", onEsc);
    const ocrCleanupObserver = new MutationObserver(() => {
      if (!document.body.contains(overlay)) {
        document.removeEventListener("keydown", onEsc);
        ocrCleanupObserver.disconnect();
      }
    });
    ocrCleanupObserver.observe(document.body, { childList: true });
  }
  const MAX_TEXT_BYTES = 5e4;
  const CONTENT_SELECTORS = [
    "article",
    "main",
    '[role="main"]',
    ".post-content",
    ".article-body",
    ".entry-content",
    ".post-body",
    ".article-content",
    ".story-body",
    ".content-body",
    "#article-body",
    "#content",
    ".markdown-body",
    // GitHub
    ".wiki-content"
  ];
  const JUNK_SELECTORS = [
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    "noscript",
    "iframe",
    ".sidebar",
    ".menu",
    ".nav",
    ".navigation",
    ".cookie",
    ".cookie-banner",
    ".cookie-consent",
    ".popup",
    ".modal",
    ".overlay",
    ".ad",
    ".ads",
    ".advert",
    ".advertisement",
    ".social-share",
    ".share-buttons",
    ".social-buttons",
    ".comments",
    ".comment-section",
    "#comments",
    ".related-posts",
    ".recommended",
    ".newsletter",
    ".subscribe",
    '[role="navigation"]',
    '[role="banner"]',
    '[role="contentinfo"]',
    '[role="complementary"]',
    '[aria-hidden="true"]',
    ".sr-only",
    ".visually-hidden"
  ];
  function findContentRoot() {
    for (const sel of CONTENT_SELECTORS) {
      const el = document.querySelector(sel);
      if (el && el.textContent && el.textContent.trim().length > 200) {
        return el;
      }
    }
    return document.body;
  }
  function cleanClone(root) {
    const clone = root.cloneNode(true);
    for (const sel of JUNK_SELECTORS) {
      clone.querySelectorAll(sel).forEach((el) => el.remove());
    }
    return clone;
  }
  function domToStructuredText(root) {
    const parts = [];
    const BLOCK_TAGS = /* @__PURE__ */ new Set([
      "P",
      "DIV",
      "SECTION",
      "ARTICLE",
      "BLOCKQUOTE",
      "PRE",
      "H1",
      "H2",
      "H3",
      "H4",
      "H5",
      "H6",
      "UL",
      "OL",
      "LI",
      "TABLE",
      "TR",
      "DT",
      "DD",
      "FIGURE",
      "FIGCAPTION",
      "HR",
      "BR"
    ]);
    function walk(node) {
      var _a;
      if (node.nodeType === Node.TEXT_NODE) {
        const text = (node.textContent || "").replace(/\s+/g, " ");
        if (text.trim()) parts.push(text);
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;
      const el = node;
      const tag = el.tagName;
      if (el.hasAttribute("hidden") || ((_a = el.style) == null ? void 0 : _a.display) === "none") return;
      if (/^H[1-6]$/.test(tag)) {
        const level = parseInt(tag[1]);
        const prefix = "#".repeat(Math.min(level, 3)) + " ";
        const headingText = (el.textContent || "").trim();
        if (headingText) {
          parts.push("\n\n" + prefix + headingText + "\n");
        }
        return;
      }
      if (tag === "LI") {
        const text = (el.textContent || "").trim();
        if (text) {
          parts.push("\n- " + text);
        }
        return;
      }
      if (tag === "A") {
        const href = el.href;
        const text = (el.textContent || "").trim();
        if (text && href && !href.startsWith("javascript:")) {
          parts.push(text + " (" + href + ")");
        } else if (text) {
          parts.push(text);
        }
        return;
      }
      if (tag === "HR") {
        parts.push("\n\n---\n\n");
        return;
      }
      if (tag === "BR") {
        parts.push("\n");
        return;
      }
      if (tag === "PRE") {
        const text = (el.textContent || "").trim();
        if (text) parts.push("\n\n```\n" + text + "\n```\n\n");
        return;
      }
      const isBlock = BLOCK_TAGS.has(tag);
      if (isBlock) parts.push("\n\n");
      for (const child of el.childNodes) {
        walk(child);
      }
      if (isBlock) parts.push("\n");
    }
    walk(root);
    return parts.join("").replace(/\n{3,}/g, "\n\n").replace(/[ \t]+/g, " ").trim();
  }
  function extractYouTubeContent() {
    var _a, _b;
    const url = window.location.href;
    const titleEl = document.querySelector(
      "h1.ytd-watch-metadata, h1.ytd-video-primary-info-renderer, #title h1"
    );
    const videoTitle = ((_a = titleEl == null ? void 0 : titleEl.textContent) == null ? void 0 : _a.trim()) || document.title.replace(/ - YouTube$/, "").trim();
    let transcript = "";
    const transcriptSegments = document.querySelectorAll(
      "ytd-transcript-segment-renderer .segment-text, yt-formatted-string.ytd-transcript-segment-renderer, #segments-container ytd-transcript-segment-renderer"
    );
    if (transcriptSegments.length > 0) {
      const lines = [];
      transcriptSegments.forEach((seg) => {
        var _a2;
        const text2 = (_a2 = seg.textContent) == null ? void 0 : _a2.trim();
        if (text2) lines.push(text2);
      });
      transcript = lines.join(" ");
    }
    let description = "";
    const descEl = document.querySelector(
      "ytd-text-inline-expander #plain-snippet-text, #description-inline-expander, ytd-expander .content, #description .content"
    );
    if (descEl) {
      description = ((_b = descEl.textContent) == null ? void 0 : _b.trim()) || "";
    }
    const commentEls = document.querySelectorAll(
      "ytd-comment-thread-renderer #content-text"
    );
    let comments = "";
    if (commentEls.length > 0) {
      const commentLines = [];
      commentEls.forEach((el, i) => {
        var _a2;
        if (i >= 10) return;
        const text2 = (_a2 = el.textContent) == null ? void 0 : _a2.trim();
        if (text2) commentLines.push("- " + text2);
      });
      if (commentLines.length > 0) {
        comments = "\n\n## Top Comments\n" + commentLines.join("\n");
      }
    }
    let text = `# ${videoTitle}

`;
    if (transcript) {
      text += `## Transcript
${transcript}

`;
    }
    if (description) {
      text += `## Description
${description}

`;
    }
    text += comments;
    if (text.length > MAX_TEXT_BYTES) {
      text = text.slice(0, MAX_TEXT_BYTES) + "\n\n[...truncated]";
    }
    const wordCount = text.split(/\s+/).filter(Boolean).length;
    return {
      text,
      title: videoTitle,
      url,
      wordCount,
      isYouTube: true,
      videoTitle,
      transcript: transcript || void 0
    };
  }
  function extractMainContent() {
    var _a, _b, _c, _d;
    try {
      const url = window.location.href;
      const title = document.title;
      if (url.match(/\.pdf($|\?|#)/i) || document.contentType === "application/pdf") {
        return {
          text: ((_b = (_a = document.body) == null ? void 0 : _a.innerText) == null ? void 0 : _b.slice(0, MAX_TEXT_BYTES)) || "[PDF document]",
          title,
          url,
          wordCount: 0,
          isPdf: true
        };
      }
      if (url.includes("youtube.com/watch") || url.includes("youtu.be/")) {
        return extractYouTubeContent();
      }
      const root = findContentRoot();
      const cleaned = cleanClone(root);
      let text = domToStructuredText(cleaned);
      if (text.length < 100) {
        text = ((_c = document.body) == null ? void 0 : _c.innerText) || "";
      }
      if (text.length > MAX_TEXT_BYTES) {
        text = text.slice(0, MAX_TEXT_BYTES) + "\n\n[...truncated]";
      }
      const wordCount = text.split(/\s+/).filter(Boolean).length;
      return { text, title, url, wordCount };
    } catch (_e) {
      const fallbackText = (((_d = document.body) == null ? void 0 : _d.innerText) || "").slice(0, MAX_TEXT_BYTES);
      return {
        text: fallbackText,
        title: document.title,
        url: window.location.href,
        wordCount: fallbackText.split(/\s+/).filter(Boolean).length
      };
    }
  }
  const QUICK_ACTIONS_DEF = [
    { label: "Improve", icon: '<path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/>', action: "improve" },
    { label: "Expand", icon: '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>', action: "expand" },
    { label: "Shorten", icon: '<polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/>', action: "shorten" },
    { label: "Fix grammar", icon: '<polyline points="20 6 9 17 4 12"/>', action: "fix_grammar" },
    { label: "Translate", icon: '<circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/>', action: "translate" }
  ];
  const SKIP_INPUT_TYPES = /* @__PURE__ */ new Set(["password", "hidden", "file", "checkbox", "radio", "range", "color", "date", "datetime-local", "month", "week", "time", "submit", "reset", "button", "image"]);
  const MIN_INPUT_WIDTH = 200;
  function initQuickActionsOnInputs(ext2, safeSend2, showToast2) {
    const qaHost = document.createElement("div");
    qaHost.id = "aura-quick-action-host";
    Object.assign(qaHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      zIndex: "2147483646",
      pointerEvents: "none"
    });
    document.documentElement.appendChild(qaHost);
    const qaShadow = qaHost.attachShadow({ mode: "closed" });
    const qaStyle = document.createElement("style");
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
    const qaContainer = document.createElement("div");
    qaShadow.appendChild(qaContainer);
    let _qaActiveInput = null;
    let _qaTriggerEl = null;
    let _qaMenuEl = null;
    let _qaTranslateSub = null;
    function isEligibleInput(el) {
      if (el.tagName === "TEXTAREA") return true;
      if (el.tagName === "INPUT") {
        const inputEl = el;
        const inputType = (inputEl.type || "text").toLowerCase();
        if (SKIP_INPUT_TYPES.has(inputType)) return false;
        return true;
      }
      if (el.isContentEditable && el.getAttribute("contenteditable") === "true") return true;
      return false;
    }
    function removeQaMenu() {
      if (_qaMenuEl) {
        _qaMenuEl.remove();
        _qaMenuEl = null;
      }
      if (_qaTranslateSub) {
        _qaTranslateSub.remove();
        _qaTranslateSub = null;
      }
    }
    function removeQaTrigger() {
      if (_qaTriggerEl) {
        _qaTriggerEl.remove();
        _qaTriggerEl = null;
      }
      removeQaMenu();
      _qaActiveInput = null;
    }
    function positionTrigger(field) {
      const rect = field.getBoundingClientRect();
      if (rect.width < MIN_INPUT_WIDTH) {
        removeQaTrigger();
        return;
      }
      const offscreen = rect.bottom < 0 || rect.top > window.innerHeight || rect.right < 0 || rect.left > window.innerWidth;
      if (!_qaTriggerEl) {
        _qaTriggerEl = document.createElement("div");
        _qaTriggerEl.className = "qa-trigger";
        _qaTriggerEl.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;
        _qaTriggerEl.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (_qaMenuEl) {
            removeQaMenu();
            return;
          }
          showQaMenu();
        });
        qaContainer.appendChild(_qaTriggerEl);
      }
      _qaTriggerEl.style.display = offscreen ? "none" : "";
      if (offscreen) {
        removeQaMenu();
        return;
      }
      const trigSize = 20;
      const pad = 6;
      _qaTriggerEl.style.top = `${Math.round(rect.top + (rect.height - trigSize) / 2)}px`;
      _qaTriggerEl.style.left = `${Math.round(rect.right - trigSize - pad)}px`;
    }
    function getInputValue(el) {
      if (el.isContentEditable) {
        return el.innerText || "";
      }
      return el.value || "";
    }
    function setInputValue(el, value) {
      if (el.isContentEditable) {
        el.innerText = value;
      } else {
        el.value = value;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function executeQuickAction(action, language) {
      if (!_qaActiveInput) return;
      const text = getInputValue(_qaActiveInput);
      if (!text.trim()) {
        removeQaMenu();
        return;
      }
      if (_qaMenuEl) {
        _qaMenuEl.querySelectorAll(".qa-menu-item").forEach((item) => {
          item.classList.add("loading");
        });
      }
      const targetField = _qaActiveInput;
      safeSend2(
        { type: "QUICK_ACTION", action, text, language },
        (response) => {
          if (response && response.ok && response.result) {
            setInputValue(targetField, response.result);
            showToast2("Text updated by AURA");
          } else {
            showToast2((response == null ? void 0 : response.error) || "Quick action failed", 3e3);
          }
          removeQaMenu();
        }
      );
    }
    function toggleTranslateSub(anchor) {
      if (_qaTranslateSub) {
        _qaTranslateSub.remove();
        _qaTranslateSub = null;
        return;
      }
      const LANGUAGES = ["English", "Spanish", "French", "German", "Chinese", "Russian", "Japanese", "Arabic", "Portuguese", "Azerbaijani"];
      _qaTranslateSub = document.createElement("div");
      _qaTranslateSub.className = "qa-translate-sub";
      LANGUAGES.forEach((lang) => {
        const item = document.createElement("button");
        item.className = "qa-menu-item";
        item.textContent = lang;
        item.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          executeQuickAction("translate", lang);
        });
        _qaTranslateSub.appendChild(item);
      });
      if (_qaMenuEl && anchor.parentNode === _qaMenuEl) {
        anchor.after(_qaTranslateSub);
      }
    }
    function showQaMenu() {
      if (!_qaTriggerEl || !_qaActiveInput) return;
      removeQaMenu();
      _qaMenuEl = document.createElement("div");
      _qaMenuEl.className = "qa-menu";
      QUICK_ACTIONS_DEF.forEach((qa) => {
        const item = document.createElement("button");
        item.className = "qa-menu-item";
        item.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${qa.icon}</svg><span>${qa.label}</span>`;
        item.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (qa.action === "translate") {
            toggleTranslateSub(item);
          } else {
            executeQuickAction(qa.action);
          }
        });
        _qaMenuEl.appendChild(item);
      });
      qaContainer.appendChild(_qaMenuEl);
      const trigRect = _qaTriggerEl.getBoundingClientRect();
      const menuGap = 6;
      _qaMenuEl.style.top = `${Math.round(trigRect.bottom + menuGap)}px`;
      _qaMenuEl.style.left = `${Math.round(trigRect.right - 150)}px`;
      requestAnimationFrame(() => {
        if (!_qaMenuEl) return;
        const mRect = _qaMenuEl.getBoundingClientRect();
        if (mRect.right > window.innerWidth - 8) {
          _qaMenuEl.style.left = `${Math.round(window.innerWidth - mRect.width - 8)}px`;
        }
        if (mRect.left < 8) {
          _qaMenuEl.style.left = "8px";
        }
        if (mRect.bottom > window.innerHeight - 8) {
          _qaMenuEl.style.top = `${Math.round(trigRect.top - mRect.height - menuGap)}px`;
        }
      });
    }
    function onFieldFocus(e) {
      const target = e.target;
      if (!target || !isEligibleInput(target)) return;
      const rect = target.getBoundingClientRect();
      if (rect.width < MIN_INPUT_WIDTH) return;
      _qaActiveInput = target;
      positionTrigger(target);
    }
    function onFieldBlur(_e) {
      setTimeout(() => {
        if (_qaMenuEl || _qaTranslateSub) return;
        const active = document.activeElement;
        if (active && active === _qaActiveInput) return;
        removeQaTrigger();
      }, 200);
    }
    function repositionQaTrigger() {
      if (_qaActiveInput) {
        positionTrigger(_qaActiveInput);
      }
    }
    document.addEventListener("mousedown", (e) => {
      if (!_qaMenuEl && !_qaTriggerEl) return;
      const path = e.composedPath();
      if (path.includes(qaHost)) return;
      removeQaMenu();
    }, true);
    document.addEventListener("focusin", onFieldFocus, true);
    document.addEventListener("focusout", onFieldBlur, true);
    window.addEventListener("scroll", repositionQaTrigger, { passive: true });
    window.addEventListener("resize", repositionQaTrigger, { passive: true });
    const qaObserver = new MutationObserver(() => {
      if (_qaActiveInput && !document.body.contains(_qaActiveInput)) {
        removeQaTrigger();
      }
    });
    qaObserver.observe(document.body, { childList: true, subtree: true });
  }
  function initYoutubeRelay(safeSend2) {
    document.addEventListener("aura-yt-subtitles", (e) => {
      try {
        const d = e.detail;
        safeSend2({
          type: "YT_SUBTITLES",
          videoId: d.videoId || "",
          lang: d.lang || "",
          segments: d.segments || []
        });
      } catch {
      }
    });
    document.addEventListener("aura-yt-metadata", (e) => {
      try {
        const d = e.detail;
        safeSend2({
          type: "YT_METADATA",
          videoId: d.videoId || "",
          title: d.title || "",
          duration: d.duration || 0,
          description: d.description || "",
          channelName: d.channelName || "",
          chapters: d.chapters || [],
          captionTracks: d.captionTracks || []
        });
      } catch {
      }
    });
  }
  function initNetflixRelay(safeSend2) {
    document.addEventListener("aura-netflix-subtitles", (e) => {
      try {
        const d = e.detail;
        safeSend2({
          type: "NETFLIX_SUBTITLES",
          movieId: d.movieId || "",
          lang: d.lang || "",
          trackId: d.trackId || "",
          segments: d.segments || []
        });
      } catch {
      }
    });
    document.addEventListener("aura-netflix-metadata", (e) => {
      try {
        const d = e.detail;
        safeSend2({
          type: "NETFLIX_METADATA",
          movieId: d.movieId || "",
          title: d.title || "",
          episodeTitle: d.episodeTitle || "",
          seasonNumber: d.seasonNumber || 0,
          episodeNumber: d.episodeNumber || 0,
          duration: d.duration || 0
        });
      } catch {
      }
    });
  }
  const TRANSLATABLE_SELECTORS = "p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption";
  const AURA_TRANSLATE_ATTR = "data-aura-translated";
  const BATCH_SIZE = 10;
  const MAX_CONCURRENT = 10;
  function initTranslation(ext2) {
    const state = {
      mode: "bilingual",
      targetLang: "English",
      active: false,
      badge: null,
      elements: [],
      activeCount: 0
    };
    function getTranslatableElements() {
      const all = document.querySelectorAll(TRANSLATABLE_SELECTORS);
      const results = [];
      for (const el of all) {
        if (el.hasAttribute(AURA_TRANSLATE_ATTR)) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;
        if (el.closest("#aura-host, #aura-dock-shadow, #aura-quick-action-host, .aura-translate-badge")) continue;
        if (el.tagName === "SPAN" && (el.textContent || "").trim().length <= 20) continue;
        const text = (el.textContent || "").trim();
        if (text.length < 5) continue;
        results.push(el);
      }
      return results;
    }
    function createTranslationElement(originalEl) {
      const translationDiv = document.createElement("div");
      translationDiv.className = "aura-page-translation";
      translationDiv.setAttribute("data-aura-translation", "true");
      Object.assign(translationDiv.style, {
        borderLeft: "2px solid rgba(124, 58, 237, 0.6)",
        background: "rgba(124, 58, 237, 0.05)",
        padding: "6px 10px",
        marginTop: "4px",
        marginBottom: "4px",
        fontSize: "0.95em",
        color: "inherit",
        opacity: "0",
        fontFamily: "inherit",
        lineHeight: "1.5",
        borderRadius: "0 4px 4px 0",
        transition: "opacity 0.3s ease",
        fontStyle: "italic"
      });
      translationDiv.textContent = "Translating...";
      translationDiv.style.color = "rgba(124, 58, 237, 0.5)";
      originalEl.setAttribute(AURA_TRANSLATE_ATTR, "true");
      originalEl.after(translationDiv);
      requestAnimationFrame(() => {
        translationDiv.style.opacity = "0.6";
      });
      return translationDiv;
    }
    function fadeInTranslation(el, text) {
      el.style.opacity = "0";
      el.textContent = text;
      el.style.fontStyle = "normal";
      el.style.color = "inherit";
      requestAnimationFrame(() => {
        el.style.opacity = "0.85";
      });
    }
    function translateBatchRequest(texts, lang) {
      return new Promise((resolve) => {
        try {
          ext2.runtime.sendMessage(
            { type: "TRANSLATE_BATCH", texts, targetLang: lang },
            (response) => {
              if (ext2.runtime.lastError) {
                resolve(texts.map(() => "[Translation failed]"));
                return;
              }
              if ((response == null ? void 0 : response.ok) && response.translations) {
                resolve(response.translations);
              } else {
                resolve(texts.map(() => (response == null ? void 0 : response.error) || "[Translation failed]"));
              }
            }
          );
        } catch {
          resolve(texts.map(() => "[Translation failed]"));
        }
      });
    }
    function updateBadgeText() {
      if (!state.badge) return;
      const modeBtn = state.badge.querySelector("[data-badge-mode]");
      if (modeBtn) {
        modeBtn.textContent = state.mode === "bilingual" ? "Bilingual" : "Translated";
      }
    }
    function setMode(mode) {
      state.mode = mode;
      for (const pair of state.elements) {
        if (mode === "translated") {
          pair.original.style.display = "none";
          pair.translation.style.marginTop = "0";
        } else {
          pair.original.style.display = "";
          pair.translation.style.marginTop = "4px";
        }
      }
      updateBadgeText();
    }
    function remove() {
      state.active = false;
      for (const pair of state.elements) {
        pair.translation.remove();
        pair.original.removeAttribute(AURA_TRANSLATE_ATTR);
        pair.original.style.display = "";
      }
      state.elements = [];
      if (state.badge) {
        state.badge.remove();
        state.badge = null;
      }
    }
    function showTranslateBadge() {
      if (state.badge) {
        state.badge.remove();
        state.badge = null;
      }
      state.badge = document.createElement("div");
      state.badge.className = "aura-translate-badge";
      Object.assign(state.badge.style, {
        position: "fixed",
        bottom: "20px",
        right: "20px",
        zIndex: "2147483646",
        background: "rgba(10, 8, 24, 0.92)",
        backdropFilter: "blur(20px) saturate(1.5)",
        WebkitBackdropFilter: "blur(20px) saturate(1.5)",
        border: "1px solid rgba(124, 58, 237, 0.35)",
        borderRadius: "12px",
        padding: "8px 12px",
        display: "flex",
        alignItems: "center",
        gap: "8px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.05) inset",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif",
        fontSize: "12px",
        color: "rgba(226, 232, 240, 0.9)"
      });
      const dot = document.createElement("span");
      Object.assign(dot.style, {
        width: "6px",
        height: "6px",
        borderRadius: "50%",
        background: "#7c3aed",
        flexShrink: "0"
      });
      state.badge.appendChild(dot);
      const label = document.createElement("span");
      label.style.color = "rgba(160, 148, 210, 0.8)";
      label.textContent = "Translation active";
      state.badge.appendChild(label);
      const sep1 = document.createElement("span");
      Object.assign(sep1.style, { width: "1px", height: "14px", background: "rgba(255,255,255,0.1)", flexShrink: "0" });
      state.badge.appendChild(sep1);
      const langSpan = document.createElement("span");
      langSpan.setAttribute("data-badge-lang", "");
      langSpan.textContent = state.targetLang;
      langSpan.style.color = "rgba(124, 58, 237, 0.9)";
      langSpan.style.fontWeight = "600";
      state.badge.appendChild(langSpan);
      const badgeBtnBase = {
        background: "rgba(124, 58, 237, 0.15)",
        border: "1px solid rgba(124, 58, 237, 0.3)",
        borderRadius: "6px",
        color: "rgba(226, 232, 240, 0.9)",
        padding: "3px 8px",
        cursor: "pointer",
        fontSize: "11px",
        fontFamily: "inherit",
        transition: "background 0.15s, border-color 0.15s"
      };
      const modeBtn = document.createElement("button");
      modeBtn.setAttribute("data-badge-mode", "");
      modeBtn.textContent = "Bilingual";
      Object.assign(modeBtn.style, badgeBtnBase);
      modeBtn.addEventListener("mouseenter", () => {
        modeBtn.style.background = "rgba(124, 58, 237, 0.3)";
      });
      modeBtn.addEventListener("mouseleave", () => {
        modeBtn.style.background = "rgba(124, 58, 237, 0.15)";
      });
      modeBtn.addEventListener("click", () => {
        setMode(state.mode === "bilingual" ? "translated" : "bilingual");
      });
      state.badge.appendChild(modeBtn);
      const removeBtn = document.createElement("button");
      removeBtn.textContent = "✕";
      Object.assign(removeBtn.style, { ...badgeBtnBase, padding: "3px 6px", color: "rgba(226, 232, 240, 0.6)" });
      removeBtn.title = "Remove translation";
      removeBtn.addEventListener("mouseenter", () => {
        removeBtn.style.background = "rgba(239, 68, 68, 0.2)";
        removeBtn.style.borderColor = "rgba(239, 68, 68, 0.4)";
        removeBtn.style.color = "rgba(239, 68, 68, 0.9)";
      });
      removeBtn.addEventListener("mouseleave", () => {
        removeBtn.style.background = "rgba(124, 58, 237, 0.15)";
        removeBtn.style.borderColor = "rgba(124, 58, 237, 0.3)";
        removeBtn.style.color = "rgba(226, 232, 240, 0.6)";
      });
      removeBtn.addEventListener("click", () => {
        remove();
      });
      state.badge.appendChild(removeBtn);
      document.body.appendChild(state.badge);
    }
    async function start(targetLang) {
      state.targetLang = targetLang;
      state.active = true;
      state.mode = "bilingual";
      state.elements = [];
      state.activeCount = 0;
      showTranslateBadge();
      const elements = getTranslatableElements();
      if (elements.length === 0) return;
      const pairs = [];
      for (const el of elements) {
        const text = (el.textContent || "").trim();
        if (!text) continue;
        const translationDiv = createTranslationElement(el);
        state.elements.push({ original: el, translation: translationDiv });
        pairs.push({ original: el, translation: translationDiv, text });
      }
      const batches = [];
      for (let i = 0; i < pairs.length; i += BATCH_SIZE) {
        batches.push(pairs.slice(i, i + BATCH_SIZE));
      }
      const processBatch = async (batch) => {
        while (state.activeCount >= MAX_CONCURRENT) {
          await new Promise((r) => setTimeout(r, 100));
        }
        if (!state.active) return;
        state.activeCount++;
        try {
          const texts = batch.map((p) => p.text);
          const translations = await translateBatchRequest(texts, state.targetLang);
          if (!state.active) return;
          batch.forEach((pair, idx) => {
            if (!state.active) return;
            fadeInTranslation(pair.translation, translations[idx] || "[No translation]");
            if (state.mode === "translated") {
              pair.original.style.display = "none";
            }
          });
        } finally {
          state.activeCount--;
        }
      };
      const promises = batches.map((batch) => processBatch(batch));
      await Promise.all(promises);
    }
    return { start, remove, setMode };
  }
  const PAGE_CSS_PROPS = [
    "display",
    "position",
    "flex-direction",
    "align-items",
    "justify-content",
    "gap",
    "flex-wrap",
    "flex",
    "flex-grow",
    "flex-shrink",
    "width",
    "height",
    "min-width",
    "min-height",
    "max-width",
    "max-height",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "border",
    "border-radius",
    "border-color",
    "border-width",
    "border-style",
    "background",
    "background-color",
    "background-image",
    "background-size",
    "color",
    "font-size",
    "font-weight",
    "font-family",
    "line-height",
    "letter-spacing",
    "text-align",
    "text-decoration",
    "text-transform",
    "box-shadow",
    "opacity",
    "overflow",
    "z-index",
    "grid-template-columns",
    "grid-template-rows",
    "grid-gap",
    "transform",
    "transition"
  ];
  function extractComputedStylesForPage(el) {
    const styles = window.getComputedStyle(el);
    const result = {};
    for (const prop of PAGE_CSS_PROPS) {
      const val = styles.getPropertyValue(prop);
      if (val && val !== "none" && val !== "normal" && val !== "auto" && val !== "0px" && val !== "rgba(0, 0, 0, 0)") {
        result[prop] = val;
      }
    }
    return result;
  }
  function buildPageCssSelector(el) {
    const tag = el.tagName.toLowerCase();
    const cls = el.className && typeof el.className === "string" ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
    return tag + cls;
  }
  function extractFullPageData() {
    const clone = document.documentElement.cloneNode(true);
    const removeSelectors = [
      "script",
      "noscript",
      'iframe[src*="ads"]',
      'iframe[src*="track"]',
      'iframe[src*="pixel"]',
      'iframe[width="0"]',
      'iframe[height="0"]',
      'img[src*="pixel"]',
      'img[src*="track"]',
      'img[width="1"]',
      'img[height="1"]',
      '[id*="cookie"]',
      '[class*="cookie"]',
      '[id*="consent"]',
      '[class*="consent"]',
      '[id*="gdpr"]',
      '[class*="gdpr"]',
      '[id*="onetrust"]',
      '[class*="onetrust"]',
      '[id*="CybotCookiebot"]',
      '[data-testid*="cookie"]',
      '[id*="ad-"]',
      '[class*="ad-container"]',
      '[class*="ad-wrapper"]',
      'link[rel="preconnect"]',
      'link[rel="dns-prefetch"]',
      'meta[http-equiv="Content-Security-Policy"]',
      "style[data-emotion]"
      // runtime CSS-in-JS noise
    ];
    for (const sel of removeSelectors) {
      try {
        clone.querySelectorAll(sel).forEach((el) => el.remove());
      } catch (_e) {
      }
    }
    clone.querySelectorAll("*").forEach((el) => {
      const attrs = el.getAttributeNames();
      for (const attr of attrs) {
        if (attr.startsWith("on") || attr === "data-analytics" || attr === "data-tracking") {
          el.removeAttribute(attr);
        }
      }
    });
    const cleanHtml = clone.outerHTML;
    const cssMap = {};
    const keySelectors = [
      "body",
      "header",
      "nav",
      "main",
      "footer",
      "aside",
      "section",
      "article",
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "p",
      "a",
      "button",
      "input",
      "textarea",
      "ul",
      "ol",
      "li",
      "img",
      "form",
      "table",
      "th",
      "td",
      '[class*="hero"]',
      '[class*="card"]',
      '[class*="btn"]',
      '[class*="nav"]',
      '[class*="header"]',
      '[class*="footer"]',
      '[class*="sidebar"]',
      '[class*="container"]',
      '[class*="wrapper"]',
      '[class*="grid"]',
      '[class*="flex"]',
      '[class*="modal"]',
      '[class*="banner"]'
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
      } catch (_e) {
      }
    }
    const cssLines = [];
    for (const [selector, props] of Object.entries(cssMap)) {
      cssLines.push(`${selector} {`);
      for (const [prop, val] of Object.entries(props)) {
        cssLines.push(`  ${prop}: ${val};`);
      }
      cssLines.push("}");
      cssLines.push("");
    }
    const cssString = cssLines.join("\n");
    const colorSet = /* @__PURE__ */ new Set();
    const colorProps = ["color", "background-color", "border-color", "outline-color"];
    const sampleEls = document.querySelectorAll("*");
    let sampleCount = 0;
    for (const el of sampleEls) {
      if (sampleCount >= 500) break;
      const cs = window.getComputedStyle(el);
      for (const cp of colorProps) {
        const val = cs.getPropertyValue(cp);
        if (val && val !== "rgba(0, 0, 0, 0)" && val !== "transparent" && val !== "inherit" && val !== "initial") {
          colorSet.add(val);
        }
      }
      sampleCount++;
    }
    const colors = Array.from(colorSet).slice(0, 50);
    const fontSet = /* @__PURE__ */ new Set();
    for (const el of sampleEls) {
      if (fontSet.size >= 20) break;
      const cs = window.getComputedStyle(el);
      const ff = cs.getPropertyValue("font-family");
      if (ff) {
        const fonts2 = ff.split(",").map((f) => f.trim().replace(/^["']|["']$/g, ""));
        for (const font of fonts2) {
          if (font && !font.includes("inherit") && !font.includes("initial") && font.length < 50) {
            fontSet.add(font);
          }
        }
      }
    }
    const fonts = Array.from(fontSet).slice(0, 20);
    const getMeta = (name) => {
      const el = document.querySelector(`meta[property="${name}"], meta[name="${name}"]`);
      return (el == null ? void 0 : el.getAttribute("content")) || "";
    };
    const faviconEl = document.querySelector('link[rel="icon"], link[rel="shortcut icon"]');
    const metadata = {
      title: document.title || "",
      description: getMeta("description"),
      og_image: getMeta("og:image"),
      og_title: getMeta("og:title"),
      og_description: getMeta("og:description"),
      og_type: getMeta("og:type"),
      og_site_name: getMeta("og:site_name"),
      favicon: (faviconEl == null ? void 0 : faviconEl.getAttribute("href")) || ""
    };
    const viewport = {
      width: window.innerWidth,
      height: window.innerHeight
    };
    const mediaQueries = [];
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
        } catch (_e) {
        }
        if (mediaQueries.length >= 20) break;
      }
    } catch (_e) {
    }
    const images = [];
    document.querySelectorAll("img[src]").forEach((img) => {
      const src = img.getAttribute("src");
      if (src && !src.startsWith("data:") && images.length < 50) {
        try {
          images.push(new URL(src, location.href).href);
        } catch (_e) {
          images.push(src);
        }
      }
    });
    const stylesheets = [];
    document.querySelectorAll('link[rel="stylesheet"][href]').forEach((link) => {
      const href = link.getAttribute("href");
      if (href && stylesheets.length < 20) {
        try {
          stylesheets.push(new URL(href, location.href).href);
        } catch (_e) {
          stylesheets.push(href);
        }
      }
    });
    const elementCount = document.querySelectorAll("*").length;
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
      element_count: elementCount
    };
  }
  function initGoogleSerp(ext2, safeSend2) {
    var _a;
    let SERP_BACKEND = "https://aura-elnur.duckdns.org";
    let SERP_API_KEY = "";
    function loadSerpConfig() {
      return new Promise((resolve) => {
        var _a2;
        if (!((_a2 = ext2 == null ? void 0 : ext2.storage) == null ? void 0 : _a2.local)) {
          resolve();
          return;
        }
        ext2.storage.local.get(["backendUrl", "apiKey"], (d) => {
          var _a3, _b;
          if ((_a3 = d == null ? void 0 : d.backendUrl) == null ? void 0 : _a3.trim()) SERP_BACKEND = d.backendUrl.trim().replace(/\/+$/, "");
          if ((_b = d == null ? void 0 : d.apiKey) == null ? void 0 : _b.trim()) SERP_API_KEY = d.apiKey.trim();
          resolve();
        });
      });
    }
    loadSerpConfig();
    if ((_a = ext2 == null ? void 0 : ext2.storage) == null ? void 0 : _a.onChanged) {
      ext2.storage.onChanged.addListener((changes, area) => {
        var _a2, _b, _c;
        if (area !== "local") return;
        if ((_a2 = changes.backendUrl) == null ? void 0 : _a2.newValue) {
          SERP_BACKEND = changes.backendUrl.newValue.trim().replace(/\/+$/, "");
        }
        if (((_b = changes.apiKey) == null ? void 0 : _b.newValue) !== void 0) {
          SERP_API_KEY = ((_c = changes.apiKey.newValue) == null ? void 0 : _c.trim()) || "";
        }
      });
    }
    function isGoogleSearchPage() {
      const hostname = window.location.hostname;
      const pathname = window.location.pathname;
      const params = new URLSearchParams(window.location.search);
      if (!hostname.match(/^(www\.)?google\./)) return false;
      if (pathname !== "/search") return false;
      if (!params.get("q")) return false;
      const tbm = params.get("tbm");
      if (tbm && ["isch", "lcl", "vid", "shop", "nws", "bks", "fin"].includes(tbm)) return false;
      const udm = params.get("udm");
      if (udm && ["2", "14"].includes(udm)) return false;
      return true;
    }
    function getSearchQuery() {
      const params = new URLSearchParams(window.location.search);
      const qParam = params.get("q") || "";
      if (qParam) return qParam;
      const input = document.querySelector('input[name="q"]');
      return (input == null ? void 0 : input.value) || "";
    }
    function detectGoogleTheme() {
      const bg = window.getComputedStyle(document.body).backgroundColor;
      if (!bg || bg === "transparent") return "light";
      const rgbMatch = bg.match(/\d+/g);
      if (rgbMatch && rgbMatch.length >= 3) {
        const [r, g, b] = rgbMatch.map(Number);
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        return luminance < 128 ? "dark" : "light";
      }
      return "light";
    }
    function serpEscapeHtml(text) {
      return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    }
    function serpRenderMarkdown(text) {
      let html = serpEscapeHtml(text);
      html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      html = html.replace(/__(.+?)__/g, "<strong>$1</strong>");
      html = html.replace(new RegExp("(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)", "g"), "<em>$1</em>");
      html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
      html = html.replace(
        /\[([^\]]+)\]\((https?:\/\/[^)"]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
      html = html.replace(/^[\s]*[-*]\s+(.+)$/gm, "<li>$1</li>");
      html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");
      html = html.replace(/^[\s]*\d+\.\s+(.+)$/gm, "<li>$1</li>");
      html = html.replace(/\n\n+/g, "</p><p>");
      html = "<p>" + html + "</p>";
      html = html.replace(/\n/g, "<br>");
      html = html.replace(/<p>\s*<\/p>/g, "");
      html = html.replace(/<\/?(?!(?:strong|em|code|a|li|ul|ol|p|br)\b)[^>]*>/gi, "");
      return html;
    }
    function serpAddCitations(bodyEl, fullText) {
      const citationRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
      const citations = [];
      let citMatch;
      while ((citMatch = citationRegex.exec(fullText)) !== null) {
        citations.push({ title: citMatch[1], url: citMatch[2] });
      }
      if (citations.length === 0) return;
      const citationsContainer = document.createElement("div");
      citationsContainer.className = "serp-citations";
      const citLabel = document.createElement("div");
      citLabel.className = "serp-citations-label";
      citLabel.textContent = "Sources";
      citationsContainer.appendChild(citLabel);
      const citList = document.createElement("div");
      citList.className = "serp-citation-list";
      citations.forEach((cit, idx) => {
        const chip = document.createElement("a");
        chip.className = "serp-citation-chip";
        chip.href = cit.url;
        chip.target = "_blank";
        chip.rel = "noopener noreferrer";
        const num = document.createElement("span");
        num.className = "serp-citation-num";
        num.textContent = String(idx + 1);
        chip.appendChild(num);
        const chipText = document.createTextNode(" " + cit.title);
        chip.appendChild(chipText);
        citList.appendChild(chip);
      });
      citationsContainer.appendChild(citList);
      bodyEl.appendChild(citationsContainer);
    }
    function serpAddFooter(cardEl, query, fullText) {
      const footer = document.createElement("div");
      footer.className = "serp-footer";
      const followupBtn = document.createElement("button");
      followupBtn.className = "serp-followup-btn";
      followupBtn.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Ask follow-up`;
      followupBtn.addEventListener("click", () => {
        safeSend2({
          type: "OPEN_WITH_TEXT",
          action: "ask",
          text: `I searched for "${query}" and got the following AI answer:

${fullText}

I have a follow-up question: `,
          url: window.location.href,
          title: document.title
        });
      });
      const powered = document.createElement("span");
      powered.className = "serp-powered";
      powered.textContent = "Powered by AURA";
      footer.appendChild(followupBtn);
      footer.appendChild(powered);
      cardEl.appendChild(footer);
    }
    async function initGoogleSerpIntegration() {
      if (!isGoogleSearchPage()) return;
      const stored = await new Promise((resolve) => {
        ext2.storage.local.get(["aura_serp_hidden"], resolve);
      });
      if (stored.aura_serp_hidden) return;
      const query = getSearchQuery();
      if (!query) return;
      const serpHost = document.createElement("div");
      serpHost.id = "aura-serp-host";
      Object.assign(serpHost.style, {
        position: "fixed",
        top: "80px",
        right: "16px",
        width: "340px",
        maxHeight: "calc(100vh - 100px)",
        zIndex: "2147483640",
        pointerEvents: "auto"
      });
      document.documentElement.appendChild(serpHost);
      const serpShadow = serpHost.attachShadow({ mode: "closed" });
      const theme = detectGoogleTheme();
      const isDark = theme === "dark";
      const serpStyle = document.createElement("style");
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
        background: ${isDark ? "rgba(30, 27, 48, 0.92)" : "rgba(255, 255, 255, 0.95)"};
        backdrop-filter: blur(24px) saturate(1.4);
        -webkit-backdrop-filter: blur(24px) saturate(1.4);
        border-radius: 16px;
        overflow-y: auto;
        max-height: calc(100vh - 120px);
        box-shadow: ${isDark ? "0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.06)" : "0 8px 40px rgba(0,0,0,0.12), 0 0 0 1px rgba(0,0,0,0.06)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.15)"};
        padding: 20px 24px 16px;
        animation: serp-fade-in 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
      }
      .serp-card:hover {
        border-color: ${isDark ? "rgba(124, 58, 237, 0.35)" : "rgba(124, 58, 237, 0.3)"};
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
        color: ${isDark ? "rgba(160, 148, 210, 0.9)" : "rgba(124, 58, 237, 0.85)"};
        background: ${isDark ? "rgba(124, 58, 237, 0.12)" : "rgba(124, 58, 237, 0.08)"};
        border-radius: 8px;
        flex-shrink: 0;
      }
      .serp-title {
        font-size: 14px;
        font-weight: 600;
        color: ${isDark ? "rgba(226, 232, 240, 0.9)" : "rgba(30, 27, 48, 0.9)"};
        letter-spacing: -0.01em;
      }
      .serp-title-sub {
        font-size: 11px;
        font-weight: 400;
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.6)"};
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
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.5)"};
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s, color 0.15s;
        padding: 0;
      }
      .serp-ctrl-btn:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.15)" : "rgba(124, 58, 237, 0.1)"};
        color: ${isDark ? "rgba(224, 214, 255, 1)" : "rgba(124, 58, 237, 0.9)"};
      }
      .serp-ctrl-btn[title="Hide AURA answers"]:hover {
        background: rgba(239, 68, 68, 0.12);
        color: rgba(239, 68, 68, 0.9);
      }

      .serp-body {
        font-size: 14px;
        line-height: 1.7;
        color: ${isDark ? "rgba(226, 232, 240, 0.85)" : "rgba(30, 27, 48, 0.85)"};
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
        background: ${isDark ? "rgba(124, 58, 237, 0.6)" : "rgba(124, 58, 237, 0.5)"};
        animation: serp-pulse 1.2s ease-in-out infinite;
      }
      .serp-loading-dots span:nth-child(2) { animation-delay: 0.2s; }
      .serp-loading-dots span:nth-child(3) { animation-delay: 0.4s; }
      .serp-loading-text {
        font-size: 13px;
        color: ${isDark ? "rgba(160, 148, 210, 0.6)" : "rgba(100, 90, 140, 0.6)"};
      }

      .serp-answer {
        white-space: pre-wrap;
        word-break: break-word;
      }
      .serp-answer p { margin-bottom: 8px; }
      .serp-answer p:last-child { margin-bottom: 0; }
      .serp-answer strong, .serp-answer b {
        font-weight: 600;
        color: ${isDark ? "rgba(226, 232, 240, 0.95)" : "rgba(30, 27, 48, 0.95)"};
      }
      .serp-answer code {
        background: ${isDark ? "rgba(124, 58, 237, 0.1)" : "rgba(124, 58, 237, 0.06)"};
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
        color: ${isDark ? "rgba(160, 148, 255, 0.9)" : "rgba(100, 58, 237, 0.9)"};
        text-decoration: none;
      }
      .serp-answer a:hover { text-decoration: underline; }

      .serp-citations {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"};
      }
      .serp-citations-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: ${isDark ? "rgba(160, 148, 210, 0.5)" : "rgba(100, 90, 140, 0.5)"};
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
        background: ${isDark ? "rgba(124, 58, 237, 0.1)" : "rgba(124, 58, 237, 0.06)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.15)" : "rgba(124, 58, 237, 0.1)"};
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 12px;
        color: ${isDark ? "rgba(200, 180, 255, 0.8)" : "rgba(100, 58, 237, 0.8)"};
        text-decoration: none;
        transition: background 0.15s, border-color 0.15s;
        max-width: 280px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      .serp-citation-chip:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.12)"};
        border-color: ${isDark ? "rgba(124, 58, 237, 0.3)" : "rgba(124, 58, 237, 0.2)"};
      }
      .serp-citation-num {
        width: 16px;
        height: 16px;
        border-radius: 4px;
        background: ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.1)"};
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
        border-top: 1px solid ${isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)"};
      }
      .serp-followup-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: ${isDark ? "rgba(124, 58, 237, 0.12)" : "rgba(124, 58, 237, 0.08)"};
        border: 1px solid ${isDark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.15)"};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12.5px;
        font-weight: 500;
        font-family: inherit;
        color: ${isDark ? "rgba(200, 180, 255, 0.9)" : "rgba(100, 58, 237, 0.9)"};
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s, color 0.15s, transform 0.15s;
      }
      .serp-followup-btn:hover {
        background: ${isDark ? "rgba(124, 58, 237, 0.22)" : "rgba(124, 58, 237, 0.15)"};
        border-color: ${isDark ? "rgba(124, 58, 237, 0.35)" : "rgba(124, 58, 237, 0.3)"};
        transform: scale(1.01);
      }
      .serp-followup-btn:active { transform: scale(0.98); }
      .serp-powered {
        font-size: 11px;
        color: ${isDark ? "rgba(160, 148, 210, 0.35)" : "rgba(100, 90, 140, 0.35)"};
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
        color: ${isDark ? "rgba(226, 232, 240, 0.5)" : "rgba(30, 27, 48, 0.5)"};
      }

      .serp-error {
        font-size: 13px;
        color: ${isDark ? "rgba(239, 150, 150, 0.8)" : "rgba(200, 50, 50, 0.7)"};
        padding: 4px 0;
      }
    `;
      serpShadow.appendChild(serpStyle);
      const card = document.createElement("div");
      card.className = "serp-card";
      const serpHeader = document.createElement("div");
      serpHeader.className = "serp-header";
      const headerLeft = document.createElement("div");
      headerLeft.className = "serp-header-left";
      const serpLogo = document.createElement("div");
      serpLogo.className = "serp-logo";
      serpLogo.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
      const titleWrap = document.createElement("div");
      const titleText = document.createElement("span");
      titleText.className = "serp-title";
      titleText.textContent = "AI Answer";
      const titleSub = document.createElement("span");
      titleSub.className = "serp-title-sub";
      titleSub.textContent = "by AURA";
      titleWrap.appendChild(titleText);
      titleWrap.appendChild(titleSub);
      headerLeft.appendChild(serpLogo);
      headerLeft.appendChild(titleWrap);
      const controls = document.createElement("div");
      controls.className = "serp-controls";
      const collapseBtn = document.createElement("button");
      collapseBtn.className = "serp-ctrl-btn";
      collapseBtn.title = "Collapse";
      collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
      const hideBtn = document.createElement("button");
      hideBtn.className = "serp-ctrl-btn";
      hideBtn.title = "Hide AURA answers";
      hideBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
      controls.appendChild(collapseBtn);
      controls.appendChild(hideBtn);
      serpHeader.appendChild(headerLeft);
      serpHeader.appendChild(controls);
      card.appendChild(serpHeader);
      const serpBody = document.createElement("div");
      serpBody.className = "serp-body";
      const serpLoading = document.createElement("div");
      serpLoading.className = "serp-loading";
      const serpDots = document.createElement("div");
      serpDots.className = "serp-loading-dots";
      serpDots.innerHTML = "<span></span><span></span><span></span>";
      const serpLoadingText = document.createElement("span");
      serpLoadingText.className = "serp-loading-text";
      serpLoadingText.textContent = `Thinking about "${query.slice(0, 60)}${query.length > 60 ? "..." : ""}"`;
      serpLoading.appendChild(serpDots);
      serpLoading.appendChild(serpLoadingText);
      serpBody.appendChild(serpLoading);
      card.appendChild(serpBody);
      serpShadow.appendChild(card);
      let isSerpCollapsed = false;
      collapseBtn.addEventListener("click", () => {
        isSerpCollapsed = !isSerpCollapsed;
        if (isSerpCollapsed) {
          serpBody.classList.add("collapsed");
          collapseBtn.title = "Expand";
          collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>`;
        } else {
          serpBody.classList.remove("collapsed");
          collapseBtn.title = "Collapse";
          collapseBtn.innerHTML = `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 15 12 9 6 15"/></svg>`;
        }
      });
      hideBtn.addEventListener("click", () => {
        ext2.storage.local.set({ aura_serp_hidden: true });
        serpHost.remove();
      });
      try {
        const fetchBody = JSON.stringify({
          message: query,
          conversation_id: "__serp_answer__",
          stream: false,
          system_context: `The user searched Google for: "${query}". Provide a concise, direct answer to their query. Be helpful and factual. Use markdown formatting sparingly — bold for emphasis, lists where appropriate. If you reference sources, format them as [Source Title](URL) and they will be rendered as citation chips. Keep the answer focused and under 200 words unless the topic requires more detail.`
        });
        let proxyResult = null;
        (async () => {
          try {
            proxyResult = await new Promise((resolve, reject) => {
              ext2.runtime.sendMessage(
                { type: "SERP_FETCH", url: `${SERP_BACKEND}/api/chat`, body: fetchBody, apiKey: SERP_API_KEY },
                (response) => {
                  if (ext2.runtime.lastError) {
                    reject(new Error(ext2.runtime.lastError.message));
                  } else {
                    resolve(response);
                  }
                }
              );
            });
          } catch {
            const serpHeaders = { "Content-Type": "application/json" };
            if (SERP_API_KEY) serpHeaders["X-API-Key"] = SERP_API_KEY;
            const directResp = await fetch(`${SERP_BACKEND}/api/chat`, {
              method: "POST",
              headers: serpHeaders,
              body: fetchBody,
              signal: AbortSignal.timeout(3e4)
            });
            if (!directResp.ok) throw new Error(`HTTP ${directResp.status}`);
            proxyResult = { ok: true, text: await directResp.text() };
          }
          if (!(proxyResult == null ? void 0 : proxyResult.ok)) {
            throw new Error((proxyResult == null ? void 0 : proxyResult.error) || "Backend unreachable");
          }
          serpLoading.remove();
          const answerEl = document.createElement("div");
          answerEl.className = "serp-answer";
          serpBody.appendChild(answerEl);
          let fullText = "";
          const responseText = proxyResult.text || "";
          const lines = responseText.split("\n").filter((l) => l.trim());
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
          answerEl.innerHTML = serpRenderMarkdown(fullText);
          if (!fullText.trim()) {
            answerEl.innerHTML = '<span class="serp-error">No response from AI.</span>';
            return;
          }
          serpAddCitations(serpBody, fullText);
          serpAddFooter(card, query, fullText);
        })().catch((_err) => {
          serpLoading.remove();
          const offline = document.createElement("div");
          offline.className = "serp-offline";
          const offDot = document.createElement("div");
          offDot.className = "serp-offline-dot";
          const offText = document.createElement("span");
          offText.className = "serp-offline-text";
          offText.textContent = `AURA is offline — backend did not respond (${(_err == null ? void 0 : _err.message) || "timeout"})`;
          offline.appendChild(offDot);
          offline.appendChild(offText);
          serpBody.appendChild(offline);
        });
      } catch (_err) {
        serpLoading.remove();
        const offline = document.createElement("div");
        offline.className = "serp-offline";
        const offDot = document.createElement("div");
        offDot.className = "serp-offline-dot";
        const offText = document.createElement("span");
        offText.className = "serp-offline-text";
        offText.textContent = `AURA is offline — backend did not respond`;
        offline.appendChild(offDot);
        offline.appendChild(offText);
        serpBody.appendChild(offline);
      }
    }
    initGoogleSerpIntegration();
  }
  function setupMessageListener(ext2, handlers) {
    ext2.runtime.onMessage.addListener(
      (msg, _sender, sendResponse) => {
        if (msg.type === "EXTRACT_PAGE") {
          sendResponse(handlers.extractMainContent());
          return false;
        }
        if (msg.type === "GET_DOM") {
          sendResponse({ ok: true, dom: handlers.serializeDOM(), url: location.href, title: document.title });
          return false;
        }
        if (msg.type === "EXEC_ACTION") {
          sendResponse(handlers.execAction(msg.action));
          return false;
        }
        if (msg.type === "FILL_FORM") {
          const fields = msg.fields;
          let filled = 0;
          for (const field of fields || []) {
            const result = handlers.execAction({ action: "type", selector: field.selector, text: field.value });
            if (result.ok) filled++;
          }
          sendResponse({ ok: true, filled, total: (fields == null ? void 0 : fields.length) || 0 });
          return false;
        }
        if (msg.type === "SHOW_OCR_OVERLAY") {
          handlers.showOcrOverlay(msg.dataUrl, sendResponse);
          return true;
        }
        if (msg.type === "PAGE_TRANSLATE") {
          if (handlers.translateActive) handlers.removePageTranslation();
          handlers.startPageTranslation(msg.targetLang).then(() => {
            sendResponse({ ok: true });
          }).catch((err) => {
            sendResponse({ ok: false, error: err.message });
          });
          return true;
        }
        if (msg.type === "TRANSLATE_TOGGLE_MODE") {
          handlers.setTranslateMode(msg.mode);
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "TRANSLATE_REMOVE") {
          handlers.removePageTranslation();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "TRANSLATE_CHANGE_LANG") {
          if (handlers.translateActive) {
            handlers.removePageTranslation();
            handlers.startPageTranslation(msg.targetLang).then(() => {
              sendResponse({ ok: true });
            }).catch((err) => {
              sendResponse({ ok: false, error: err.message });
            });
            return true;
          }
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "SCROLL_TO_HIGHLIGHT") {
          handlers.scrollToHighlight(msg.id);
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "SHOW_DOCK") {
          handlers.showDock();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "START_CAPTURE_MODE") {
          handlers.startCaptureMode();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "STOP_CAPTURE_MODE") {
          handlers.stopCaptureMode();
          sendResponse({ ok: true });
          return false;
        }
        if (msg.type === "EXTRACT_FULL_PAGE") {
          try {
            const data = handlers.extractFullPageData();
            sendResponse({ ok: true, data });
          } catch (err) {
            sendResponse({ ok: false, error: err.message || "Extraction failed" });
          }
          return false;
        }
        return void 0;
      }
    );
  }
  const ext = typeof browser !== "undefined" ? browser : chrome;
  const STALE_IDS = [
    "aura-shadow-host",
    "aura-dock-shadow",
    "aura-host",
    "aura-quick-action-host",
    "aura-highlight-host",
    "aura-img-toolbar-host",
    "aura-capture-host"
  ];
  let _shadowHost = null;
  function safeSend(msg, cb) {
    var _a;
    try {
      if (cb) {
        ext.runtime.sendMessage(msg, cb);
      } else {
        ext.runtime.sendMessage(msg);
      }
    } catch (err) {
      const msg_ = (err == null ? void 0 : err.message) ?? "";
      if (msg_.includes("Extension context invalidated")) {
        _shadowHost == null ? void 0 : _shadowHost.remove();
        for (const id of STALE_IDS) {
          (_a = document.getElementById(id)) == null ? void 0 : _a.remove();
        }
      }
    }
  }
  function showToast(message, duration = 2e3) {
    const toast = document.createElement("div");
    Object.assign(toast.style, {
      position: "fixed",
      top: "16px",
      left: "50%",
      transform: "translateX(-50%)",
      background: "rgba(10,8,24,0.92)",
      backdropFilter: "blur(16px)",
      WebkitBackdropFilter: "blur(16px)",
      border: "1px solid rgba(124,58,237,0.35)",
      borderRadius: "8px",
      padding: "8px 16px",
      color: "rgba(226,232,240,0.92)",
      fontSize: "13px",
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif',
      fontWeight: "500",
      zIndex: "2147483647",
      pointerEvents: "none",
      boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
      whiteSpace: "nowrap"
    });
    toast.textContent = message;
    document.documentElement.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
  }
  function init() {
    var _a;
    if (window.__auraToolbarMounted) return;
    window.__auraToolbarMounted = true;
    for (const id of STALE_IDS) {
      (_a = document.getElementById(id)) == null ? void 0 : _a.remove();
    }
    const shadowHost = document.createElement("div");
    shadowHost.id = "aura-shadow-host";
    Object.assign(shadowHost.style, {
      position: "fixed",
      top: "0",
      left: "0",
      width: "0",
      height: "0",
      zIndex: "2147483647",
      pointerEvents: "none",
      overflow: "visible"
    });
    document.documentElement.appendChild(shadowHost);
    _shadowHost = shadowHost;
    const shadow = shadowHost.attachShadow({ mode: "open" });
    const styleEl = document.createElement("style");
    styleEl.textContent = buildStylesheet();
    shadow.appendChild(styleEl);
    const store = createContextStore();
    initContextEngine(store, ext);
    store.subscribe((signal) => {
      shadowHost.style.setProperty("--aura-accent", signal.accent);
      shadowHost.style.setProperty("--aura-glow", signal.glow);
    });
    function makeContainer(id) {
      const el = document.createElement("div");
      el.dataset.auraModule = id;
      Object.assign(el.style, { all: "unset", pointerEvents: "none" });
      shadow.appendChild(el);
      return el;
    }
    const fabContainer = makeContainer("fab");
    const ghostBarContainer = makeContainer("ghost-bar");
    const modalContainer = makeContainer("modal");
    const highlightsContainer = makeContainer("highlights");
    const captureContainer = makeContainer("capture");
    const linkPreviewContainer = makeContainer("link-preview");
    const fab = createFab();
    const ghostBar = createGhostBar();
    const modal = createModal();
    const highlights = createHighlights();
    const gmail = createGmail();
    const capture = createCapture();
    const linkPreview = createLinkPreview();
    fab.init(fabContainer, store, ext);
    ghostBar.init(ghostBarContainer, store, ext);
    modal.init(modalContainer, store, ext);
    highlights.init(highlightsContainer, store, ext);
    gmail.init(document.body, store, ext);
    capture.init(captureContainer, store, ext);
    linkPreview.init(linkPreviewContainer, store, ext);
    ghostBar.onAskClicked((content) => {
      if (content.type === "text") {
        modal.openWithText(content.text, content.rect);
      } else {
        modal.openWithImage(content.imageUrl, content.rect);
      }
    });
    modal.onAction((action, text, _model) => {
      safeSend({
        type: "OPEN_WITH_TEXT",
        action,
        text,
        url: location.href,
        title: document.title
      });
    });
    highlights.setShowToast(showToast);
    const translation = initTranslation(ext);
    let translateActive = false;
    initQuickActionsOnInputs(ext, safeSend, showToast);
    initYoutubeRelay(safeSend);
    initNetflixRelay(safeSend);
    initGoogleSerp(ext, safeSend);
    setupMessageListener(ext, {
      extractMainContent,
      serializeDOM,
      execAction,
      showOcrOverlay,
      startPageTranslation: async (targetLang) => {
        await translation.start(targetLang);
        translateActive = true;
      },
      removePageTranslation: () => {
        translation.remove();
        translateActive = false;
      },
      setTranslateMode: (mode) => translation.setMode(mode),
      scrollToHighlight: (id) => highlights.scrollTo(id),
      showDock: () => fab.showDock(),
      startCaptureMode: () => capture.start(),
      stopCaptureMode: () => capture.stop(),
      extractFullPageData,
      get translateActive() {
        return translateActive;
      }
    });
  }
  init();
})();
