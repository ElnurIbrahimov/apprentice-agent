/**
 * AURA Content Script — Coordinator
 * Initializes all modules, mounts shared Shadow DOM, wires inter-module events.
 */

import type { ContextStore } from './types';
import { createContextStore, initContextEngine } from './context-engine';
import { buildStylesheet } from './styles';
import { createFab } from './fab';
import { createGhostBar } from './ghost-bar';
import { createModal } from './modal';
import { createHighlights } from './highlights';
import { createGmail } from './gmail';
import { createCapture } from './capture';
import { createLinkPreview } from './link-preview';
import {
  setupMessageListener,
  initQuickActionsOnInputs,
  initYoutubeRelay,
  initNetflixRelay,
  initTranslation,
  initGoogleSerp,
  extractMainContent,
  serializeDOM,
  execAction,
  showOcrOverlay,
  extractFullPageData,
} from './page-services';
import { initStuckDetector } from './stuck-detector';
import { initGhostText } from './ghost-text';
import { initGhostTextInline } from './ghost-text-inline';
import { initLifelog } from './lifelog';
import { initAmbientSurface } from './ambient-surface';

// ── Firefox compat shim ────────────────────────────────────────────────────────

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

// ── Legacy element IDs to clean up from previous injection ─────────────────────

const STALE_IDS = [
  'aura-shadow-host',
  'aura-dock-shadow',
  'aura-host',
  'aura-quick-action-host',
  'aura-highlight-host',
  'aura-img-toolbar-host',
  'aura-capture-host',
];

// ── safeSend ───────────────────────────────────────────────────────────────────

let _shadowHost: HTMLElement | null = null;

function safeSend(msg: any, cb?: (r: any) => void): void {
  try {
    if (cb) {
      ext.runtime.sendMessage(msg, cb);
    } else {
      ext.runtime.sendMessage(msg);
    }
  } catch (err: any) {
    const msg_ = err?.message ?? '';
    if (msg_.includes('Extension context invalidated')) {
      // Clean up the shadow host and stale elements
      _shadowHost?.remove();
      for (const id of STALE_IDS) {
        document.getElementById(id)?.remove();
      }
    }
  }
}

// ── showToast ──────────────────────────────────────────────────────────────────

function showToast(message: string, duration = 2000): void {
  const toast = document.createElement('div');
  Object.assign(toast.style, {
    position: 'fixed',
    top: '16px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: 'rgba(10,8,24,0.92)',
    backdropFilter: 'blur(16px)',
    WebkitBackdropFilter: 'blur(16px)',
    border: '1px solid rgba(124,58,237,0.35)',
    borderRadius: '8px',
    padding: '8px 16px',
    color: 'rgba(226,232,240,0.92)',
    fontSize: '13px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", system-ui, sans-serif',
    fontWeight: '500',
    zIndex: '2147483647',
    pointerEvents: 'none',
    boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
    whiteSpace: 'nowrap',
  });
  toast.textContent = message;
  document.documentElement.appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

// ── Main init ──────────────────────────────────────────────────────────────────

export function init(): void {
  // Guard against double-mount
  if ((window as any).__auraToolbarMounted) return;
  (window as any).__auraToolbarMounted = true;

  // Clean stale elements from previous injection
  for (const id of STALE_IDS) {
    document.getElementById(id)?.remove();
  }

  // ── Create shared Shadow DOM host ──────────────────────────────────────────

  const shadowHost = document.createElement('div');
  shadowHost.id = 'aura-shadow-host';
  Object.assign(shadowHost.style, {
    position: 'fixed',
    top: '0',
    left: '0',
    width: '0',
    height: '0',
    zIndex: '2147483647',
    pointerEvents: 'none',
    overflow: 'visible',
  });
  document.documentElement.appendChild(shadowHost);
  _shadowHost = shadowHost;

  const shadow = shadowHost.attachShadow({ mode: 'open' });

  // Inject stylesheet
  const styleEl = document.createElement('style');
  styleEl.textContent = buildStylesheet();
  shadow.appendChild(styleEl);

  // ── Init context engine ────────────────────────────────────────────────────

  const store: ContextStore = createContextStore();
  const cleanupContextEngine = initContextEngine(store, ext as typeof chrome);

  // Subscribe to store changes — update CSS custom properties on shadow host
  store.subscribe((signal) => {
    shadowHost.style.setProperty('--aura-accent', signal.accent);
    shadowHost.style.setProperty('--aura-glow', signal.glow);
  });

  // ── Create module containers ───────────────────────────────────────────────

  function makeContainer(id: string): HTMLElement {
    const el = document.createElement('div');
    el.dataset.auraModule = id;
    Object.assign(el.style, { all: 'unset', pointerEvents: 'none' });
    shadow.appendChild(el);
    return el;
  }

  const fabContainer       = makeContainer('fab');
  const ghostBarContainer  = makeContainer('ghost-bar');
  const modalContainer     = makeContainer('modal');
  const highlightsContainer = makeContainer('highlights');
  const captureContainer   = makeContainer('capture');
  const linkPreviewContainer = makeContainer('link-preview');

  // ── Create and init all modules ────────────────────────────────────────────

  const fab         = createFab();
  const ghostBar    = createGhostBar();
  const modal       = createModal();
  const highlights  = createHighlights();
  const gmail       = createGmail();
  const capture     = createCapture();
  const linkPreview = createLinkPreview();

  fab.init(fabContainer, store, ext as typeof chrome);
  ghostBar.init(ghostBarContainer, store, ext as typeof chrome);
  modal.init(modalContainer, store, ext as typeof chrome);
  highlights.init(highlightsContainer, store, ext as typeof chrome);
  // Gmail injects into page DOM — pass document.body
  gmail.init(document.body, store, ext as typeof chrome);
  capture.init(captureContainer, store, ext as typeof chrome);
  linkPreview.init(linkPreviewContainer, store, ext as typeof chrome);

  // ── Wire ghost bar → modal ─────────────────────────────────────────────────

  ghostBar.onAskClicked((content) => {
    if (content.type === 'text') {
      modal.openWithText(content.text, content.rect);
    } else {
      modal.openWithImage(content.imageUrl, content.rect);
    }
  });

  // ── Wire modal → background ────────────────────────────────────────────────

  modal.onAction((action, text, _model) => {
    safeSend({
      type: 'OPEN_WITH_TEXT',
      action,
      text,
      url: location.href,
      title: document.title,
    });
  });

  // ── Wire highlights toast ──────────────────────────────────────────────────

  highlights.setShowToast(showToast);

  // ── Init translation (returns object with state) ───────────────────────────

  const translation = initTranslation(ext as typeof chrome);

  // Track translation active state — initTranslation doesn't expose it
  let translateActive = false;

  // ── Init page services ─────────────────────────────────────────────────────

  initQuickActionsOnInputs(ext as typeof chrome, safeSend, showToast);
  initYoutubeRelay(safeSend);
  initNetflixRelay(safeSend);
  initGoogleSerp(ext as typeof chrome, safeSend);
  initStuckDetector();
  initGhostText();
  initGhostTextInline();
  initLifelog();
  initAmbientSurface();

  // ── Setup message listener ─────────────────────────────────────────────────

  setupMessageListener(ext as typeof chrome, {
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
    },
  });
}
