import type { PageContext } from './types';

// ── Context palette ───────────────────────────────────────────────────────────

export interface ContextPalette {
  accent: string;  // hex
  glow: string;    // rgba
  icon: string;    // SVG markup
}

export const PALETTES: Record<PageContext, ContextPalette> = {
  general: {
    accent: '#7c3aed',
    glow: 'rgba(124, 58, 237, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 1L10 6H15L11 9.5L12.5 14.5L8 11.5L3.5 14.5L5 9.5L1 6H6L8 1Z"/>
    </svg>`,
  },
  article: {
    accent: '#3b82f6',
    glow: 'rgba(59, 130, 246, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M3 2h10a1 1 0 011 1v10a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1zm1 3v1h8V5H4zm0 3v1h8V8H4zm0 3v1h5v-1H4z"/>
    </svg>`,
  },
  media: {
    accent: '#f59e0b',
    glow: 'rgba(245, 158, 11, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M2 4h1v8H2V4zm2 2h1v4H4V6zm2-1h1v6H6V5zm2 2h1v2H8V7zm2-2h1v6h-1V5zm2 1h1v4h-1V6z"/>
    </svg>`,
  },
  code: {
    accent: '#10b981',
    glow: 'rgba(16, 185, 129, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M5.5 3.5L1 8l4.5 4.5 1-1L3 8l3.5-3.5-1-1zm5 0l-1 1L13 8l-3.5 3.5 1 1L15 8l-4.5-4.5z"/>
    </svg>`,
  },
  email: {
    accent: '#6366f1',
    glow: 'rgba(99, 102, 241, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 9.5c-.3 0-.6-.1-.8-.3L2 4.5V12h12V4.5l-5.2 4.7c-.2.2-.5.3-.8.3zM2 3h12l-6 5.4L2 3z"/>
    </svg>`,
  },
  shopping: {
    accent: '#ec4899',
    glow: 'rgba(236, 72, 153, 0.35)',
    icon: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor">
      <path d="M6 1l-1.5 4H1l3 2.5-1 4L6 9l2 1.5L10 9l3 3-1-4L15 5h-3.5L10 1H6zm0 1.5h4l1 2.5h2l-2 1.5.7 2.8L10 8l-2 1.5L6 8 4.3 9.3 5 6.5 3 5h2L6 2.5z"/>
    </svg>`,
  },
};

// ── Context actions ───────────────────────────────────────────────────────────

export const CONTEXT_ACTIONS: Record<PageContext, string[]> = {
  general:  ['ask', 'summarize', 'explain', 'translate', 'save', 'copy'],
  article:  ['ask', 'summarize', 'highlight', 'translate', 'save', 'explain', 'copy'],
  media:    ['ask', 'describe', 'transcript', 'summarize', 'translate', 'save'],
  code:     ['ask', 'explain', 'review', 'debug', 'refactor', 'copy', 'save'],
  email:    ['ask', 'summarize', 'reply', 'translate', 'save', 'action-items'],
  shopping: ['ask', 'compare', 'summarize', 'pros-cons', 'save', 'price-history'],
};

// ── Animation tokens ──────────────────────────────────────────────────────────

export const ANIM = {
  morphDuration: 350,
  morphEasing: 'cubic-bezier(0.4, 0, 0.0, 1)',
  flowDuration: 500,
  glowPulse: 3000,
  sequentialStagger: 40,
  dismissDelay: 400,
  crossFadeDuration: 400,
  selectionDelay: 300,
  imageHoverDelay: 800,
} as const;

// ── Glassmorphism tokens ──────────────────────────────────────────────────────

export const GLASS = {
  bg: 'rgba(10, 8, 24, 0.88)',
  bgHeavy: 'rgba(10, 8, 24, 0.75)',
  backdrop: 'blur(20px) saturate(1.5)',
  borderOpacity: 0.25,
  shadowBase: '0 8px 32px rgba(0,0,0,0.4)',
} as const;

// ── Ghost bar tokens ──────────────────────────────────────────────────────────

export const GHOST_BAR = {
  height: 28,
  iconSize: 15,
  imageIconSize: 16,
  imageBarHeight: 32,
  maxActionsPerRow: 7,
} as const;

// ── Modal tokens ──────────────────────────────────────────────────────────────

export const MODAL = {
  maxWidth: 520,
  maxHeight: 480,
  previewMaxLines: 6,
  previewMaxChars: 2000,
  imagePreviewMaxHeight: 200,
} as const;

// ── FAB tokens ────────────────────────────────────────────────────────────────

export const FAB = {
  pillPadding: '6px 10px',
  glowIntensityMin: 0.15,
  glowIntensityMax: 0.35,
  logoSize: 20,
  expandDuration: 220,
  dragThreshold: 4,
  edgeMargin: 12,
} as const;

// ── Typography ────────────────────────────────────────────────────────────────

export const FONT_STACK =
  "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";

// ── Z-index ───────────────────────────────────────────────────────────────────

export const Z_TOP = 2147483647;
export const Z_MID = 2147483645;
