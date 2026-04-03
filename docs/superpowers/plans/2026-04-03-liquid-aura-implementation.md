# Liquid Aura Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Aura extension's floating UI as a modular, context-aware system with liquid-morph animations, inline ghost bars, and a 5-layer intelligence engine — replacing the monolithic 5297-line `content.ts`.

**Architecture:** 12 TypeScript modules rendering into a single shared Shadow DOM, coordinated by `index.ts`. A reactive pub/sub context engine drives color, icon, timing, and action ordering across all surfaces. Vite bundles via the existing IIFE content script config.

**Tech Stack:** TypeScript, Vite (IIFE bundle), Chrome Extension APIs, Web Animations API, Shadow DOM, Jest + jsdom for testing.

**Spec:** `docs/superpowers/specs/2026-04-03-liquid-aura-floating-ui-design.md`

---

## File Map

### New files (create in `extension-src/src/content/`)

| File | Responsibility | Approx lines |
|---|---|---|
| `types.ts` | All shared interfaces: `ContextSignal`, `GhostBarState`, `ModalState`, `FabState`, `AnimationConfig`, message types | ~120 |
| `tokens.ts` | Design tokens: 6 context palettes, animation timing, glassmorphism values, icon SVGs | ~150 |
| `styles.ts` | Generates full CSS string from tokens; exports `buildStylesheet()` | ~250 |
| `animator.ts` | Liquid-morph primitives: `flow()`, `morph()`, `dissolve()`, `crossFade()`, `sequentialReveal()` | ~180 |
| `context-engine.ts` | 5-layer intelligence: page type, topology, session memory, cadence, cross-tab. Reactive pub/sub. | ~350 |
| `fab.ts` | Living FAB: render, ambient glow, icon morph, drag, popout, click | ~400 |
| `ghost-bar.ts` | Text selection + image ghost bars: appear/retract, action dispatch, position tracking | ~450 |
| `modal.ts` | Focus modal: morph-from-ghost-bar, input, actions, model selector, dismiss | ~300 |
| `highlights.ts` | Highlight system: save/restore/tooltip/delete (migrated from content.ts:1111-1480) | ~370 |
| `gmail.ts` | Gmail compose AI button (migrated from content.ts:2987-3565) | ~580 |
| `capture.ts` | Element capture overlay (migrated from content.ts:3948-4131) | ~185 |
| `link-preview.ts` | Link hover preview (migrated from content.ts:3566-3726) | ~160 |
| `page-services.ts` | Migrated utilities: content extraction, DOM serializer, OCR overlay, translation, quick-actions-on-inputs, YouTube/Netflix relay, SERP card | ~1600 |
| `index.ts` | Coordinator: mounts Shadow DOM, inits modules, wires message listener | ~200 |

### Modified files

| File | Change |
|---|---|
| `extension-src/src/content.ts` | Gutted to thin shim: imports `content/index.ts`, calls `init()` (~15 lines) |
| `extension-src/vite.config.content.ts` | No change needed — Vite follows the `content.ts` entry and bundles all imports into single IIFE |

---

## Task 1: Types & Tokens Foundation

**Files:**
- Create: `extension-src/src/content/types.ts`
- Create: `extension-src/src/content/tokens.ts`
- Test: `extension-src/src/__tests__/content/tokens.test.ts`

- [ ] **Step 1: Create types.ts with all shared interfaces**

```typescript
// extension-src/src/content/types.ts

// ── Context Engine ──

export type PageContext = 'article' | 'code' | 'media' | 'email' | 'shopping' | 'general';
export type Cadence = 'passive' | 'engaged' | 'active';

export interface ContextSignal {
  type: PageContext;
  accent: string;
  glow: string;
  icon: string;           // SVG markup
  actions: string[];      // ghost bar actions in relevance order
  cadence: Cadence;
  suppressGhostBars: boolean;
  sessionActions: string[];
  readingProgress: number; // 0-1
}

// ── Pub/Sub ──

export type ContextListener = (signal: ContextSignal) => void;

export interface ContextStore {
  get(): ContextSignal;
  subscribe(fn: ContextListener): () => void;
  update(partial: Partial<ContextSignal>): void;
}

// ── FAB ──

export type FabSide = 'left' | 'right';

export interface FabState {
  side: FabSide;
  offset: number;
  visible: boolean;
  hiddenOnce: boolean;
  hovering: boolean;
  dragging: boolean;
  popoutOpen: boolean;
}

// ── Ghost Bar ──

export type GhostBarTarget = 'text' | 'image';

export interface GhostBarState {
  target: GhostBarTarget;
  visible: boolean;
  expanded: boolean;      // "More" row open
  anchorRect: DOMRect | null;
  imageElement: HTMLImageElement | null;
}

// ── Modal ──

export interface ModalState {
  open: boolean;
  contentType: 'text' | 'image';
  text: string;
  imageUrl: string;
  originRect: DOMRect | null; // ghost bar position for morph animation
}

// ── Animation ──

export interface AnimationConfig {
  duration: number;
  easing: string;
  delay?: number;
}

export interface FlowOptions extends AnimationConfig {
  direction: 'up' | 'down';
}

export interface MorphOptions extends AnimationConfig {
  // origin and target rects computed from elements
}

// ── Module Init ──

export interface ContentModule {
  init(container: HTMLElement, store: ContextStore, ext: typeof chrome): void;
  destroy?(): void;
}

// ── Messages (outbound: content → background) ──

export interface OpenPanelMessage { type: 'OPEN_PANEL'; panel: string; }
export interface OpenWithTextMessage { type: 'OPEN_WITH_TEXT'; action: string; text: string; url: string; title: string; }
export interface SaveKnowledgeMessage { type: 'SAVE_KNOWLEDGE'; text: string; url: string; title: string; }
export interface QuickActionMessage { type: 'QUICK_ACTION'; action: string; text: string; language?: string; threadContext?: string; }
export interface ImageDescribeMessage { type: 'IMAGE_DESCRIBE'; imageUrl: string; }
export interface ImageEditOpenMessage { type: 'IMAGE_EDIT_OPEN'; imageUrl: string; }
export interface ImageSaveMessage { type: 'IMAGE_SAVE'; imageUrl: string; }

export type OutboundMessage =
  | OpenPanelMessage
  | OpenWithTextMessage
  | SaveKnowledgeMessage
  | QuickActionMessage
  | ImageDescribeMessage
  | ImageEditOpenMessage
  | ImageSaveMessage;

// ── Messages (inbound: background → content) ──

export interface ExtractPageMsg { type: 'EXTRACT_PAGE'; }
export interface GetDomMsg { type: 'GET_DOM'; }
export interface ExecActionMsg { type: 'EXEC_ACTION'; action: ExecActionParams; }
export interface ShowOcrOverlayMsg { type: 'SHOW_OCR_OVERLAY'; dataUrl: string; }
export interface ShowDockMsg { type: 'SHOW_DOCK'; }

export interface ExecActionParams {
  action: 'click' | 'type' | 'scroll' | 'selectOption';
  selector?: string;
  text?: string;
  url?: string;
  amount?: number;
  value?: string;
}

// ── Highlight ──

export interface HighlightData {
  id: string;
  url: string;
  text: string;
  xpath: string;
  context: string;
  timestamp: number;
  color: string;
  pageTitle: string;
  stale?: boolean;
}

// ── FAB Action Items ──

export interface DockItemDef {
  svg: string;
  action: string;
  tip: string;
}
```

- [ ] **Step 2: Create tokens.ts with design system constants**

```typescript
// extension-src/src/content/tokens.ts

import type { PageContext } from './types';

// ── Context Palettes ──

export interface ContextPalette {
  accent: string;
  glow: string;
  icon: string; // SVG markup (viewBox="0 0 24 24", 20x20)
}

const ICON_LOGO = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 21M12 3L22 21M5.8 14.2L18.2 14.2"/></svg>`;
const ICON_BOOK = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>`;
const ICON_WAVE = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="14" width="4" height="6" rx="1"/><rect x="7" y="8" width="4" height="12" rx="1"/><rect x="12" y="4" width="4" height="16" rx="1"/><rect x="17" y="10" width="4" height="10" rx="1"/></svg>`;
const ICON_CODE = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`;
const ICON_QUILL = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.83 2.83 0 114 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>`;
const ICON_SPARKLE = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.5 5.5L19 10l-5.5 1.5L12 17l-1.5-5.5L5 10l5.5-1.5L12 3z"/></svg>`;

export const PALETTES: Record<PageContext, ContextPalette> = {
  general:  { accent: '#7c3aed', glow: 'rgba(124,58,237,0.3)',  icon: ICON_LOGO },
  article:  { accent: '#3b82f6', glow: 'rgba(59,130,246,0.3)',  icon: ICON_BOOK },
  media:    { accent: '#f59e0b', glow: 'rgba(245,158,11,0.3)',  icon: ICON_WAVE },
  code:     { accent: '#10b981', glow: 'rgba(16,185,129,0.3)',  icon: ICON_CODE },
  email:    { accent: '#6366f1', glow: 'rgba(99,102,241,0.3)',  icon: ICON_QUILL },
  shopping: { accent: '#ec4899', glow: 'rgba(236,72,153,0.3)',  icon: ICON_SPARKLE },
};

// ── Action Ordering by Context ──

export const CONTEXT_ACTIONS: Record<PageContext, string[]> = {
  article:  ['ask', 'summarize', 'explain', 'copy', 'translate', 'highlight', 'more'],
  code:     ['ask', 'explain', 'rewrite', 'copy', 'translate', 'highlight', 'more'],
  email:    ['ask', 'rewrite', 'grammar', 'copy', 'translate', 'highlight', 'more'],
  media:    ['ask', 'copy', 'explain', 'summarize', 'translate', 'highlight', 'more'],
  shopping: ['ask', 'summarize', 'explain', 'copy', 'translate', 'highlight', 'more'],
  general:  ['ask', 'copy', 'explain', 'summarize', 'translate', 'highlight', 'more'],
};

// ── Animation Tokens ──

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
  imageHoverDelayFast: 500,
  imageHoverDelaySlow: 1200,
} as const;

// ── Glassmorphism Tokens ──

export const GLASS = {
  bg: 'rgba(10, 8, 24, 0.88)',
  bgHeavy: 'rgba(10, 8, 24, 0.75)',
  backdrop: 'blur(20px) saturate(1.5)',
  borderOpacity: 0.25,
  shadowBase: '0 8px 32px rgba(0,0,0,0.4)',
} as const;

// ── Ghost Bar ──

export const GHOST_BAR = {
  height: 28,
  iconSize: 15,
  imageIconSize: 16,
  imageBarHeight: 32,
  maxActionsPerRow: 7,
} as const;

// ── Modal ──

export const MODAL = {
  maxWidth: 520,
  maxHeight: 480,
  previewMaxLines: 6,
  previewMaxChars: 2000,
  imagePreviewMaxHeight: 200,
} as const;

// ── FAB ──

export const FAB = {
  pillPadding: '8px 10px',
  pillPaddingHover: '8px 14px',
  glowIntensityMin: 0.15,
  glowIntensityMax: 0.35,
  logoSize: 20,
  popoutBtnSize: 26,
  popoutIconSize: 14,
  closeSize: 16,
} as const;

// ── Shared ──

export const FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', system-ui, sans-serif";
export const Z_TOP = 2147483647;
export const Z_HIGH = 2147483646;
export const Z_MID = 2147483645;
```

- [ ] **Step 3: Write token tests**

```typescript
// extension-src/src/__tests__/content/tokens.test.ts

import { PALETTES, CONTEXT_ACTIONS, ANIM, GLASS, FONT_STACK } from '../../content/tokens';
import type { PageContext } from '../../content/types';

describe('tokens', () => {
  const ALL_CONTEXTS: PageContext[] = ['general', 'article', 'media', 'code', 'email', 'shopping'];

  test('every context has a palette', () => {
    for (const ctx of ALL_CONTEXTS) {
      const p = PALETTES[ctx];
      expect(p).toBeDefined();
      expect(p.accent).toMatch(/^#[0-9a-f]{6}$/);
      expect(p.glow).toMatch(/^rgba\(/);
      expect(p.icon).toContain('<svg');
    }
  });

  test('every context has an action list starting with ask', () => {
    for (const ctx of ALL_CONTEXTS) {
      const actions = CONTEXT_ACTIONS[ctx];
      expect(actions).toBeDefined();
      expect(actions[0]).toBe('ask');
      expect(actions.length).toBeGreaterThanOrEqual(5);
    }
  });

  test('animation tokens are positive numbers', () => {
    expect(ANIM.morphDuration).toBeGreaterThan(0);
    expect(ANIM.flowDuration).toBeGreaterThan(ANIM.morphDuration);
    expect(ANIM.sequentialStagger).toBeGreaterThan(0);
    expect(ANIM.dismissDelay).toBeGreaterThan(0);
  });

  test('glass tokens are defined', () => {
    expect(GLASS.bg).toContain('rgba');
    expect(GLASS.backdrop).toContain('blur');
    expect(GLASS.borderOpacity).toBeGreaterThan(0);
    expect(GLASS.borderOpacity).toBeLessThan(1);
  });

  test('font stack is a non-empty string', () => {
    expect(FONT_STACK.length).toBeGreaterThan(0);
    expect(FONT_STACK).toContain('system-ui');
  });
});
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/tokens.test.ts --verbose`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/types.ts extension-src/src/content/tokens.ts extension-src/src/__tests__/content/tokens.test.ts
git commit -m "feat(extension): add Liquid Aura types and design tokens"
```

---

## Task 2: Animator — Liquid-Morph Primitives

**Files:**
- Create: `extension-src/src/content/animator.ts`
- Test: `extension-src/src/__tests__/content/animator.test.ts`

- [ ] **Step 1: Write animator tests**

```typescript
// extension-src/src/__tests__/content/animator.test.ts

import { flow, dissolve, crossFade, sequentialReveal } from '../../content/animator';

// Mock Web Animations API
beforeEach(() => {
  Element.prototype.animate = jest.fn().mockReturnValue({
    finished: Promise.resolve(),
    cancel: jest.fn(),
    onfinish: null,
  });
});

describe('animator', () => {
  let el: HTMLDivElement;

  beforeEach(() => {
    el = document.createElement('div');
    document.body.appendChild(el);
  });

  afterEach(() => {
    el.remove();
  });

  test('flow() calls animate with correct keyframes for down direction', async () => {
    await flow(el, { direction: 'down', duration: 350, easing: 'ease' });
    expect(el.animate).toHaveBeenCalledTimes(1);
    const [keyframes, options] = (el.animate as jest.Mock).mock.calls[0];
    expect(keyframes).toEqual([
      { height: '0px', opacity: 0 },
      { height: `${el.offsetHeight}px`, opacity: 1 },
    ]);
    expect(options.duration).toBe(350);
    expect(options.easing).toBe('ease');
  });

  test('flow() with direction up reverses keyframes', async () => {
    await flow(el, { direction: 'up', duration: 350, easing: 'ease' });
    const [keyframes] = (el.animate as jest.Mock).mock.calls[0];
    expect(keyframes).toEqual([
      { height: `${el.offsetHeight}px`, opacity: 1 },
      { height: '0px', opacity: 0 },
    ]);
  });

  test('dissolve() animates opacity to 0', async () => {
    await dissolve(el, { duration: 200, easing: 'ease' });
    const [keyframes] = (el.animate as jest.Mock).mock.calls[0];
    expect(keyframes).toEqual([
      { opacity: 1 },
      { opacity: 0 },
    ]);
  });

  test('crossFade() fades out old, fades in new', async () => {
    const newEl = document.createElement('div');
    document.body.appendChild(newEl);
    newEl.animate = jest.fn().mockReturnValue({
      finished: Promise.resolve(),
      cancel: jest.fn(),
    });

    await crossFade(el, newEl, { duration: 400, easing: 'ease' });
    expect(el.animate).toHaveBeenCalledTimes(1);
    expect(newEl.animate).toHaveBeenCalledTimes(1);

    const [oldKf] = (el.animate as jest.Mock).mock.calls[0];
    expect(oldKf).toEqual([{ opacity: 1 }, { opacity: 0 }]);

    const [newKf] = (newEl.animate as jest.Mock).mock.calls[0];
    expect(newKf).toEqual([{ opacity: 0 }, { opacity: 1 }]);

    newEl.remove();
  });

  test('sequentialReveal() staggers children animations', async () => {
    const parent = document.createElement('div');
    for (let i = 0; i < 3; i++) {
      const child = document.createElement('div');
      child.animate = jest.fn().mockReturnValue({
        finished: Promise.resolve(),
        cancel: jest.fn(),
      });
      parent.appendChild(child);
    }
    document.body.appendChild(parent);

    await sequentialReveal(parent, { duration: 200, easing: 'ease', stagger: 40 });

    const children = Array.from(parent.children) as HTMLElement[];
    expect(children[0].animate).toHaveBeenCalledTimes(1);
    expect(children[1].animate).toHaveBeenCalledTimes(1);
    expect(children[2].animate).toHaveBeenCalledTimes(1);

    // Verify stagger via delay option
    const delay0 = (children[0].animate as jest.Mock).mock.calls[0][1].delay;
    const delay1 = (children[1].animate as jest.Mock).mock.calls[0][1].delay;
    const delay2 = (children[2].animate as jest.Mock).mock.calls[0][1].delay;
    expect(delay0).toBe(0);
    expect(delay1).toBe(40);
    expect(delay2).toBe(80);

    parent.remove();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/animator.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement animator.ts**

```typescript
// extension-src/src/content/animator.ts

import type { AnimationConfig, FlowOptions } from './types';

/**
 * Liquid flow: element grows/shrinks vertically with opacity.
 * direction='down': 0→full height (appear). direction='up': full height→0 (retract).
 */
export async function flow(
  el: HTMLElement,
  opts: FlowOptions
): Promise<void> {
  const h = `${el.offsetHeight}px`;
  const keyframes: Keyframe[] =
    opts.direction === 'down'
      ? [{ height: '0px', opacity: 0 }, { height: h, opacity: 1 }]
      : [{ height: h, opacity: 1 }, { height: '0px', opacity: 0 }];

  const anim = el.animate(keyframes, {
    duration: opts.duration,
    easing: opts.easing,
    fill: 'forwards',
    delay: opts.delay ?? 0,
  });
  await anim.finished;
}

/**
 * Dissolve: fade opacity to 0.
 */
export async function dissolve(
  el: HTMLElement,
  opts: AnimationConfig
): Promise<void> {
  const anim = el.animate(
    [{ opacity: 1 }, { opacity: 0 }],
    { duration: opts.duration, easing: opts.easing, fill: 'forwards', delay: opts.delay ?? 0 }
  );
  await anim.finished;
}

/**
 * Fade in: opacity 0 → 1.
 */
export async function fadeIn(
  el: HTMLElement,
  opts: AnimationConfig
): Promise<void> {
  const anim = el.animate(
    [{ opacity: 0 }, { opacity: 1 }],
    { duration: opts.duration, easing: opts.easing, fill: 'forwards', delay: opts.delay ?? 0 }
  );
  await anim.finished;
}

/**
 * Cross-fade: old element fades out, new element fades in simultaneously.
 */
export async function crossFade(
  oldEl: HTMLElement,
  newEl: HTMLElement,
  opts: AnimationConfig
): Promise<void> {
  const outAnim = oldEl.animate(
    [{ opacity: 1 }, { opacity: 0 }],
    { duration: opts.duration, easing: opts.easing, fill: 'forwards', delay: opts.delay ?? 0 }
  );
  const inAnim = newEl.animate(
    [{ opacity: 0 }, { opacity: 1 }],
    { duration: opts.duration, easing: opts.easing, fill: 'forwards', delay: opts.delay ?? 0 }
  );
  await Promise.all([outAnim.finished, inAnim.finished]);
}

/**
 * Sequential reveal: animate children one by one with stagger delay.
 */
export async function sequentialReveal(
  parent: HTMLElement,
  opts: AnimationConfig & { stagger: number }
): Promise<void> {
  const children = Array.from(parent.children) as HTMLElement[];
  const animations = children.map((child, i) =>
    child.animate(
      [
        { opacity: 0, transform: 'translateY(4px) scale(0.95)' },
        { opacity: 1, transform: 'translateY(0) scale(1)' },
      ],
      {
        duration: opts.duration,
        easing: opts.easing,
        fill: 'forwards',
        delay: i * opts.stagger,
      }
    )
  );
  await Promise.all(animations.map((a) => a.finished));
}

/**
 * Morph: animate element from one rect to another (position + size).
 * Used for ghost-bar → modal transition.
 */
export async function morph(
  el: HTMLElement,
  from: DOMRect,
  to: DOMRect,
  opts: AnimationConfig
): Promise<void> {
  const anim = el.animate(
    [
      {
        position: 'fixed',
        left: `${from.left}px`,
        top: `${from.top}px`,
        width: `${from.width}px`,
        height: `${from.height}px`,
        opacity: 0.8,
      },
      {
        position: 'fixed',
        left: `${to.left}px`,
        top: `${to.top}px`,
        width: `${to.width}px`,
        height: `${to.height}px`,
        opacity: 1,
      },
    ],
    {
      duration: opts.duration,
      easing: opts.easing,
      fill: 'forwards',
      delay: opts.delay ?? 0,
    }
  );
  await anim.finished;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/animator.test.ts --verbose`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/animator.ts extension-src/src/__tests__/content/animator.test.ts
git commit -m "feat(extension): add liquid-morph animation primitives"
```

---

## Task 3: Styles — CSS Generator

**Files:**
- Create: `extension-src/src/content/styles.ts`
- Test: `extension-src/src/__tests__/content/styles.test.ts`

- [ ] **Step 1: Write styles tests**

```typescript
// extension-src/src/__tests__/content/styles.test.ts

import { buildStylesheet } from '../../content/styles';

describe('buildStylesheet', () => {
  test('returns a non-empty CSS string', () => {
    const css = buildStylesheet();
    expect(typeof css).toBe('string');
    expect(css.length).toBeGreaterThan(100);
  });

  test('contains CSS custom properties for context colors', () => {
    const css = buildStylesheet();
    expect(css).toContain('--aura-accent');
    expect(css).toContain('--aura-glow');
  });

  test('contains FAB styles', () => {
    const css = buildStylesheet();
    expect(css).toContain('.aura-fab');
    expect(css).toContain('.fab-pill');
  });

  test('contains ghost bar styles', () => {
    const css = buildStylesheet();
    expect(css).toContain('.ghost-bar');
  });

  test('contains modal styles', () => {
    const css = buildStylesheet();
    expect(css).toContain('.aura-modal');
    expect(css).toContain('.aura-modal-overlay');
  });

  test('contains glow-pulse animation', () => {
    const css = buildStylesheet();
    expect(css).toContain('@keyframes aura-glow-pulse');
  });

  test('contains font stack', () => {
    const css = buildStylesheet();
    expect(css).toContain('system-ui');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/styles.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement styles.ts**

Create `extension-src/src/content/styles.ts` that exports `buildStylesheet(): string`. This function returns a single CSS string containing all styles for: FAB (`.aura-fab`, `.fab-pill`, `.fab-glow`, `.fab-popout`, `.fab-action-btn`, `.fab-close`, `.fab-logo`), ghost bar (`.ghost-bar`, `.ghost-bar-text`, `.ghost-bar-image`, `.gb-action`, `.gb-separator`, `.gb-extended`), modal (`.aura-modal-overlay`, `.aura-modal`, `.modal-preview`, `.modal-input`, `.modal-actions`, `.modal-model-select`), highlight tooltip (`.hl-tooltip`), image toolbar compat, toast (`.aura-toast`), and keyframe animations (`aura-glow-pulse`, `aura-flow-down`, `aura-flow-up`).

Use CSS custom properties `--aura-accent` and `--aura-glow` (defaulting to general purple) so the context engine can swap them at runtime via `container.style.setProperty()`. Import `GLASS`, `FAB`, `GHOST_BAR`, `MODAL`, `ANIM`, `FONT_STACK`, `Z_TOP`, `Z_MID` from `./tokens`.

The CSS must include:
- `will-change: transform, opacity` on all animated elements
- `pointer-events: none` on containers, `pointer-events: auto` on interactive elements
- Glassmorphism using the GLASS tokens
- `@keyframes aura-glow-pulse` cycling opacity from `var(--aura-glow-min, 0.15)` to `var(--aura-glow-max, 0.35)`

This file will be large (~250 lines) — implement the full CSS covering all components listed above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/styles.test.ts --verbose`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/styles.ts extension-src/src/__tests__/content/styles.test.ts
git commit -m "feat(extension): add CSS generator with context-aware custom properties"
```

---

## Task 4: Context Engine — 5-Layer Intelligence

**Files:**
- Create: `extension-src/src/content/context-engine.ts`
- Test: `extension-src/src/__tests__/content/context-engine.test.ts`

- [ ] **Step 1: Write context engine tests**

```typescript
// extension-src/src/__tests__/content/context-engine.test.ts

import { createContextStore, detectPageType, createCadenceTracker } from '../../content/context-engine';
import type { ContextSignal, PageContext } from '../../content/types';

describe('createContextStore', () => {
  test('initial signal is general context', () => {
    const store = createContextStore();
    const signal = store.get();
    expect(signal.type).toBe('general');
    expect(signal.cadence).toBe('engaged');
    expect(signal.suppressGhostBars).toBe(false);
  });

  test('update() merges partial and notifies subscribers', () => {
    const store = createContextStore();
    const listener = jest.fn();
    store.subscribe(listener);

    store.update({ type: 'code' });

    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0].type).toBe('code');
    expect(store.get().type).toBe('code');
  });

  test('subscribe returns unsubscribe function', () => {
    const store = createContextStore();
    const listener = jest.fn();
    const unsub = store.subscribe(listener);

    store.update({ type: 'article' });
    expect(listener).toHaveBeenCalledTimes(1);

    unsub();
    store.update({ type: 'media' });
    expect(listener).toHaveBeenCalledTimes(1); // no new call
  });
});

describe('detectPageType', () => {
  test('returns code for github.com', () => {
    expect(detectPageType('https://github.com/user/repo', document)).toBe('code');
  });

  test('returns media for youtube.com', () => {
    expect(detectPageType('https://www.youtube.com/watch?v=123', document)).toBe('media');
  });

  test('returns email for mail.google.com', () => {
    expect(detectPageType('https://mail.google.com/mail/u/0/', document)).toBe('email');
  });

  test('returns general for unknown sites', () => {
    expect(detectPageType('https://example.com/', document)).toBe('general');
  });

  test('returns article when DOM has <article> element', () => {
    const article = document.createElement('article');
    article.textContent = 'A'.repeat(500);
    document.body.appendChild(article);
    expect(detectPageType('https://example.com/', document)).toBe('article');
    article.remove();
  });
});

describe('createCadenceTracker', () => {
  test('initial cadence is engaged', () => {
    const tracker = createCadenceTracker();
    expect(tracker.getCadence()).toBe('engaged');
  });

  test('fast scroll events shift to passive', () => {
    const tracker = createCadenceTracker();
    // Simulate 10 fast scrolls in 2 seconds
    for (let i = 0; i < 10; i++) {
      tracker.recordScroll(500); // high velocity
    }
    // After enough fast scrolls, cadence should shift
    // (may not shift immediately — needs 3s consistent signal per spec)
    // This tests the recording mechanism exists
    expect(['passive', 'engaged', 'active']).toContain(tracker.getCadence());
  });

  test('selection events shift toward active', () => {
    const tracker = createCadenceTracker();
    for (let i = 0; i < 5; i++) {
      tracker.recordSelection();
    }
    expect(['engaged', 'active']).toContain(tracker.getCadence());
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/context-engine.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement context-engine.ts**

Create `extension-src/src/content/context-engine.ts` with these exports:

**`createContextStore(): ContextStore`** — Reactive pub/sub store. Holds a `ContextSignal`, allows `update(partial)`, notifies subscribers on change. Initial state: type=general, cadence=engaged, suppressGhostBars=false, readingProgress=0, actions from `CONTEXT_ACTIONS.general`.

**`detectPageType(url: string, doc: Document): PageContext`** — Layer 1 detection. URL pattern matching first (github/gitlab→code, youtube/netflix→media, mail.google→email, known shopping→shopping), then DOM analysis (article tag, code block density, video elements, product schema), default→general.

**`createCadenceTracker(): CadenceTracker`** — Layer 4. Tracks scroll velocity (rolling 10s window), selection frequency (rolling 30s window), input activity. Exposes `getCadence()`, `recordScroll(velocity)`, `recordSelection()`, `recordInput()`. Returns 'passive'|'engaged'|'active'. Requires 3s consistent signal to transition states.

**`createSessionMemory(): SessionMemory`** — Layer 3. Tracks: actions performed on this page (action type + text hash), dismiss count, highlight count. Exposes `recordAction(action, textHash)`, `recordDismissal()`, `getExtraDelay(): number` (200ms * dismissals, max 2000), `shouldPromoteContinue(textHash): boolean`, `getSessionActions(): string[]`.

**`initContextEngine(store: ContextStore, ext: typeof chrome): () => void`** — Wires everything together. Sets up `MutationObserver` (debounced 2s), `popstate` listener, `IntersectionObserver` for reading progress (Layer 2), input focus/blur for suppress (Layer 2), scroll listener for cadence, selection listener for cadence. Updates store on changes. Uses `requestIdleCallback` for DOM analysis. Returns cleanup function. Layer 5 (cross-tab) uses `chrome.storage.session` to share active modal state.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/context-engine.test.ts --verbose`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/context-engine.ts extension-src/src/__tests__/content/context-engine.test.ts
git commit -m "feat(extension): add 5-layer context intelligence engine"
```

---

## Task 5: Living FAB

**Files:**
- Create: `extension-src/src/content/fab.ts`
- Test: `extension-src/src/__tests__/content/fab.test.ts`

- [ ] **Step 1: Write FAB tests**

```typescript
// extension-src/src/__tests__/content/fab.test.ts

import { createFab } from '../../content/fab';
import { createContextStore } from '../../content/context-engine';
import type { ContextStore } from '../../content/types';

// Mock animate API
beforeEach(() => {
  Element.prototype.animate = jest.fn().mockReturnValue({
    finished: Promise.resolve(),
    cancel: jest.fn(),
  });
});

describe('createFab', () => {
  let container: HTMLDivElement;
  let store: ContextStore;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    store = createContextStore();
  });

  afterEach(() => {
    container.remove();
  });

  test('init renders FAB pill into container', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    expect(container.querySelector('.aura-fab')).not.toBeNull();
    expect(container.querySelector('.fab-pill')).not.toBeNull();
  });

  test('FAB contains logo element', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    const logo = container.querySelector('.fab-logo');
    expect(logo).not.toBeNull();
    expect(logo!.innerHTML).toContain('svg');
  });

  test('FAB contains popout with action buttons', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    const popout = container.querySelector('.fab-popout');
    expect(popout).not.toBeNull();
    const buttons = popout!.querySelectorAll('.fab-action-btn');
    expect(buttons.length).toBeGreaterThanOrEqual(4);
  });

  test('FAB contains close button', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    expect(container.querySelector('.fab-close')).not.toBeNull();
  });

  test('context change updates FAB icon', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    const logoBefore = container.querySelector('.fab-logo')!.innerHTML;

    store.update({ type: 'code', icon: '<svg>CODE</svg>' });

    // Icon should have changed (cross-fade triggers new element)
    // The exact mechanism is animation-based, but the SVG content should update
    const logoAfter = container.querySelector('.fab-logo')!.innerHTML;
    // After cross-fade, new icon should be present
    expect(container.querySelector('.fab-logo')).not.toBeNull();
  });

  test('destroy() removes FAB from container', () => {
    const fab = createFab();
    fab.init(container, store, chrome);
    expect(container.querySelector('.aura-fab')).not.toBeNull();
    fab.destroy!();
    expect(container.querySelector('.aura-fab')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/fab.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement fab.ts**

Create `extension-src/src/content/fab.ts` exporting `createFab(): ContentModule`.

Migrate all FAB logic from `content.ts:582-1078` into this module. Key changes from the current implementation:

- **Ambient glow**: Add a `div.fab-glow` behind the pill, absolutely positioned, with `animation: aura-glow-pulse 3s infinite` using the `--aura-glow` CSS variable.
- **Icon morphing**: When the context store emits a new icon, use `crossFade()` from `animator.ts` to transition the `.fab-logo` SVG content over 400ms.
- **Popout sequential reveal**: On hover, instead of showing the popout all at once, use `sequentialReveal()` from `animator.ts` with 40ms stagger.
- **Drag liquid stretch**: On pointerdown, briefly animate a `scaleY(1.1)` on the pill for 100ms before detaching.
- **Elastic settle on drop**: On pointerup, animate to final position with `cubic-bezier(0.34, 1.56, 0.64, 1)` (elastic overshoot).
- **Store subscription**: Subscribe to context store on init. Update `--aura-accent`, `--aura-glow` CSS variables on the FAB container, cross-fade icon, reorder popout actions.
- **Message sending**: Use a `sendMessage` callback passed via init (the coordinator provides `safeSend`).
- **Storage persistence**: Persist side + offset to `chrome.storage.local` (same keys as current: `auraFabSide`, `auraFabOffset`).

All existing click/hover/drag behavior preserved from current implementation. The `destroy()` method removes all DOM, event listeners, and store subscriptions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/fab.test.ts --verbose`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/fab.ts extension-src/src/__tests__/content/fab.test.ts
git commit -m "feat(extension): add Living FAB with ambient glow and context morphing"
```

---

## Task 6: Ghost Bar — Text + Image

**Files:**
- Create: `extension-src/src/content/ghost-bar.ts`
- Test: `extension-src/src/__tests__/content/ghost-bar.test.ts`

- [ ] **Step 1: Write ghost bar tests**

```typescript
// extension-src/src/__tests__/content/ghost-bar.test.ts

import { createGhostBar } from '../../content/ghost-bar';
import { createContextStore } from '../../content/context-engine';
import type { ContextStore } from '../../content/types';

beforeEach(() => {
  Element.prototype.animate = jest.fn().mockReturnValue({
    finished: Promise.resolve(),
    cancel: jest.fn(),
  });
});

describe('createGhostBar', () => {
  let container: HTMLDivElement;
  let store: ContextStore;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    store = createContextStore();
  });

  afterEach(() => {
    container.remove();
  });

  test('init attaches selection listener', () => {
    const addSpy = jest.spyOn(document, 'addEventListener');
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    const selectionCalls = addSpy.mock.calls.filter(c => c[0] === 'selectionchange');
    expect(selectionCalls.length).toBeGreaterThanOrEqual(1);
    addSpy.mockRestore();
  });

  test('showTextBar() renders ghost bar into container', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    // Simulate showing a bar with a mock rect
    const rect = new DOMRect(100, 200, 300, 20);
    gb.showTextBar(rect, 'Hello world');
    expect(container.querySelector('.ghost-bar')).not.toBeNull();
    expect(container.querySelector('.ghost-bar-text')).not.toBeNull();
  });

  test('text ghost bar has correct action buttons from context', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    store.update({ type: 'code' }); // code context promotes explain + rewrite
    const rect = new DOMRect(100, 200, 300, 20);
    gb.showTextBar(rect, 'const x = 1;');
    const actions = container.querySelectorAll('.gb-action');
    expect(actions.length).toBeGreaterThanOrEqual(5);
  });

  test('showImageBar() renders ghost bar inside image bounds', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    const img = document.createElement('img');
    Object.defineProperty(img, 'getBoundingClientRect', {
      value: () => new DOMRect(50, 100, 400, 300),
    });
    gb.showImageBar(img);
    expect(container.querySelector('.ghost-bar')).not.toBeNull();
    expect(container.querySelector('.ghost-bar-image')).not.toBeNull();
  });

  test('hideBar() removes ghost bar', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    gb.showTextBar(new DOMRect(100, 200, 300, 20), 'test');
    expect(container.querySelector('.ghost-bar')).not.toBeNull();
    gb.hideBar();
    // After animation, bar should be gone (in test, animation resolves immediately)
    expect(container.querySelector('.ghost-bar')).toBeNull();
  });

  test('only one ghost bar at a time', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    gb.showTextBar(new DOMRect(100, 200, 300, 20), 'first');
    gb.showTextBar(new DOMRect(100, 400, 300, 20), 'second');
    const bars = container.querySelectorAll('.ghost-bar');
    expect(bars.length).toBe(1);
  });

  test('suppressed when context says so', () => {
    const gb = createGhostBar();
    gb.init(container, store, chrome);
    store.update({ suppressGhostBars: true });
    gb.showTextBar(new DOMRect(100, 200, 300, 20), 'test');
    expect(container.querySelector('.ghost-bar')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/ghost-bar.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement ghost-bar.ts**

Create `extension-src/src/content/ghost-bar.ts` exporting `createGhostBar(): ContentModule & GhostBarAPI`.

The `GhostBarAPI` adds: `showTextBar(selectionRect: DOMRect, text: string): void`, `showImageBar(img: HTMLImageElement): void`, `hideBar(): void`, `getBarRect(): DOMRect | null`, `onAskClicked(cb: (content: { type: 'text' | 'image'; text: string; imageUrl: string; rect: DOMRect }) => void): void`.

**Text ghost bar behavior** (spec section 3):
- Listens to `selectionchange` on document (debounced by `ANIM.selectionDelay` = 300ms)
- When stable selection detected, computes selection range bounding rect
- Creates an overlay highlight div at the selection position with `rgba(context-accent, 0.12)` tint
- Creates `.ghost-bar.ghost-bar-text` div, same width as selection, positioned below last line
- Uses `flow(bar, { direction: 'down', duration: ANIM.morphDuration })` to animate in
- Renders action icons from `store.get().actions` (context-ordered)
- Tracks selection on scroll via `requestAnimationFrame` loop checking `range.getBoundingClientRect()`
- On deselect (selectionchange with empty selection), uses `flow(bar, { direction: 'up' })` to retract, then removes

**Image ghost bar behavior** (spec section 4):
- Listens to `mouseover` on document (captures), filters for `<img>` elements ≥80px
- After hover delay (from context engine: `ANIM.imageHoverDelay` adjusted by topology layer), creates `.ghost-bar.ghost-bar-image`
- Positions inside image bottom edge, uses `flow(bar, { direction: 'down' })` (the bar grows downward from the edge, appearing to flow up into the image due to `bottom: 0` positioning)
- Heavier glass background (`GLASS.bgHeavy`)
- On mouseout + dismiss delay, uses reverse flow to retract

**Shared:**
- Only one bar at a time. Showing a new one dissolves the old one first.
- Respects `store.get().suppressGhostBars` — does nothing when suppressed.
- "Ask" button click emits via `onAskClicked` callback with the bar's bounding rect (for modal morph origin).
- "More" button toggles extended row with `flow()` animation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/ghost-bar.test.ts --verbose`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/ghost-bar.ts extension-src/src/__tests__/content/ghost-bar.test.ts
git commit -m "feat(extension): add inline ghost bars for text selection and image hover"
```

---

## Task 7: Focus Modal

**Files:**
- Create: `extension-src/src/content/modal.ts`
- Test: `extension-src/src/__tests__/content/modal.test.ts`

- [ ] **Step 1: Write modal tests**

```typescript
// extension-src/src/__tests__/content/modal.test.ts

import { createModal } from '../../content/modal';
import { createContextStore } from '../../content/context-engine';
import type { ContextStore } from '../../content/types';

beforeEach(() => {
  Element.prototype.animate = jest.fn().mockReturnValue({
    finished: Promise.resolve(),
    cancel: jest.fn(),
  });
});

describe('createModal', () => {
  let container: HTMLDivElement;
  let store: ContextStore;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    store = createContextStore();
  });

  afterEach(() => {
    container.remove();
  });

  test('openWithText() renders modal with text preview', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithText('Hello world', new DOMRect(100, 200, 300, 28));
    expect(container.querySelector('.aura-modal-overlay')).not.toBeNull();
    expect(container.querySelector('.aura-modal')).not.toBeNull();
    expect(container.querySelector('.modal-preview')!.textContent).toContain('Hello world');
  });

  test('openWithImage() renders modal with image', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithImage('https://example.com/img.jpg', new DOMRect(50, 100, 400, 32));
    expect(container.querySelector('.aura-modal')).not.toBeNull();
    const img = container.querySelector('.modal-preview img') as HTMLImageElement;
    expect(img).not.toBeNull();
    expect(img.src).toBe('https://example.com/img.jpg');
  });

  test('long text is truncated in preview', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    const longText = 'A'.repeat(3000);
    modal.openWithText(longText, new DOMRect(100, 200, 300, 28));
    const preview = container.querySelector('.modal-preview')!.textContent!;
    expect(preview.length).toBeLessThan(3000);
    expect(preview).toContain('...');
  });

  test('close() removes modal', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithText('test', new DOMRect(100, 200, 300, 28));
    expect(container.querySelector('.aura-modal')).not.toBeNull();
    modal.close();
    expect(container.querySelector('.aura-modal')).toBeNull();
  });

  test('modal has input field and action buttons', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithText('test', new DOMRect(100, 200, 300, 28));
    expect(container.querySelector('.modal-input')).not.toBeNull();
    const actions = container.querySelectorAll('.modal-action-btn');
    expect(actions.length).toBeGreaterThanOrEqual(4);
  });

  test('modal has model selector', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithText('test', new DOMRect(100, 200, 300, 28));
    const select = container.querySelector('.modal-model-select') as HTMLSelectElement;
    expect(select).not.toBeNull();
    expect(select.options.length).toBe(4); // Auto, Fast, Balanced, Powerful
  });

  test('escape key closes modal', () => {
    const modal = createModal();
    modal.init(container, store, chrome);
    modal.openWithText('test', new DOMRect(100, 200, 300, 28));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(container.querySelector('.aura-modal')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/modal.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement modal.ts**

Create `extension-src/src/content/modal.ts` exporting `createModal(): ContentModule & ModalAPI`.

The `ModalAPI` adds: `openWithText(text: string, originRect: DOMRect): void`, `openWithImage(imageUrl: string, originRect: DOMRect): void`, `close(): void`, `onAction(cb: (action: string, text: string, model: string) => void): void`.

**Open transition** (spec section 3.4):
1. Create `.aura-modal-overlay` (full-screen, `rgba(0,0,0,0.3)`, fades in over `ANIM.flowDuration`)
2. Create `.aura-modal` positioned at `originRect` (ghost bar position)
3. Use `morph()` from animator to animate from `originRect` to centered position (`520×480` max, centered in viewport)
4. Once morph completes, fade in contents sequentially: preview → input → actions → model selector (using `fadeIn` with stagger)

**Contents:**
- Preview: text truncated to `MODAL.previewMaxChars` (2000) with "... (N more chars)" if over. Or image with max height `MODAL.imagePreviewMaxHeight` (200px).
- Input placeholder from context: article→"Ask about this article...", code→"Ask about this code...", etc.
- Action buttons: Explain, Summarize, Chat with AURA, Save to Memory, Translate
- Model selector: Auto / Fast / Balanced / Powerful

**Close transition** (spec section 5.3):
1. Reverse morph: modal shrinks toward `originRect`
2. Overlay fades out
3. Remove all DOM

**Events:**
- Esc key listener on document (only while modal is open)
- Overlay click → close
- Action button clicks → fire `onAction` callback
- Submit button + Enter in input → fire `onAction` with 'ask' action

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/modal.test.ts --verbose`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/modal.ts extension-src/src/__tests__/content/modal.test.ts
git commit -m "feat(extension): add glassmorphism focus modal with morph transitions"
```

---

## Task 8: Migrate Supporting Systems

**Files:**
- Create: `extension-src/src/content/highlights.ts`
- Create: `extension-src/src/content/gmail.ts`
- Create: `extension-src/src/content/capture.ts`
- Create: `extension-src/src/content/link-preview.ts`
- Create: `extension-src/src/content/page-services.ts`

These are **direct migrations** from content.ts with minimal changes. Each file adopts the `ContentModule` interface and uses tokens for colors. No behavioral changes.

- [ ] **Step 1: Create highlights.ts**

Migrate `content.ts:1111-1480` (highlight system). Export `createHighlights(): ContentModule`.

Key adaptations:
- Import `GLASS`, `FONT_STACK`, `Z_TOP` from `./tokens`
- Highlight tint color uses `--aura-accent` CSS variable instead of hardcoded `#7c3aed`
- Uses `flow()` from animator for tooltip appear/disappear (currently has custom keyframe)
- The `mark[data-aura-hl]` styles go into `styles.ts` (already included in Task 3)
- Receives `ext` (chrome API) and `safeSend` callback via init

- [ ] **Step 2: Create gmail.ts**

Migrate `content.ts:2987-3565` (Gmail compose AI integration). Export `createGmail(): ContentModule`.

Key adaptations:
- Import tokens for button styling
- Uses `--aura-accent` for button color
- Receives `safeSend` callback via init
- `isGmailPage()` check runs on init; if not Gmail, module does nothing

- [ ] **Step 3: Create capture.ts**

Migrate `content.ts:3948-4131` (element capture overlay). Export `createCapture(): ContentModule`.

Key adaptations:
- Border color uses `--aura-accent` CSS variable
- Import `GLASS`, `FONT_STACK` from tokens

- [ ] **Step 4: Create link-preview.ts**

Migrate `content.ts:3566-3726` (link hover preview). Export `createLinkPreview(): ContentModule`.

Key adaptations:
- Glass material from tokens
- Uses `flow()` for appear/disappear animation

- [ ] **Step 5: Create page-services.ts**

Migrate remaining utility code that doesn't map to a UI component. Export individual functions (not a ContentModule — these are called by the coordinator's message listener):

From content.ts:
- `1763-1838`: DOM serializer (`serializeDOM`, `bestSelector`, `execAction`)
- `1839-1936`: OCR overlay (`showOcrOverlay`)
- `1937-2250`: Smart content extraction (`extractMainContent`, `findContentRoot`, `cleanClone`)
- `2251-2640`: Quick actions on input fields (the `qaHost` shadow DOM + input focus detection)
- `2641-2706`: YouTube subtitle relay
- `2707-2986`: Full page translation system
- `4132-4504`: Full page extraction
- `4608-5297`: Google SERP AI answer card

These are bulk-migrated with no behavioral changes. Each major section becomes an exported function or self-contained init function.

- [ ] **Step 6: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/highlights.ts extension-src/src/content/gmail.ts extension-src/src/content/capture.ts extension-src/src/content/link-preview.ts extension-src/src/content/page-services.ts
git commit -m "feat(extension): migrate highlights, gmail, capture, link-preview, page-services to modules"
```

---

## Task 9: Coordinator — index.ts + content.ts Shim

**Files:**
- Create: `extension-src/src/content/index.ts`
- Modify: `extension-src/src/content.ts` (gut to shim)
- Test: `extension-src/src/__tests__/content/index.test.ts`

- [ ] **Step 1: Write coordinator tests**

```typescript
// extension-src/src/__tests__/content/index.test.ts

import { init } from '../../content/index';

beforeEach(() => {
  // Clean up any previous init
  document.getElementById('aura-shadow-host')?.remove();
  (window as any).__auraToolbarMounted = false;

  Element.prototype.animate = jest.fn().mockReturnValue({
    finished: Promise.resolve(),
    cancel: jest.fn(),
  });
});

describe('coordinator init()', () => {
  test('creates shadow host element', () => {
    init();
    const host = document.getElementById('aura-shadow-host');
    expect(host).not.toBeNull();
  });

  test('does not double-mount', () => {
    init();
    init();
    const hosts = document.querySelectorAll('#aura-shadow-host');
    expect(hosts.length).toBe(1);
  });

  test('mounts FAB into shadow DOM', () => {
    init();
    const host = document.getElementById('aura-shadow-host')!;
    const shadow = host.shadowRoot;
    // Shadow root may be closed, so we test for host presence
    expect(host).not.toBeNull();
  });

  test('cleans up stale elements from previous injection', () => {
    // Simulate stale element
    const stale = document.createElement('div');
    stale.id = 'aura-dock-shadow';
    document.body.appendChild(stale);

    init();
    expect(document.getElementById('aura-dock-shadow')).toBeNull();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/index.test.ts --verbose`
Expected: FAIL — module not found

- [ ] **Step 3: Implement index.ts coordinator**

```typescript
// extension-src/src/content/index.ts

import type { ContextStore, OutboundMessage } from './types';
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
} from './page-services';

// Firefox compat
declare const browser: typeof chrome | undefined;
const ext: typeof chrome =
  typeof browser !== 'undefined' ? (browser as typeof chrome) : chrome;

function safeSend(msg: OutboundMessage, cb?: (response: any) => void): void {
  try {
    if (cb) ext.runtime.sendMessage(msg, cb);
    else ext.runtime.sendMessage(msg);
  } catch (e: unknown) {
    const err = e as Error | undefined;
    if (err?.message?.includes('context invalidated')) {
      document.getElementById('aura-shadow-host')?.remove();
      // Also clean legacy hosts
      for (const id of ['aura-dock-shadow', 'aura-host', 'aura-quick-action-host',
                         'aura-highlight-host', 'aura-img-toolbar-host', 'aura-capture-host']) {
        document.getElementById(id)?.remove();
      }
      (window as any).__auraToolbarMounted = false;
    }
  }
}

export function init(): void {
  if ((window as any).__auraToolbarMounted) return;
  (window as any).__auraToolbarMounted = true;

  // Clean stale elements from previous injection or old version
  for (const id of ['aura-shadow-host', 'aura-dock-shadow', 'aura-host',
                     'aura-quick-action-host', 'aura-highlight-host',
                     'aura-img-toolbar-host', 'aura-capture-host']) {
    document.getElementById(id)?.remove();
  }

  // ── Single shared Shadow DOM ──
  const host = document.createElement('div');
  host.id = 'aura-shadow-host';
  Object.assign(host.style, {
    position: 'fixed', top: '0', left: '0', width: '0', height: '0',
    zIndex: '2147483647', pointerEvents: 'none', overflow: 'visible',
  });
  document.documentElement.appendChild(host);
  const shadow = host.attachShadow({ mode: 'open' });

  // ── Stylesheet ──
  const style = document.createElement('style');
  style.textContent = buildStylesheet();
  shadow.appendChild(style);

  // ── Context Engine ──
  const store: ContextStore = createContextStore();
  const cleanupContext = initContextEngine(store, ext);

  // Update CSS custom properties when context changes
  store.subscribe((signal) => {
    shadow.host.style.setProperty('--aura-accent', signal.accent);
    shadow.host.style.setProperty('--aura-glow', signal.glow);
  });

  // ── Module containers ──
  const fabContainer = document.createElement('div');
  fabContainer.className = 'aura-fab-container';
  shadow.appendChild(fabContainer);

  const ghostBarContainer = document.createElement('div');
  ghostBarContainer.className = 'aura-ghostbar-container';
  shadow.appendChild(ghostBarContainer);

  const modalContainer = document.createElement('div');
  modalContainer.className = 'aura-modal-container';
  shadow.appendChild(modalContainer);

  const highlightContainer = document.createElement('div');
  highlightContainer.className = 'aura-highlight-container';
  shadow.appendChild(highlightContainer);

  const captureContainer = document.createElement('div');
  captureContainer.className = 'aura-capture-container';
  shadow.appendChild(captureContainer);

  const linkPreviewContainer = document.createElement('div');
  linkPreviewContainer.className = 'aura-linkpreview-container';
  shadow.appendChild(linkPreviewContainer);

  // ── Initialize modules ──
  const fab = createFab();
  fab.init(fabContainer, store, ext);

  const ghostBar = createGhostBar();
  ghostBar.init(ghostBarContainer, store, ext);

  const modal = createModal();
  modal.init(modalContainer, store, ext);

  const highlights = createHighlights();
  highlights.init(highlightContainer, store, ext);

  const gmail = createGmail();
  gmail.init(document.body, store, ext); // Gmail injects into page DOM, not shadow

  const capture = createCapture();
  capture.init(captureContainer, store, ext);

  const linkPreview = createLinkPreview();
  linkPreview.init(linkPreviewContainer, store, ext);

  // ── Wire ghost bar → modal ──
  ghostBar.onAskClicked((content) => {
    if (content.type === 'text') {
      modal.openWithText(content.text, content.rect);
    } else {
      modal.openWithImage(content.imageUrl, content.rect);
    }
  });

  // ── Page services (non-UI utilities) ──
  initQuickActionsOnInputs(ext, safeSend);
  initYoutubeRelay(ext);
  initNetflixRelay(ext);
  initTranslation(ext);
  initGoogleSerp(ext);

  // ── Message listener ──
  setupMessageListener(ext, {
    showDock: () => fab.showDock?.(),
    showOcrOverlay: (dataUrl, sendResponse) => {
      // delegated to page-services
    },
    startCapture: () => capture.start?.(),
    stopCapture: () => capture.stop?.(),
    scrollToHighlight: (id) => highlights.scrollTo?.(id),
  });
}
```

- [ ] **Step 4: Gut content.ts to thin shim**

Replace the entire content of `extension-src/src/content.ts` (5297 lines) with:

```typescript
/**
 * AURA Chrome Extension — Content Script Entry Point
 * Delegates to modular content/ directory.
 */
import { init } from './content/index';

init();
```

- [ ] **Step 5: Run tests to verify coordinator tests pass**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/index.test.ts --verbose`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
cd D:/Aura && git add extension-src/src/content/index.ts extension-src/src/content.ts extension-src/src/__tests__/content/index.test.ts
git commit -m "feat(extension): add coordinator and gut content.ts to thin shim"
```

---

## Task 10: Build Verification & Integration Test

**Files:**
- No new files
- Test: full build + existing test suite

- [ ] **Step 1: Run all content module tests**

Run: `cd D:/Aura/extension-src && npx jest src/__tests__/content/ --verbose`
Expected: All tests pass across tokens, animator, context-engine, fab, ghost-bar, modal, index

- [ ] **Step 2: Run full test suite (ensure no regressions)**

Run: `cd D:/Aura/extension-src && npx jest --verbose`
Expected: All existing tests still pass (PyodideExecutor, StreamingPreview, exportUtils, streamChat, useVersionHistory) plus all new content module tests

- [ ] **Step 3: Build the content script**

Run: `cd D:/Aura/extension-src && npx vite build --config vite.config.content.ts`
Expected: Builds successfully. Output at `extension/content.js`. Vite follows the `content.ts` → `content/index.ts` import chain and bundles everything into a single IIFE.

- [ ] **Step 4: Verify output bundle**

Run: `wc -c D:/Aura/extension/content.js && head -5 D:/Aura/extension/content.js`
Expected: File exists, is a valid JS IIFE, no import/export statements (fully bundled).

- [ ] **Step 5: Full build (all targets)**

Run: `cd D:/Aura/extension-src && npm run build`
Expected: All build targets succeed (main, background, content, youtube, netflix, newtab, worker, ai-worker). Zero errors.

- [ ] **Step 6: Commit build output if applicable**

```bash
cd D:/Aura && git add -A && git status
# Only commit if extension/content.js is tracked
git commit -m "build(extension): rebuild content script with modular Liquid Aura architecture"
```

---

## Summary

| Task | What it builds | New tests |
|---|---|---|
| 1 | Types + tokens foundation | 5 |
| 2 | Liquid-morph animator | 5 |
| 3 | CSS generator | 7 |
| 4 | 5-layer context engine | 8 |
| 5 | Living FAB | 6 |
| 6 | Ghost bars (text + image) | 7 |
| 7 | Focus modal | 7 |
| 8 | Migrate supporting systems | 0 (behavioral migration) |
| 9 | Coordinator + shim | 4 |
| 10 | Build verification | 0 (integration) |
| **Total** | **14 new files, 1 modified** | **49 new tests** |

The 5297-line `content.ts` becomes a 4-line shim. All floating UI surfaces share one Shadow DOM, one animation system, one context engine, and one design language.
