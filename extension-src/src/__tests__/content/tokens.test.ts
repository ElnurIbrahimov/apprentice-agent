import {
  PALETTES,
  CONTEXT_ACTIONS,
  ANIM,
  GLASS,
  GHOST_BAR,
  MODAL,
  FAB,
  FONT_STACK,
  Z_TOP,
  Z_MID,
} from '../../content/tokens';
import type { PageContext } from '../../content/types';

const CONTEXTS: PageContext[] = ['general', 'article', 'media', 'code', 'email', 'shopping'];

// ── Palettes ──────────────────────────────────────────────────────────────────

describe('PALETTES', () => {
  test.each(CONTEXTS)('%s has a valid accent hex', (ctx) => {
    expect(PALETTES[ctx].accent).toMatch(/^#[0-9a-fA-F]{6}$/);
  });

  test.each(CONTEXTS)('%s has a valid glow rgba', (ctx) => {
    expect(PALETTES[ctx].glow).toMatch(/^rgba\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*[\d.]+\s*\)$/);
  });

  test.each(CONTEXTS)('%s has an SVG icon', (ctx) => {
    const icon = PALETTES[ctx].icon;
    expect(icon).toContain('<svg');
    expect(icon).toContain('</svg>');
  });

  test('all six contexts are covered', () => {
    expect(Object.keys(PALETTES).sort()).toEqual([...CONTEXTS].sort());
  });
});

// ── Context actions ───────────────────────────────────────────────────────────

describe('CONTEXT_ACTIONS', () => {
  test.each(CONTEXTS)('%s actions start with "ask"', (ctx) => {
    expect(CONTEXT_ACTIONS[ctx][0]).toBe('ask');
  });

  test.each(CONTEXTS)('%s has at least 5 actions', (ctx) => {
    expect(CONTEXT_ACTIONS[ctx].length).toBeGreaterThanOrEqual(5);
  });

  test('all six contexts are covered', () => {
    expect(Object.keys(CONTEXT_ACTIONS).sort()).toEqual([...CONTEXTS].sort());
  });
});

// ── Animation tokens ──────────────────────────────────────────────────────────

describe('ANIM', () => {
  const numberKeys: (keyof typeof ANIM)[] = [
    'morphDuration', 'flowDuration', 'glowPulse', 'sequentialStagger',
    'dismissDelay', 'crossFadeDuration', 'selectionDelay',
    'imageHoverDelay',
  ];

  test.each(numberKeys)('%s is a positive number', (key) => {
    expect(typeof ANIM[key]).toBe('number');
    expect(ANIM[key]).toBeGreaterThan(0);
  });

  test('flowDuration > morphDuration', () => {
    expect(ANIM.flowDuration).toBeGreaterThan(ANIM.morphDuration);
  });

  test('morphEasing is a non-empty string', () => {
    expect(typeof ANIM.morphEasing).toBe('string');
    expect(ANIM.morphEasing.length).toBeGreaterThan(0);
  });
});

// ── Glass tokens ──────────────────────────────────────────────────────────────

describe('GLASS', () => {
  test('bg is a rgba string', () => {
    expect(GLASS.bg).toMatch(/^rgba\(/);
  });

  test('bgHeavy is a rgba string', () => {
    expect(GLASS.bgHeavy).toMatch(/^rgba\(/);
  });

  test('backdrop contains blur', () => {
    expect(GLASS.backdrop).toContain('blur(');
  });

  test('borderOpacity is between 0 and 1', () => {
    expect(GLASS.borderOpacity).toBeGreaterThan(0);
    expect(GLASS.borderOpacity).toBeLessThan(1);
  });

  test('shadowBase is defined', () => {
    expect(GLASS.shadowBase.length).toBeGreaterThan(0);
  });
});

// ── Font stack ────────────────────────────────────────────────────────────────

describe('FONT_STACK', () => {
  test('is non-empty', () => {
    expect(FONT_STACK.length).toBeGreaterThan(0);
  });

  test('contains system-ui', () => {
    expect(FONT_STACK).toContain('system-ui');
  });
});

// ── Z-index ───────────────────────────────────────────────────────────────────

describe('Z-index constants', () => {
  test('Z_TOP is max safe integer for CSS', () => {
    expect(Z_TOP).toBe(2147483647);
  });

  test('Z_MID < Z_TOP', () => {
    expect(Z_MID).toBeLessThan(Z_TOP);
  });
});

// ── Other token sanity checks ─────────────────────────────────────────────────

describe('GHOST_BAR', () => {
  test('height and iconSize are positive', () => {
    expect(GHOST_BAR.height).toBeGreaterThan(0);
    expect(GHOST_BAR.iconSize).toBeGreaterThan(0);
  });
});

describe('MODAL', () => {
  test('maxWidth and maxHeight are positive', () => {
    expect(MODAL.maxWidth).toBeGreaterThan(0);
    expect(MODAL.maxHeight).toBeGreaterThan(0);
  });
});

describe('FAB', () => {
  test('glow intensity range is valid', () => {
    expect(FAB.glowIntensityMin).toBeGreaterThan(0);
    expect(FAB.glowIntensityMax).toBeGreaterThan(FAB.glowIntensityMin);
    expect(FAB.glowIntensityMax).toBeLessThanOrEqual(1);
  });

  test('logoSize is positive', () => {
    expect(FAB.logoSize).toBeGreaterThan(0);
  });
});
