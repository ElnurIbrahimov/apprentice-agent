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
