import { init } from '../../content/index';

beforeEach(() => {
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
    expect(document.getElementById('aura-shadow-host')).not.toBeNull();
  });

  test('does not double-mount', () => {
    init();
    init();
    expect(document.querySelectorAll('#aura-shadow-host').length).toBe(1);
  });

  test('cleans up stale elements from previous injection', () => {
    const stale = document.createElement('div');
    stale.id = 'aura-dock-shadow';
    document.body.appendChild(stale);
    init();
    expect(document.getElementById('aura-dock-shadow')).toBeNull();
  });
});
