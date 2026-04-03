/**
 * Tests for inline ghost bars — text selection and image hover.
 */
import { createGhostBar } from '../../content/ghost-bar';
import { createContextStore } from '../../content/context-engine';
import { CONTEXT_ACTIONS } from '../../content/tokens';

function makeRect(overrides: Partial<DOMRect> = {}): DOMRect {
  return {
    left: 100, top: 200, right: 400, bottom: 220,
    width: 300, height: 20,
    x: 100, y: 200,
    toJSON: () => ({}),
    ...overrides,
  } as DOMRect;
}

function makeImage(overrides: Partial<HTMLImageElement> = {}): HTMLImageElement {
  const img = document.createElement('img') as HTMLImageElement;
  img.src = 'https://example.com/photo.jpg';
  // Stub getBoundingClientRect
  const rect = makeRect({ left: 50, top: 100, right: 350, bottom: 400, width: 300, height: 300 });
  jest.spyOn(img, 'getBoundingClientRect').mockReturnValue(rect);
  Object.defineProperty(img, 'naturalWidth', { value: 300, configurable: true });
  Object.defineProperty(img, 'naturalHeight', { value: 300, configurable: true });
  Object.defineProperty(img, 'offsetWidth', { value: 300, configurable: true });
  Object.defineProperty(img, 'offsetHeight', { value: 300, configurable: true });
  Object.assign(img, overrides);
  return img;
}

describe('createGhostBar', () => {
  let container: HTMLElement;
  let store: ReturnType<typeof createContextStore>;
  let ext: typeof chrome;
  let animateMock: jest.Mock;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    store = createContextStore();
    ext = { runtime: { sendMessage: jest.fn() } } as unknown as typeof chrome;

    animateMock = jest.fn(() => ({ finished: Promise.resolve(), cancel: jest.fn() }));
    Element.prototype.animate = animateMock;
  });

  afterEach(() => {
    container.remove();
    jest.restoreAllMocks();
    // Clean up any lingering ghost bars in body
    document.querySelectorAll('.ghost-bar').forEach(el => el.remove());
  });

  // 1. init attaches selection listener
  test('init attaches selection listener', () => {
    const addEventSpy = jest.spyOn(document, 'addEventListener');
    const gb = createGhostBar();
    gb.init(container, store, ext);

    const calls = addEventSpy.mock.calls.map(c => c[0]);
    expect(calls).toContain('selectionchange');
  });

  // 2. showTextBar() renders ghost bar into container
  test('showTextBar() renders ghost bar into container', () => {
    const gb = createGhostBar();
    gb.init(container, store, ext);

    const rect = makeRect();
    gb.showTextBar(rect, 'Hello world');

    expect(container.querySelector('.ghost-bar')).not.toBeNull();
    expect(container.querySelector('.ghost-bar-text')).not.toBeNull();
  });

  // 3. text ghost bar has correct action buttons from context
  test('text ghost bar has correct action buttons from context', () => {
    store.update({
      type: 'code',
      actions: CONTEXT_ACTIONS.code,
    });

    const gb = createGhostBar();
    gb.init(container, store, ext);

    gb.showTextBar(makeRect(), 'const x = 1');

    const buttons = container.querySelectorAll('.gb-action');
    expect(buttons.length).toBeGreaterThanOrEqual(5);
  });

  // 4. showImageBar() renders ghost bar inside image bounds
  test('showImageBar() renders ghost bar inside image bounds', () => {
    const gb = createGhostBar();
    gb.init(container, store, ext);

    const img = makeImage();
    gb.showImageBar(img);

    expect(container.querySelector('.ghost-bar')).not.toBeNull();
    expect(container.querySelector('.ghost-bar-image')).not.toBeNull();
  });

  // 5. hideBar() removes ghost bar
  test('hideBar() removes ghost bar', async () => {
    const gb = createGhostBar();
    gb.init(container, store, ext);

    gb.showTextBar(makeRect(), 'test');
    expect(container.querySelector('.ghost-bar')).not.toBeNull();

    await gb.hideBar();

    expect(container.querySelector('.ghost-bar')).toBeNull();
    expect(gb.getBarRect()).toBeNull();
  });

  // 6. only one ghost bar at a time
  test('only one ghost bar at a time', () => {
    const gb = createGhostBar();
    gb.init(container, store, ext);

    gb.showTextBar(makeRect(), 'first selection');
    gb.showTextBar(makeRect({ left: 50 }), 'second selection');

    const bars = container.querySelectorAll('.ghost-bar');
    expect(bars.length).toBe(1);
  });

  // 7. suppressed when context says so
  test('suppressed when context says so', () => {
    store.update({ suppressGhostBars: true });

    const gb = createGhostBar();
    gb.init(container, store, ext);

    gb.showTextBar(makeRect(), 'should not appear');

    expect(container.querySelector('.ghost-bar')).toBeNull();
  });
});
