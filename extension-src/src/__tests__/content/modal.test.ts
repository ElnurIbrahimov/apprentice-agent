/**
 * Tests for the glassmorphism focus modal with morph transitions.
 */
import { createModal } from '../../content/modal';
import { createContextStore } from '../../content/context-engine';

/** Flush all pending microtasks (multiple awaits in async chains). */
async function flushPromises(ticks = 10): Promise<void> {
  for (let i = 0; i < ticks; i++) await Promise.resolve();
}

function makeRect(overrides: Partial<DOMRect> = {}): DOMRect {
  return {
    left: 100, top: 200, right: 400, bottom: 228,
    width: 300, height: 28,
    x: 100, y: 200,
    toJSON: () => ({}),
    ...overrides,
  } as DOMRect;
}

describe('createModal', () => {
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

    // Stub viewport dimensions for centering calculations
    Object.defineProperty(window, 'innerWidth', { value: 1280, writable: true, configurable: true });
    Object.defineProperty(window, 'innerHeight', { value: 800, writable: true, configurable: true });
  });

  afterEach(() => {
    container.remove();
    jest.restoreAllMocks();
    // Clean up any lingering modals
    document.querySelectorAll('.aura-modal-overlay').forEach(el => el.remove());
    document.querySelectorAll('.aura-modal').forEach(el => el.remove());
  });

  // 1. openWithText() renders modal with text preview
  test('openWithText() renders modal with text preview', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    const rect = makeRect();
    modal.openWithText('Hello world text', rect);

    // Wait for async operations
    await flushPromises();

    const overlay = document.querySelector('.aura-modal-overlay');
    const modalEl = document.querySelector('.aura-modal');
    const preview = document.querySelector('.modal-preview');

    expect(overlay).not.toBeNull();
    expect(modalEl).not.toBeNull();
    expect(preview).not.toBeNull();
    expect(preview!.textContent).toContain('Hello world text');
  });

  // 2. openWithImage() renders modal with image
  test('openWithImage() renders modal with image', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    const rect = makeRect();
    modal.openWithImage('https://example.com/photo.jpg', rect);

    await Promise.resolve();
    await Promise.resolve();

    const modalEl = document.querySelector('.aura-modal');
    const img = document.querySelector('.modal-preview img') as HTMLImageElement | null;

    expect(modalEl).not.toBeNull();
    expect(img).not.toBeNull();
    expect(img!.src).toContain('https://example.com/photo.jpg');
  });

  // 3. long text is truncated in preview
  test('long text is truncated in preview', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    const longText = 'A'.repeat(3000);
    modal.openWithText(longText, makeRect());

    await Promise.resolve();
    await Promise.resolve();

    const preview = document.querySelector('.modal-preview');
    expect(preview).not.toBeNull();
    const content = preview!.textContent ?? '';
    expect(content).toContain('...');
    expect(content.length).toBeLessThan(3000);
  });

  // 4. close() removes modal
  test('close() removes modal', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    modal.openWithText('some text', makeRect());

    await Promise.resolve();
    await Promise.resolve();

    expect(document.querySelector('.aura-modal')).not.toBeNull();

    await modal.close();

    expect(document.querySelector('.aura-modal')).toBeNull();
  });

  // 5. modal has input field and action buttons
  test('modal has input field and action buttons', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    modal.openWithText('test content', makeRect());

    await Promise.resolve();
    await Promise.resolve();

    const input = document.querySelector('.modal-input');
    const actionBtns = document.querySelectorAll('.modal-action-btn');

    expect(input).not.toBeNull();
    expect(actionBtns.length).toBeGreaterThanOrEqual(4);
  });

  // 6. modal has model selector
  test('modal has model selector', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    modal.openWithText('test content', makeRect());

    await Promise.resolve();
    await Promise.resolve();

    const select = document.querySelector('.modal-model-select') as HTMLSelectElement | null;
    expect(select).not.toBeNull();
    expect(select!.options.length).toBe(4);

    const optionValues = Array.from(select!.options).map(o => o.value);
    expect(optionValues).toContain('auto');
    expect(optionValues).toContain('fast');
    expect(optionValues).toContain('balanced');
    expect(optionValues).toContain('powerful');
  });

  // 7. escape key closes modal
  test('escape key closes modal', async () => {
    const modal = createModal();
    modal.init(container, store, ext);

    modal.openWithText('some text', makeRect());

    await Promise.resolve();
    await Promise.resolve();

    expect(document.querySelector('.aura-modal')).not.toBeNull();

    const escEvent = new KeyboardEvent('keydown', { key: 'Escape', bubbles: true });
    document.dispatchEvent(escEvent);

    // Allow async close to run
    await flushPromises();

    expect(document.querySelector('.aura-modal')).toBeNull();
  });
});
