/**
 * Tests for the Living FAB (Floating Action Button).
 */
import { createFab } from '../../content/fab';
import { createContextStore } from '../../content/context-engine';
import { chromeMock } from '../setup';

describe('createFab', () => {
  let container: HTMLElement;
  let animateMock: jest.Mock;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);

    animateMock = jest.fn(() => ({ finished: Promise.resolve(), cancel: jest.fn() }));
    Element.prototype.animate = animateMock;

    // Reset chrome mocks
    (chromeMock.runtime.sendMessage as jest.Mock).mockClear();
  });

  afterEach(() => {
    document.body.removeChild(container);
    jest.restoreAllMocks();
  });

  test('init renders FAB pill into container', () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    expect(container.querySelector('.aura-fab')).not.toBeNull();
    expect(container.querySelector('.fab-pill')).not.toBeNull();
  });

  test('FAB contains logo element', () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    const logo = container.querySelector('.fab-logo');
    expect(logo).not.toBeNull();
    expect(logo!.querySelector('svg')).not.toBeNull();
  });

  test('FAB contains popout with action buttons', () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    const popout = container.querySelector('.fab-popout');
    expect(popout).not.toBeNull();

    const actionBtns = popout!.querySelectorAll('.fab-action-btn');
    expect(actionBtns.length).toBeGreaterThanOrEqual(4);
  });

  test('FAB contains close button', () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    expect(container.querySelector('.fab-close')).not.toBeNull();
  });

  test('context change updates FAB icon', async () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    const logoBefore = container.querySelector('.fab-logo')!.innerHTML;

    store.update({ type: 'code', icon: '<svg><text>CODE</text></svg>' });

    // Wait for crossFade to complete (async)
    await Promise.resolve();
    await Promise.resolve();

    const logoAfter = container.querySelector('.fab-logo')!.innerHTML;
    expect(logoAfter).not.toBe(logoBefore);
  });

  test('destroy() removes FAB from container', () => {
    const store = createContextStore();
    const fab = createFab();
    fab.init(container, store, chrome);

    expect(container.querySelector('.aura-fab')).not.toBeNull();

    fab.destroy!();

    expect(container.querySelector('.aura-fab')).toBeNull();
  });
});
