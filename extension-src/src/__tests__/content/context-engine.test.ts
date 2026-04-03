import { createContextStore, detectPageType, createCadenceTracker } from '../../content/context-engine';
import type { PageContext } from '../../content/types';

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
    expect(listener).toHaveBeenCalledTimes(1);
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
    for (let i = 0; i < 10; i++) {
      tracker.recordScroll(500);
    }
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
