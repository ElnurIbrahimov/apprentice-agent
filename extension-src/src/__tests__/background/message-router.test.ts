/**
 * Message router tests for background.ts
 *
 * Strategy: mock `chrome` and `fetch` globally, import background.ts
 * via jest.isolateModules to trigger listener registration, then simulate
 * messages through the onMessage listener.
 */

// ── Polyfill Response for jsdom ─────────────────────────────────────────────
if (typeof (globalThis as any).Response === 'undefined') {
  (globalThis as any).Response = class Response {
    status: number;
    ok: boolean;
    headers: Map<string, string>;
    private _body: string;
    constructor(body: string, init?: ResponseInit) {
      this._body = body;
      this.status = init?.status ?? 200;
      this.ok = this.status >= 200 && this.status < 300;
      this.headers = new Map(Object.entries(init?.headers ?? {}));
    }
    async text() { return this._body; }
    async json() { return JSON.parse(this._body); }
  };
}

// ── Mock `fetch` before any imports ────────────────────────────────────────
const mockFetchImpl = jest.fn(() =>
  Promise.resolve(new (globalThis as any).Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  }))
);
(globalThis as any).fetch = mockFetchImpl;

import { DEFAULT_BACKEND_URL } from '../../defaults';

// ── In-memory storage mock ──────────────────────────────────────────────────
const storage: Record<string, any> = {};
const storageListeners: Array<(changes: Record<string, any>, area: string) => void> = [];

function makeChromeMock() {
  const onMessageListeners: Array<(msg: any, sender: any, sendResponse: (r?: any) => void) => boolean | undefined> = [];

  const chromeMock = {
    storage: {
      local: {
        get: jest.fn((keys: string | string[], cb: (d: Record<string, any>) => void) => {
          const ks = Array.isArray(keys) ? keys : [keys];
          const result: Record<string, any> = {};
          for (const k of ks) {
            if (k in storage) result[k] = storage[k];
          }
          cb(result);
        }),
        set: jest.fn((data: Record<string, any>, cb?: () => void) => {
          const prev: Record<string, any> = {};
          for (const k of Object.keys(data)) {
            prev[k] = undefined;
          }
          Object.assign(storage, data);
          const changes: Record<string, any> = {};
          for (const k of Object.keys(data)) {
            changes[k] = { oldValue: prev[k], newValue: data[k] };
          }
          for (const listener of storageListeners) {
            listener(changes, 'local');
          }
          cb?.();
        }),
        remove: jest.fn((keys: string | string[], cb?: () => void) => {
          const ks = Array.isArray(keys) ? keys : [keys];
          for (const k of ks) delete storage[k];
          cb?.();
        }),
      },
      onChanged: {
        addListener: jest.fn((fn: (changes: Record<string, any>, area: string) => void) => {
          storageListeners.push(fn);
        }),
      },
    },
    runtime: {
      id: 'test-extension-id',
      getURL: jest.fn((path: string) => `chrome-extension://test-id/${path}`),
      sendMessage: jest.fn(() => Promise.resolve()),
      onMessage: {
        addListener: jest.fn((fn: any) => { onMessageListeners.push(fn); }),
        removeListener: jest.fn(),
      },
      onInstalled: { addListener: jest.fn() },
      onStartup: { addListener: jest.fn() },
      lastError: undefined as any,
    },
    alarms: {
      create: jest.fn(),
      onAlarm: { addListener: jest.fn() },
    },
    tabs: {
      query: jest.fn((q: any, cb: (tabs: any[]) => void) => { cb([]); }),
      sendMessage: jest.fn((tabId: number, msg: any, cb?: (r: any) => void) => {
        cb?.(undefined);
      }),
      captureVisibleTab: jest.fn((wndId: any, opts: any, cb: (dataUrl: string) => void) => { cb(''); }),
      update: jest.fn((tabId: number, props: any, cb?: () => void) => { cb?.(); }),
      onUpdated: { addListener: jest.fn() },
      onRemoved: { addListener: jest.fn() },
    },
    scripting: {
      executeScript: jest.fn(() => Promise.resolve([{ result: null }])),
    },
    sidePanel: {
      open: jest.fn(() => Promise.resolve()),
      setPanelBehavior: jest.fn(),
    },
    contextMenus: {
      removeAll: jest.fn((cb?: () => void) => { cb?.(); }),
      create: jest.fn(),
      remove: jest.fn(),
      onClicked: { addListener: jest.fn() },
    },
    notifications: {
      create: jest.fn(),
      clear: jest.fn(),
      onClicked: { addListener: jest.fn() },
      onButtonClicked: { addListener: jest.fn() },
    },
    debugger: {
      detach: jest.fn(),
      onDetach: { addListener: jest.fn() },
    },
    offscreen: {
      hasDocument: jest.fn(() => Promise.resolve(true)),
      createDocument: jest.fn(() => Promise.resolve()),
    },
    action: {
      setBadgeText: jest.fn(),
      setBadgeBackgroundColor: jest.fn(),
    },
    downloads: {
      download: jest.fn(),
    },
    omnibox: {
      setDefaultSuggestion: jest.fn(),
      onInputChanged: { addListener: jest.fn() },
      onInputEntered: { addListener: jest.fn() },
    },
    commands: {
      onCommand: { addListener: jest.fn() },
    },
  };

  return { chromeMock, onMessageListeners };
}

// ── Test harness ─────────────────────────────────────────────────────────

let harness: ReturnType<typeof makeChromeMock>;

function resetStorage() {
  for (const k of Object.keys(storage)) delete storage[k];
  storage.backendUrl = DEFAULT_BACKEND_URL;
  storage.apiKey = 'test-key';
}

beforeEach(() => {
  resetStorage();
  mockFetchImpl.mockReset();
  mockFetchImpl.mockImplementation(() =>
    Promise.resolve(new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    }))
  );
  harness = makeChromeMock();
  (globalThis as any).chrome = harness.chromeMock;
  (globalThis as any).browser = undefined;
});

/** Import background.ts and return the registered message listeners. */
function importBackground(): Array<(msg: any, sender: any, sendResponse: (r?: any) => void) => boolean | undefined> {
  // Clear any previous module cache
  jest.resetModules();
  jest.isolateModules(() => {
    require('../../../background');
  });
  // After import, onMessageListeners should be populated
  // But we also need to pick up listeners from the *fresh* require
  // Re-import to get the chrome mock wired up
  return (globalThis as any).__bgListeners || harness.onMessageListeners;
}

/** Simulate sending a message through the message router. */
function sendMessage(msg: any, sender?: any): Promise<any> {
  // Import background to register handlers if not already done
  if (harness.onMessageListeners.length === 0) {
    importBackground();
  }
  return new Promise((resolve) => {
    let resolved = false;
    const sendResponse = jest.fn((r?: any) => {
      if (!resolved) { resolved = true; resolve(r); }
    });
    const listener = harness.onMessageListeners[harness.onMessageListeners.length - 1];
    if (!listener) { resolve(undefined); return; }
    const result = listener(msg, sender ?? { id: 'test-extension-id', tab: { id: 1, windowId: 1 } }, sendResponse);
    if (result !== true) {
      // Sync handler — resolve immediately with the arg sendResponse was called with
      resolve(sendResponse.mock.calls[0]?.[0]);
    }
    // Async handler will call sendResponse later, which resolves the promise
  });
}

// ── Tests ──────────────────────────────────────────────────────────────────

describe('background message router', () => {
  // Force background module import for all tests
  beforeEach(() => {
    jest.resetModules();
  });

describe('SIDEBAR_READY', () => {
    test('drains pending query from storage', async () => {
      storage.pendingQuery = 'hello';
      storage.pendingAction = 'ask';
      storage.pendingUrl = 'https://example.com';
      storage.pendingTitle = 'Example';

      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({ type: 'SIDEBAR_READY' });

      // SIDEBAR_READY is handled by the switch-based (second) listener
      expect(harness.chromeMock.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'PREFILL_TEXT', text: 'hello' }),
      );
    });

    test('drains pending specialist from storage', async () => {
      storage.pendingQuery = 'research this';
      storage.pendingSpecialist = 'research-agent';

      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({ type: 'SIDEBAR_READY' });

      expect(harness.chromeMock.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'SPECIALIST_PREFILL', specialist: 'research-agent' }),
      );
    });
  });
  });

  describe('SAVE_KNOWLEDGE', () => {
    test('posts to backend and returns response', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(
        new Response(JSON.stringify({ saved: true }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );

      const result = await sendMessage({
        type: 'SAVE_KNOWLEDGE',
        text: 'hello',
        url: 'https://example.com',
        title: 'Test',
        tags: ['test'],
      });

      expect(mockFetchImpl).toHaveBeenCalledWith(
        expect.stringContaining('/api/knowledge/save'),
        expect.objectContaining({ method: 'POST' }),
      );
      expect(result).toEqual(expect.objectContaining({ ok: true }));
    });

    test('returns error on fetch failure', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockRejectedValueOnce(new Error('Network error'));

      const result = await sendMessage({
        type: 'SAVE_KNOWLEDGE',
        text: 'test',
        url: 'https://example.com',
        title: 'Test',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false }));
    });
  });

  describe('GET_CURRENT_TAB', () => {
    test('returns tab info for active tab', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([{ id: 1, url: 'https://example.com', title: 'Example' }]);
      });

      const result = await sendMessage({ type: 'GET_CURRENT_TAB' });
      expect(result).toEqual(expect.objectContaining({ ok: true, url: 'https://example.com' }));
    });

    test('returns ok:false when no active tab', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([]);
      });

      const result = await sendMessage({ type: 'GET_CURRENT_TAB' });
      expect(result).toEqual({ ok: false });
    });
  });

  describe('QUICK_ACTION', () => {
    test('improve action sends correct prompt', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(
        new Response(JSON.stringify({ response: 'Improved' }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );

      const result = await sendMessage({
        type: 'QUICK_ACTION',
        action: 'improve',
        text: 'hello world',
      });

      expect(mockFetchImpl).toHaveBeenCalledWith(
        expect.stringContaining('/api/chat'),
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('Improve this text'),
        }),
      );
      expect(result).toEqual(expect.objectContaining({ ok: true, result: 'Improved' }));
    });

    test('translate action substitutes language', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(
        new Response(JSON.stringify({ response: 'Hola' }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );

      await sendMessage({
        type: 'QUICK_ACTION',
        action: 'translate',
        text: 'hello',
        language: 'Spanish',
      });

      const callBody = JSON.parse(mockFetchImpl.mock.calls[0][1].body);
      expect(callBody.message).toContain('Spanish');
      expect(callBody.message).toContain('hello');
    });
  });

  describe('Highlight operations', () => {
    beforeEach(() => {
      jest.isolateModules(() => { require('../../../background'); });
    });

    test('SAVE_HIGHLIGHT stores highlight', async () => {
      const hl = {
        id: 'hl1', url: 'https://example.com', text: 'test',
        xpath: '/p', context: 'ctx', timestamp: Date.now(), color: '#ff0', pageTitle: 'Test', stale: false,
      };

      const result = await sendMessage({ type: 'SAVE_HIGHLIGHT', highlight: hl });
      expect(result).toEqual(expect.objectContaining({ ok: true }));
      expect(storage.aura_highlights['https://example.com']).toHaveLength(1);
    });

    test('SAVE_HIGHLIGHT rejects at 1000 limit', async () => {
      const highlights: any[] = Array.from({ length: 1000 }, (_, i) => ({
        id: `hl${i}`, url: 'https://example.com', text: `text${i}`,
        xpath: '', context: '', timestamp: i, color: '#ff0', pageTitle: 'Test', stale: false,
      }));
      storage.aura_highlights = { 'https://example.com': highlights };

      const result = await sendMessage({
        type: 'SAVE_HIGHLIGHT',
        highlight: { id: 'new', url: 'https://other.com', text: 'new', xpath: '', context: '', timestamp: Date.now(), color: '#ff0', pageTitle: 'Other', stale: false },
      });

      expect(result).toEqual(expect.objectContaining({ ok: false }));
    });

    test('GET_HIGHLIGHTS returns highlights for URL', async () => {
      storage.aura_highlights = {
        'https://example.com': [{
          id: 'hl1', url: 'https://example.com', text: 'test',
          xpath: '', context: '', timestamp: 1, color: '#ff0', pageTitle: 'Test', stale: false,
        }],
      };

      const result = await sendMessage({ type: 'GET_HIGHLIGHTS', url: 'https://example.com' });
      expect(result).toEqual(expect.objectContaining({ ok: true }));
      expect(result.highlights).toHaveLength(1);
    });

    test('DELETE_HIGHLIGHT removes specific highlight', async () => {
      storage.aura_highlights = {
        'https://example.com': [
          { id: 'hl1', url: 'https://example.com', text: 'a' },
          { id: 'hl2', url: 'https://example.com', text: 'b' },
        ],
      };

      await sendMessage({ type: 'DELETE_HIGHLIGHT', id: 'hl1', url: 'https://example.com' });
      expect(storage.aura_highlights['https://example.com']).toHaveLength(1);
    });

    test('CLEAR_URL_HIGHLIGHTS removes all highlights for URL', async () => {
      storage.aura_highlights = {
        'https://example.com': [{ id: '1', text: 'a' }],
        'https://other.com': [{ id: '2', text: 'b' }],
      };

      await sendMessage({ type: 'CLEAR_URL_HIGHLIGHTS', url: 'https://example.com' });
      expect(storage.aura_highlights['https://example.com']).toBeUndefined();
      expect(storage.aura_highlights['https://other.com']).toBeDefined();
    });

    test('SEARCH_HIGHLIGHTS finds matching highlights', async () => {
      storage.aura_highlights = {
        'https://example.com': [
          { id: '1', url: 'https://example.com', text: 'important concept', xpath: '', context: '', timestamp: 1, color: '#ff0', pageTitle: 'Research Paper', stale: false },
          { id: '2', url: 'https://example.com', text: 'other stuff', xpath: '', context: '', timestamp: 2, color: '#ff0', pageTitle: 'Something Else', stale: false },
        ],
      };

      const result = await sendMessage({ type: 'SEARCH_HIGHLIGHTS', query: 'research' });
      expect(result.ok).toBe(true);
      expect(result.highlights.length).toBeGreaterThanOrEqual(1);
    });

    test('GET_ALL_HIGHLIGHTS returns entire store', async () => {
      storage.aura_highlights = {
        'https://a.com': [{ id: '1' }],
        'https://b.com': [{ id: '2' }],
      };

      const result = await sendMessage({ type: 'GET_ALL_HIGHLIGHTS' });
      expect(result.ok).toBe(true);
      expect(result.store).toBeDefined();
    });
  });

  describe('TRANSLATE_BATCH', () => {
    test('sends batched translation and parses numbered results', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(
        new Response(JSON.stringify({ response: '[1] Bonjour\n\n[2] Au revoir' }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );

      const result = await sendMessage({
        type: 'TRANSLATE_BATCH',
        texts: ['Hello', 'Goodbye'],
        targetLang: 'French',
      });

      expect(result).toEqual(expect.objectContaining({ ok: true }));
      expect(result.translations).toHaveLength(2);
      expect(result.translations[0]).toContain('Bonjour');
    });
  });

  describe('GET_PAGE_CONTENT', () => {
    test('blocks protected chrome:// URLs', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([{ id: 1, url: 'chrome://settings', title: 'Settings' }]);
      });

      const result = await sendMessage({ type: 'GET_PAGE_CONTENT' });
      expect(result).toEqual(expect.objectContaining({ ok: false, error: expect.stringContaining('Protected') }));
    });

    test('blocks about: URLs', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([{ id: 1, url: 'about:blank', title: '' }]);
      });

      const result = await sendMessage({ type: 'GET_PAGE_CONTENT' });
      expect(result).toEqual(expect.objectContaining({ ok: false }));
    });
  });

  describe('LINK_PREVIEW', () => {
    test('fetches and parses link metadata', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const html = '<html><head><title>Test Page</title><meta name="description" content="A description"></head><body></body></html>';
      mockFetchImpl.mockResolvedValueOnce(
        new Response(html, { status: 200, headers: { 'Content-Type': 'text/html' } }),
      );

      const result = await sendMessage({
        type: 'LINK_PREVIEW',
        url: 'https://example.com/page',
      });

      expect(result).toEqual(expect.objectContaining({
        ok: true,
        title: 'Test Page',
        description: 'A description',
        domain: 'example.com',
      }));
    });

    test('blocks private/internal IPs', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'LINK_PREVIEW',
        url: 'http://192.168.1.1/admin',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false }));
    });

    test('blocks unsafe URL schemes', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'LINK_PREVIEW',
        url: 'chrome://settings',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false }));
    });
  });

  describe('START_CAPTURE_MODE / STOP_CAPTURE_MODE', () => {
    test('relays start to active tab', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([{ id: 42, url: 'https://example.com', title: 'Test' }]);
      });

      await sendMessage({ type: 'START_CAPTURE_MODE' });
      expect(harness.chromeMock.tabs.sendMessage).toHaveBeenCalledWith(42, { type: 'START_CAPTURE_MODE' });
    });

    test('relays stop to active tab', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      harness.chromeMock.tabs.query.mockImplementationOnce((q: any, cb: (t: any[]) => void) => {
        cb([{ id: 42, url: 'https://example.com', title: 'Test' }]);
      });

      await sendMessage({ type: 'STOP_CAPTURE_MODE' });
      expect(harness.chromeMock.tabs.sendMessage).toHaveBeenCalledWith(42, { type: 'STOP_CAPTURE_MODE' });
    });
  });

  describe('YT_SUBTITLES / NETFLIX_SUBTITLES relay', () => {
    test('relays YouTube subtitles', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({
        type: 'YT_SUBTITLES',
        videoId: 'abc123',
        lang: 'en',
        segments: [{ start: 0, dur: 5, text: 'hello' }],
      });

      expect(harness.chromeMock.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'YT_SUBTITLES', videoId: 'abc123' }),
      );
    });

    test('relays Netflix subtitles', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({
        type: 'NETFLIX_SUBTITLES',
        movieId: 'm1',
        lang: 'en',
        trackId: 't1',
        segments: [{ start: 0, dur: 5, text: 'scene' }],
      });

      expect(harness.chromeMock.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'NETFLIX_SUBTITLES' }),
      );
    });
  });

  describe('SERP_FETCH', () => {
    test('proxies fetch to the requested URL', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(new (globalThis as any).Response('response body', { status: 200 }));

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: JSON.stringify({ query: 'test' }),
      });

      expect(mockFetchImpl).toHaveBeenCalledWith(
        `${DEFAULT_BACKEND_URL}/api/search`,
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ query: 'test' }) }),
      );
      expect(result).toEqual(expect.objectContaining({ ok: true, text: 'response body' }));
    });

    test('returns error on non-200 status', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(new (globalThis as any).Response('error', { status: 500 }));

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: 'HTTP 500' }));
    });

    test('returns error on fetch failure', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockRejectedValueOnce(new Error('Network failure'));

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: 'Network failure' }));
    });

    test('passes apiKey as X-API-Key header when provided', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(new (globalThis as any).Response('ok', { status: 200 }));

      await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: '{}',
        apiKey: 'sk-test-key',
      });

      const callOpts = mockFetchImpl.mock.calls[0][1];
      expect(callOpts.headers).toEqual(expect.objectContaining({ 'X-API-Key': 'sk-test-key' }));
    });

    test('blocks SSRF to internal IPs (non-backend)', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: 'http://192.168.1.1/api/search',
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: expect.stringContaining('disallowed origin') }));
      expect(mockFetchImpl).not.toHaveBeenCalled();
    });

    test('blocks SSRF to localhost (non-backend)', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      // Note: localhost IS the backend in dev, so this test uses a different port
      // to simulate a non-backend localhost service
      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: 'http://localhost:8080/api/search',
        body: '{}',
      });

      // localhost:8080 has a different origin than the backend (which is localhost:8000)
      // so it gets blocked for being a disallowed origin, not a private host
      expect(result).toEqual(expect.objectContaining({ ok: false }));
      expect(mockFetchImpl).not.toHaveBeenCalled();
    });

    test('blocks SSRF to non-backend origin', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: 'https://evil.example.com/api/search',
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: expect.stringContaining('disallowed origin') }));
      expect(mockFetchImpl).not.toHaveBeenCalled();
    });

    test('blocks SSRF with unsafe scheme', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: 'file:///etc/passwd',
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: expect.stringContaining('unsafe') }));
      expect(mockFetchImpl).not.toHaveBeenCalled();
    });

    test('blocks SSRF with invalid URL', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: 'not-a-url',
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false }));
      expect(mockFetchImpl).not.toHaveBeenCalled();
    });
  });

    test('returns error on non-200 status', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(new (globalThis as any).Response('error', { status: 500 }));

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: 'HTTP 500' }));
    });

    test('returns error on fetch failure', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockRejectedValueOnce(new Error('Network failure'));

      const result = await sendMessage({
        type: 'SERP_FETCH',
        url: `${DEFAULT_BACKEND_URL}/api/search`,
        body: '{}',
      });

      expect(result).toEqual(expect.objectContaining({ ok: false, error: 'Network failure' }));
    });
  });

  describe('SAVE_TO_CLI_FEED', () => {
    test('posts payload to backend feed endpoint', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      mockFetchImpl.mockResolvedValueOnce(
        new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } }),
      );

      const payload = { title: 'Capture', url: 'https://example.com', timestamp: 1234 };
      const result = await sendMessage({ type: 'SAVE_TO_CLI_FEED', payload });

      expect(mockFetchImpl).toHaveBeenCalledWith(
        expect.stringContaining('/api/feed/save'),
        expect.objectContaining({ method: 'POST', body: JSON.stringify(payload) }),
      );
    });
  });

  describe('IMAGE_SAVE', () => {
    test('downloads image via chrome.downloads API', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({ type: 'IMAGE_SAVE', imageUrl: 'https://example.com/img.png' });

      expect(harness.chromeMock.downloads.download).toHaveBeenCalledWith(
        expect.objectContaining({ url: 'https://example.com/img.png' }),
      );
    });
  });

  describe('IMAGE_DESCRIBE', () => {
    test('stores pending query and opens side panel', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage(
        { type: 'IMAGE_DESCRIBE', imageUrl: 'https://example.com/img.png' },
        { id: 'test-extension-id', tab: { id: 1, windowId: 1 } },
      );

      expect(storage.pendingQuery).toContain('https://example.com/img.png');
      expect(storage.pendingAction).toBe('ask');
      expect(harness.chromeMock.sidePanel.open).toHaveBeenCalled();
    });
  });

  describe('OPEN_PANEL', () => {
    test('opens side panel and sends SWITCH_PANEL', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({ type: 'OPEN_PANEL', panel: 'chat' }, { id: 'test-extension-id', tab: { id: 1, windowId: 1 } });

      expect(harness.chromeMock.sidePanel.open).toHaveBeenCalled();
      expect(harness.chromeMock.runtime.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'SWITCH_PANEL', panel: 'chat' }),
      );
    });
  });

  describe('OPEN_WITH_TEXT', () => {
    test('stores pending params and opens side panel', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      await sendMessage({
        type: 'OPEN_WITH_TEXT',
        text: 'ask this',
        action: 'ask',
        url: 'https://example.com',
        title: 'Example',
      }, { id: 'test-extension-id', tab: { id: 1, windowId: 1 } });

      expect(storage.pendingQuery).toBe('ask this');
      expect(storage.pendingAction).toBe('ask');
      expect(storage.pendingUrl).toBe('https://example.com');
      expect(harness.chromeMock.sidePanel.open).toHaveBeenCalled();
    });
  });

  describe('Unknown message type', () => {
    test('returns false for unrecognized messages', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      // The switch-based handler returns false for unknown types.
      // Since we're simulating through the onMessage listener,
      // an unknown type should not trigger any specific handler.
      const result = await sendMessage({ type: 'COMPLETELY_UNKNOWN_TYPE' });
      expect(result).toBeUndefined();
    });
  });

  describe('Sender validation', () => {
    test('ignores messages from foreign extensions in WS listener', async () => {
      jest.isolateModules(() => { require('../../../background'); });

      // The first onMessage listener (WS routing) checks sender.id
      // Messages from foreign extensions should return false
      const listener = harness.onMessageListeners[0];
      const sendResponse = jest.fn();
      const result = listener(
        { type: 'AURA_WS_IN', payload: '{}' },
        { id: 'different-extension-id' },
        sendResponse,
      );
      expect(result).toBe(false);
    });
  });
});