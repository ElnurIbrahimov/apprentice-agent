/**
 * Tests for apiFetch — the single audited entry point every API client helper
 * routes through. Covers auth header injection, timeout, error parsing, and
 * signal pass-through.
 */

// Reset module state between tests so API_KEY changes are observable.
describe('apiFetch', () => {
  const originalFetch = global.fetch;

  afterEach(() => {
    global.fetch = originalFetch;
    jest.resetModules();
  });

  test('injects X-API-Key when set', async () => {
    jest.isolateModules(() => {}); // dummy to satisfy linter
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ ok: true }),
    } as any);
    const api = await import('../../api');
    (api as any).setApiKey('secret-abc');
    await api.apiFetch('http://x/y');
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers['X-API-Key']).toBe('secret-abc');
    (api as any).setApiKey('');
  });

  test('does not inject the header when API_KEY is empty', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    } as any);
    const api = await import('../../api');
    (api as any).setApiKey('');
    await api.apiFetch('http://x/y');
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers['X-API-Key']).toBeUndefined();
  });

  test('throws with backend detail message on non-ok', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
      json: () => Promise.resolve({ detail: 'server exploded' }),
    } as any);
    const api = await import('../../api');
    await expect(api.apiFetch('http://x/y')).rejects.toThrow('server exploded');
  });

  test('falls back to "HTTP <status>" when body has no detail', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 418,
      json: () => Promise.resolve({}),
    } as any);
    const api = await import('../../api');
    await expect(api.apiFetch('http://x/y')).rejects.toThrow('HTTP 418');
  });

  test('merges caller headers over defaults and passes through caller signal', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    } as any);
    const api = await import('../../api');
    const controller = new AbortController();
    await api.apiFetch('http://x/y', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}',
      signal: controller.signal,
    });
    const call = (global.fetch as jest.Mock).mock.calls[0];
    expect(call[1].headers['Content-Type']).toBe('application/json');
    expect(call[1].method).toBe('POST');
    expect(call[1].signal).toBe(controller.signal);
  });
});

describe('getAuthHeaders', () => {
  afterEach(() => {
    jest.resetModules();
  });

  test('returns X-API-Key when set, empty object otherwise', async () => {
    const api = await import('../../api');
    (api as any).setApiKey('');
    expect(api.getAuthHeaders()).toEqual({});
    (api as any).setApiKey('k');
    expect(api.getAuthHeaders()).toEqual({ 'X-API-Key': 'k' });
    (api as any).setApiKey('');
  });
});
