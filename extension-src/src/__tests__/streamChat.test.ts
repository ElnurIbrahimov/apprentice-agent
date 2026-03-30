/**
 * Tests for streamChat SSE parsing utilities.
 * Since jsdom lacks ReadableStream/Response, we test by mocking fetch
 * to return a custom object with a reader pattern.
 */

const mockFetch = jest.fn();
global.fetch = mockFetch;

jest.mock('../api', () => ({
  HTTP: 'http://localhost:8000',
  getAuthHeaders: () => ({ 'X-API-Key': 'test-key' }),
}));

import { streamRawGenerate } from '../utils/streamChat';

/** Create a mock Response with a readable body from SSE text */
function mockSSE(sseText: string, status = 200) {
  const encoded = new TextEncoder().encode(sseText);
  let read = false;
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => { try { return JSON.parse(sseText); } catch { return {}; } },
    body: {
      getReader: () => ({
        read: async () => {
          if (!read) { read = true; return { done: false, value: encoded }; }
          return { done: true, value: undefined };
        },
      }),
    },
  };
}

describe('streamRawGenerate', () => {
  afterEach(() => mockFetch.mockReset());

  test('yields chunks from SSE stream', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE(
      'data: {"type":"chunk","content":"Hello"}\ndata: {"type":"chunk","content":" World"}\ndata: [DONE]\n'
    ));

    const chunks: string[] = [];
    for await (const c of streamRawGenerate('test')) chunks.push(c);

    expect(chunks).toEqual(['Hello', ' World']);
  });

  test('sends correct request body', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE('data: {"type":"done"}\n'));

    for await (const _ of streamRawGenerate('my prompt', {
      systemPrompt: 'Be helpful',
      model: 'test-model',
      history: [{ role: 'user', content: 'hi' }],
    })) { /* consume */ }

    const call = mockFetch.mock.calls[0];
    expect(call[0]).toBe('http://localhost:8000/api/generate/raw');
    const body = JSON.parse(call[1].body);
    expect(body.message).toBe('my prompt');
    expect(body.system_prompt).toBe('Be helpful');
    expect(body.model).toBe('test-model');
    expect(body.history).toEqual([{ role: 'user', content: 'hi' }]);
  });

  test('throws on stream error', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE(
      'data: {"type":"chunk","content":"partial"}\ndata: {"type":"error","error":"Model overloaded"}\n'
    ));

    const chunks: string[] = [];
    await expect(async () => {
      for await (const c of streamRawGenerate('test')) chunks.push(c);
    }).rejects.toThrow('Model overloaded');
    expect(chunks).toEqual(['partial']);
  });

  test('throws on non-200 response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'Unauthorized' }),
    });

    await expect(async () => {
      for await (const _ of streamRawGenerate('test')) { /* */ }
    }).rejects.toThrow('Unauthorized');
  });

  test('handles done event', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE(
      'data: {"type":"chunk","content":"only"}\ndata: {"type":"done"}\n'
    ));

    const chunks: string[] = [];
    for await (const c of streamRawGenerate('test')) chunks.push(c);
    expect(chunks).toEqual(['only']);
  });

  test('skips malformed JSON lines', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE(
      'data: {"type":"chunk","content":"good"}\ndata: {BROKEN}\ndata: {"type":"done"}\n'
    ));

    const chunks: string[] = [];
    for await (const c of streamRawGenerate('test')) chunks.push(c);
    expect(chunks).toEqual(['good']);
  });

  test('includes auth headers', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE('data: {"type":"done"}\n'));

    for await (const _ of streamRawGenerate('test')) { /* */ }

    const headers = mockFetch.mock.calls[0][1].headers;
    expect(headers['X-API-Key']).toBe('test-key');
    expect(headers['Content-Type']).toBe('application/json');
  });

  test('handles empty chunk content gracefully', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE(
      'data: {"type":"chunk","content":""}\ndata: {"type":"chunk","content":"real"}\ndata: {"type":"done"}\n'
    ));

    const chunks: string[] = [];
    for await (const c of streamRawGenerate('test')) chunks.push(c);
    // Empty content chunks are skipped (content is falsy)
    expect(chunks).toEqual(['real']);
  });

  test('handles response with no body reader', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true, status: 200,
      body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
    });

    const chunks: string[] = [];
    for await (const c of streamRawGenerate('test')) chunks.push(c);
    expect(chunks).toEqual([]);
  });

  test('omits undefined optional params from body', async () => {
    mockFetch.mockResolvedValueOnce(mockSSE('data: {"type":"done"}\n'));

    for await (const _ of streamRawGenerate('test')) { /* */ }

    const body = JSON.parse(mockFetch.mock.calls[0][1].body);
    expect(body.message).toBe('test');
    expect(body.system_prompt).toBeUndefined();
    expect(body.model).toBeUndefined();
    expect(body.history).toBeUndefined();
  });
});
