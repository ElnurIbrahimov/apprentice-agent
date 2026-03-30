import { HTTP, getAuthHeaders } from '../api';

/**
 * Parse SSE stream and yield text chunks.
 * Shared logic for both chat and raw generation endpoints.
 */
async function* parseSSEStream(
  resp: Response,
): AsyncGenerator<string, void, unknown> {
  const reader = resp.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Parse SSE lines
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6);
      if (data === '[DONE]') return;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'chunk' && parsed.content) {
          yield parsed.content;
        } else if (parsed.type === 'error') {
          throw new Error(parsed.error || 'Stream error');
        } else if (parsed.type === 'done') {
          return;
        }
      } catch (e: any) {
        // Re-throw non-JSON-parse errors (actual stream/application errors)
        if (!(e instanceof SyntaxError)) throw e;
        // skip unparseable SSE lines (incomplete JSON chunks)
      }
    }
  }
}

/**
 * Stream chat response from the backend SSE endpoint.
 * Goes through the full agent pipeline (personality, tools, emotion).
 * Yields text chunks as they arrive.
 */
export async function* streamChat(
  message: string,
  model?: string,
  signal?: AbortSignal
): AsyncGenerator<string, void, unknown> {
  const resp = await fetch(`${HTTP}/api/chat/sse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ message, model: model || undefined }),
    signal,
  });

  if (!resp.ok) {
    const d = await resp.json().catch(() => ({}));
    throw new Error((d as any).detail || `HTTP ${resp.status}`);
  }

  yield* parseSSEStream(resp);
}

/**
 * Stream raw LLM generation — bypasses the agent pipeline entirely.
 * Used by WebCreator and Artifacts for direct HTML/code generation
 * with custom system prompts, no personality or tool interference.
 */
export async function* streamRawGenerate(
  message: string,
  opts: {
    systemPrompt?: string;
    model?: string;
    history?: Array<{ role: string; content: string }>;
    signal?: AbortSignal;
  } = {}
): AsyncGenerator<string, void, unknown> {
  const resp = await fetch(`${HTTP}/api/generate/raw`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({
      message,
      system_prompt: opts.systemPrompt || undefined,
      model: opts.model || undefined,
      history: opts.history || undefined,
    }),
    signal: opts.signal,
  });

  if (!resp.ok) {
    const d = await resp.json().catch(() => ({}));
    throw new Error((d as any).detail || `HTTP ${resp.status}`);
  }

  yield* parseSSEStream(resp);
}

/**
 * Non-streaming fallback — calls the regular /api/chat endpoint.
 * Returns the full response text.
 */
export async function chatFallback(
  message: string,
  model?: string,
  signal?: AbortSignal
): Promise<string> {
  const resp = await fetch(`${HTTP}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ message, model: model || undefined }),
    signal,
  });
  if (!resp.ok) {
    const d = await resp.json().catch(() => ({}));
    throw new Error((d as any).detail || `HTTP ${resp.status}`);
  }
  const data = await resp.json();
  return data.response || data.text || data.content || data.reply || '';
}
