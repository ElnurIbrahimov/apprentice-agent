/**
 * Thin fetch wrapper that always sends the session cookie.
 *
 * The web UI is currently same-origin with the backend, which means
 * browsers send the `aura_session` cookie by default. But if the UI ever
 * gets deployed on a different subdomain than the API — or if a future
 * service worker intercepts requests — `credentials: 'same-origin'`
 * silently omits the cookie. Calling sites should not have to remember.
 *
 * Always use `apiFetch` for requests that need auth instead of bare
 * `fetch`. If you need `no-store` or a different `cache` mode, override
 * via the second argument as usual; your overrides win.
 */
/**
 * Thin fetch wrapper that always sends the session cookie.
 *
 * Adds a 15-second timeout and one automatic retry on transient
 * network failures (TypeError) or HTTP 503.
 */
export function apiFetch(
  input: RequestInfo | URL,
  init?: RequestInit & { retry?: boolean }
): Promise<Response> {
  const controller = new AbortController();
  const signal = init?.signal;

  // Chain caller's abort signal into our controller
  if (signal) {
    const onAbort = () => controller.abort();
    signal.addEventListener('abort', onAbort, { once: true });
  }

  const timeoutId = setTimeout(() => controller.abort(), 15000);

  return fetch(input, {
    credentials: 'include',
    ...(init ?? {}),
    signal: controller.signal,
  }).catch((err) => {
    if (
      init?.retry !== false &&
      (err instanceof TypeError || (err as any)?.status === 503)
    ) {
      return fetch(input, { credentials: 'include', ...(init ?? {}) });
    }
    throw err;  // re-throw on final failure
  }).finally(() => {
    clearTimeout(timeoutId);
  });
}
