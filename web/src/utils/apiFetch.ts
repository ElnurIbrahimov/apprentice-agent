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
export function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return fetch(input, { credentials: 'include', ...(init ?? {}) });
}
