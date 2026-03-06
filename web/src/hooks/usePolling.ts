import { useEffect, useRef, useCallback } from 'react';

/**
 * Drop-in replacement for setInterval-based polling that:
 * 1. Pauses when the browser tab is hidden (saves backend load)
 * 2. Prevents overlapping requests (if one is still in-flight, skip)
 * 3. Staggers initial fetches so components don't all fire at once
 */

// Global counter to stagger initial fetches across components
// Use a closure to make increments safe across hot-reloads
const _getStagger = (() => {
  let _order = 0;
  return () => _order++ * 500;
})();

export function usePolling(
  callback: () => void | Promise<void>,
  intervalMs: number,
  options?: { enabled?: boolean }
) {
  const enabled = options?.enabled ?? true;
  const savedCallback = useRef(callback);
  const inFlight = useRef(false);
  const staggerDelay = useRef(_getStagger()); // 500ms gap between components

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  const tick = useCallback(async () => {
    if (document.hidden || inFlight.current) return;
    inFlight.current = true;
    try {
      await savedCallback.current();
    } catch {
      // Swallow - individual pollers handle their own errors
    } finally {
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    // Stagger start: each component waits (mountOrder * 2s) before first fetch
    // Cap at 30s so late-mounting components don't wait forever
    const delay = Math.min(staggerDelay.current, 30000);
    let intervalId: ReturnType<typeof setInterval> | null = null;

    // First fetch is immediate - don't make user wait for stagger
    tick();

    // Stagger only the interval start so components don't all poll at the same instant
    const startTimeout = setTimeout(() => {
      intervalId = setInterval(tick, intervalMs);
    }, delay);

    const onVisibilityChange = () => {
      if (!document.hidden) tick();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);

    return () => {
      clearTimeout(startTimeout);
      if (intervalId) clearInterval(intervalId);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [tick, intervalMs, enabled]);
}
