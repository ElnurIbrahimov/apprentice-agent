/// <reference types="chrome" />

/**
 * Thin wrapper over `chrome.debugger` (Chrome DevTools Protocol).
 *
 * Used by Power Mode to give the browser agent real input events and
 * screenshot-based vision. Attaches once per agent run, detaches when the
 * loop finishes (success, error, or user stop). Auto-detaches on tab close.
 *
 * Chrome shows a yellow "AURA started debugging this browser" banner while
 * attached — this is unsuppressible. Callers should surface this warning in
 * Settings before first use.
 */

const CDP_VERSION = '1.3';

export interface CdpResult<T = unknown> {
  ok: boolean;
  data?: T;
  error?: string;
}

const attachedTabs = new Set<number>();

function targetFor(tabId: number): chrome.debugger.Debuggee {
  return { tabId };
}

function sendCommand<T = unknown>(
  tabId: number,
  method: string,
  params?: Record<string, unknown>,
): Promise<CdpResult<T>> {
  return new Promise((resolve) => {
    try {
      chrome.debugger.sendCommand(targetFor(tabId), method, params || {}, (result: unknown) => {
        const err = chrome.runtime.lastError?.message;
        if (err) {
          resolve({ ok: false, error: err });
        } else {
          resolve({ ok: true, data: result as T });
        }
      });
    } catch (e) {
      resolve({ ok: false, error: (e as Error).message || String(e) });
    }
  });
}

/** Attach CDP to a tab. Idempotent — safe to call twice. */
export function attachCDP(tabId: number): Promise<CdpResult> {
  return new Promise((resolve) => {
    if (attachedTabs.has(tabId)) {
      resolve({ ok: true });
      return;
    }
    try {
      chrome.debugger.attach(targetFor(tabId), CDP_VERSION, () => {
        const err = chrome.runtime.lastError?.message;
        if (err) {
          resolve({ ok: false, error: err });
          return;
        }
        attachedTabs.add(tabId);
        // Enable Page + Input + Runtime domains for screenshots + events.
        Promise.all([
          sendCommand(tabId, 'Page.enable'),
          sendCommand(tabId, 'Runtime.enable'),
          sendCommand(tabId, 'DOM.enable'),
        ]).then(() => resolve({ ok: true }));
      });
    } catch (e) {
      resolve({ ok: false, error: (e as Error).message || String(e) });
    }
  });
}

/** Detach CDP from a tab. Idempotent — safe to call when not attached. */
export function detachCDP(tabId: number): Promise<CdpResult> {
  return new Promise((resolve) => {
    if (!attachedTabs.has(tabId)) {
      resolve({ ok: true });
      return;
    }
    try {
      chrome.debugger.detach(targetFor(tabId), () => {
        attachedTabs.delete(tabId);
        // Ignore lastError — detach from a closed tab is expected to fail.
        resolve({ ok: true });
      });
    } catch {
      attachedTabs.delete(tabId);
      resolve({ ok: true });
    }
  });
}

/** Is CDP currently attached to this tab (from our side)? */
export function isAttached(tabId: number): boolean {
  return attachedTabs.has(tabId);
}

interface ScreenshotResult {
  data: string; // base64, no prefix
}

/** Capture a JPEG screenshot of the tab via CDP. */
export async function cdpScreenshot(tabId: number, quality = 75): Promise<CdpResult<string>> {
  const r = await sendCommand<ScreenshotResult>(tabId, 'Page.captureScreenshot', {
    format: 'jpeg',
    quality,
    captureBeyondViewport: false,
  });
  if (!r.ok || !r.data) return { ok: false, error: r.error || 'no screenshot' };
  return { ok: true, data: r.data.data };
}

/** Dispatch a real mouse click at (x, y) via CDP Input domain. */
export async function cdpClick(tabId: number, x: number, y: number): Promise<CdpResult> {
  const common = { x, y, button: 'left', clickCount: 1 };
  const down = await sendCommand(tabId, 'Input.dispatchMouseEvent', { type: 'mousePressed', ...common });
  if (!down.ok) return down;
  const up = await sendCommand(tabId, 'Input.dispatchMouseEvent', { type: 'mouseReleased', ...common });
  return up;
}

/** Type text by dispatching real key events via CDP. */
export async function cdpType(tabId: number, text: string): Promise<CdpResult> {
  for (const ch of text) {
    const r = await sendCommand(tabId, 'Input.insertText', { text: ch });
    if (!r.ok) return r;
  }
  return { ok: true };
}

/** Dispatch a single key press (e.g. "Enter", "Tab"). */
export async function cdpKeyPress(tabId: number, key: string): Promise<CdpResult> {
  // Minimal key mapping; CDP infers windowsVirtualKeyCode from key name for common keys.
  const keyEvent = { key, code: key, text: '', unmodifiedText: '' };
  const down = await sendCommand(tabId, 'Input.dispatchKeyEvent', { type: 'keyDown', ...keyEvent });
  if (!down.ok) return down;
  const up = await sendCommand(tabId, 'Input.dispatchKeyEvent', { type: 'keyUp', ...keyEvent });
  return up;
}

/** Scroll the page by dispatching a wheel event at the center of the viewport. */
export async function cdpScroll(tabId: number, deltaY: number): Promise<CdpResult> {
  // Use Runtime.evaluate as a reliable cross-target scroll primitive; CDP's
  // Input.dispatchMouseEvent with type 'mouseWheel' only works in isolated world.
  return sendCommand(tabId, 'Runtime.evaluate', {
    expression: `window.scrollBy(0, ${Number(deltaY) || 0})`,
    returnByValue: true,
  });
}

/** Resolve a CSS selector to viewport coordinates via Runtime.evaluate. */
export async function cdpResolveSelector(
  tabId: number,
  selector: string,
): Promise<CdpResult<{ x: number; y: number; width: number; height: number }>> {
  const expr = `
    (() => {
      const el = document.querySelector(${JSON.stringify(selector)});
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { x: r.left + r.width / 2, y: r.top + r.height / 2, width: r.width, height: r.height };
    })()
  `;
  interface EvalResult {
    result: { value?: { x: number; y: number; width: number; height: number } | null };
  }
  const r = await sendCommand<EvalResult>(tabId, 'Runtime.evaluate', {
    expression: expr,
    returnByValue: true,
  });
  if (!r.ok) return { ok: false, error: r.error };
  const value = r.data?.result?.value;
  if (!value) return { ok: false, error: 'selector did not match' };
  return { ok: true, data: value };
}

/** Register a global tab-close handler that detaches from closed tabs. */
export function installCdpTabCleanup(): void {
  if (!chrome.tabs?.onRemoved) return;
  chrome.tabs.onRemoved.addListener((tabId: number) => {
    if (attachedTabs.has(tabId)) {
      try { chrome.debugger.detach({ tabId }); } catch { /* noop */ }
      attachedTabs.delete(tabId);
    }
  });
  // Also handle unexpected detaches (user clicked "Cancel" on the banner).
  chrome.debugger.onDetach.addListener((source: chrome.debugger.Debuggee) => {
    if (source.tabId != null) attachedTabs.delete(source.tabId);
  });
}
