/**
 * Ambient memory surfacing — when the user lands on a page, ask Aura's memory
 * backend if there's anything relevant (previous memories or prior lifelog
 * visits). If so, push a surface hint to the sidebar so it can surface the
 * recall without the user asking.
 *
 * Flow:
 *  1. On page load (once per URL per hour), content script sends
 *     AMBIENT_SURFACE_REQUEST { url, title } to background.
 *  2. Background queries /api/memory/search and /api/lifelog/search in parallel.
 *  3. Background forwards any hits to the sidebar via runtime.sendMessage
 *     with type AMBIENT_SURFACE_HINT.
 *  4. Sidebar's App.tsx handler routes it to a dismissable notification card.
 *
 * Gated by chrome.storage.local.ambientSurfaceEnabled (default off).
 */

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

const SURFACE_COOLDOWN_MS = 60 * 60 * 1000; // one surface per URL per hour
const MIN_DELAY_MS = 2500; // wait for page to stabilize

const SENSITIVE_HOSTS = [
  'accounts.google.com', 'login.microsoftonline.com', 'okta.com',
  '1password.com', 'lastpass.com', 'bitwarden.com', 'dashlane.com',
];

function isDenylisted(): boolean {
  const host = location.hostname;
  if (SENSITIVE_HOSTS.some((d) => host === d || host.endsWith('.' + d))) return true;
  if (/\bbank\b|\bpayment\b|\bcheckout\b/i.test(host)) return true;
  return false;
}

function isLoggablePage(): boolean {
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return false;
  if (isDenylisted()) return false;
  return true;
}

function urlKey(): string {
  try {
    const u = new URL(location.href);
    return u.origin + u.pathname;
  } catch {
    return location.href;
  }
}

async function wasSurfacedRecently(): Promise<boolean> {
  try {
    const data: any = await ext.storage?.local?.get(['ambientSurfaceLog']);
    const log: Record<string, number> = data?.ambientSurfaceLog || {};
    const key = urlKey();
    const last = log[key];
    if (!last) return false;
    return Date.now() - last < SURFACE_COOLDOWN_MS;
  } catch {
    return true; // fail closed
  }
}

async function markSurfaced(): Promise<void> {
  try {
    const data: any = await ext.storage?.local?.get(['ambientSurfaceLog']);
    const log: Record<string, number> = data?.ambientSurfaceLog || {};
    log[urlKey()] = Date.now();
    // Garbage-collect old entries (>24h)
    const cutoff = Date.now() - 24 * 60 * 60 * 1000;
    for (const k in log) {
      if (log[k] < cutoff) delete log[k];
    }
    ext.storage?.local?.set({ ambientSurfaceLog: log });
  } catch { /* silent */ }
}

async function requestSurface(): Promise<void> {
  if (!isLoggablePage()) return;
  if (await wasSurfacedRecently()) return;

  const title = document.title || '';
  if (!title.trim()) return;

  try {
    ext.runtime.sendMessage({
      type: 'AMBIENT_SURFACE_REQUEST',
      url: location.href,
      title,
      host: location.hostname,
    });
    markSurfaced();
  } catch { /* silent */ }
}

export function initAmbientSurface(): void {
  if (!isLoggablePage()) return;
  // Check opt-in flag before doing anything
  try {
    ext.storage?.local?.get(['ambientSurfaceEnabled'], (d: any) => {
      if (!d?.ambientSurfaceEnabled) return;
      // Wait for page to stabilize before asking
      setTimeout(() => {
        if (document.visibilityState === 'visible') {
          requestSurface();
        }
      }, MIN_DELAY_MS);
    });
  } catch { /* silent */ }
}
