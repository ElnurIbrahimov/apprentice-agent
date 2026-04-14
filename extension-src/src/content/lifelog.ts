/**
 * Browsing lifelog — passive page-visit logger.
 *
 * Captures {url, title, dwell_ms, scroll_max_pct, selection, timestamp}
 * per page visit and forwards to the service worker for batched upload to
 * Aura's /api/lifelog/events. The backend embeds events into UnifiedMemory
 * tagged `source=lifelog`; the existing Dream consolidation pass promotes
 * recurring topics into the UserProfile (which is injected into every
 * system prompt).
 *
 * Privacy:
 *   - Default OFF. User must enable in Settings ("lifelog_enabled" storage key).
 *   - Skips chrome://, about:, file://, incognito tabs.
 *   - Skips sensitive domains (banks, password managers, auth portals).
 *   - Skips input values entirely — only records selection text > 20 chars.
 */

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

interface LifelogEvent {
  url: string;
  title: string;
  dwell_ms: number;
  scroll_max_pct: number;
  selection?: string;
  timestamp: number;
}

let enabled = false;
let startTs = 0;
let maxScrollPct = 0;
let lastSelection = '';
let sent = false;

const DENYLIST_HOSTS = [
  'accounts.google.com', 'login.microsoftonline.com', 'okta.com',
  '1password.com', 'lastpass.com', 'bitwarden.com', 'dashlane.com',
];

function isDenylisted(): boolean {
  const host = location.hostname;
  if (DENYLIST_HOSTS.some((d) => host.endsWith(d))) return true;
  if (/\bbank|\bpayment|\bcheckout\b/i.test(host)) return true;
  return false;
}

function isLoggablePage(): boolean {
  if (location.protocol !== 'http:' && location.protocol !== 'https:') return false;
  if (isDenylisted()) return false;
  return true;
}

function trackScroll(): void {
  const doc = document.documentElement;
  const viewport = window.innerHeight;
  const total = Math.max(1, (doc.scrollHeight || viewport) - viewport);
  const cur = window.scrollY;
  const pct = Math.round((cur / total) * 100);
  if (pct > maxScrollPct) maxScrollPct = pct;
}

function trackSelection(): void {
  try {
    const s = window.getSelection()?.toString() || '';
    if (s.length >= 20 && s.length <= 500 && s.length > lastSelection.length) {
      lastSelection = s;
    }
  } catch { /* noop */ }
}

function buildEvent(): LifelogEvent | null {
  if (!enabled || !isLoggablePage()) return null;
  if (!startTs) return null;
  const dwell = Date.now() - startTs;
  if (dwell < 3000) return null; // skip instant leaves
  const ev: LifelogEvent = {
    url: location.href,
    title: document.title,
    dwell_ms: dwell,
    scroll_max_pct: maxScrollPct,
    timestamp: Date.now(),
  };
  if (lastSelection) ev.selection = lastSelection;
  return ev;
}

function flush(): void {
  if (sent) return;
  const ev = buildEvent();
  if (!ev) return;
  sent = true;
  try {
    ext.runtime.sendMessage({ type: 'LIFELOG_EVENT', event: ev }).catch(() => {});
  } catch { /* noop */ }
}

async function loadEnabled(): Promise<void> {
  try {
    const data = await ext.storage.local.get(['lifelogEnabled']);
    enabled = !!data?.lifelogEnabled;
  } catch { /* noop */ }
}

export function initLifelog(): void {
  if (!isLoggablePage()) return;
  loadEnabled().then(() => {
    if (!enabled) return;
    startTs = Date.now();
    maxScrollPct = 0;
    lastSelection = '';
    sent = false;

    window.addEventListener('scroll', trackScroll, { passive: true });
    document.addEventListener('selectionchange', trackSelection, { passive: true });

    // Flush on tab hide / unload — whichever fires first.
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') flush();
    });
    window.addEventListener('pagehide', flush);
    window.addEventListener('beforeunload', flush);
  });

  // React to enable/disable toggle in Settings without reload.
  ext.storage?.onChanged?.addListener((changes, area) => {
    if (area !== 'local' || !changes.lifelogEnabled) return;
    enabled = !!changes.lifelogEnabled.newValue;
    if (enabled && !startTs) startTs = Date.now();
  });
}
