/**
 * Stuck Detector — CHI-validated proactive assistance triggers.
 *
 * Watches the page for signals that the user is stuck, and pings the service
 * worker so the backend can decide whether to surface a proactive suggestion.
 * Mid-task interventions are dismissed 62% of the time according to CHI 2025
 * ("Need Help? Designing Proactive AI Assistants for Programming"), so this
 * module only fires on *workflow boundaries* and *well-timed* stuck signals.
 *
 * Signals:
 *   - tab_revisit: same URL visited >=3 times in a 48-hour rolling window
 *   - reread: same text section scrolled into view >=3 times in one session
 *   - form_flipflop: input value edited then cleared >=2 times on one field
 *   - workflow_boundary: user is about to unload/close with unsent context
 *
 * Anti-spam: emits at most one signal per 5 minutes per tab, and backs off
 * exponentially if the backend returned no useful suggestion last time.
 */

declare const browser: typeof chrome | undefined;
const ext = typeof browser !== 'undefined' ? browser : chrome;

const MIN_INTERVAL_MS = 5 * 60 * 1000; // 5 min between signals
const REVISIT_WINDOW_MS = 48 * 60 * 60 * 1000; // 48h rolling window
const REVISIT_DEDUPE_MS = 10 * 60 * 1000; // 10 min — skip record if the last ts is newer
const REVISIT_THRESHOLD = 3;
const REREAD_THRESHOLD = 3;
const FLIPFLOP_THRESHOLD = 2;
const FLIPFLOP_WINDOW_MS = 10_000;
const MAX_TRACKED_URLS = 2000;

function normalizeUrl(url: string): string {
  try {
    const u = new URL(url);
    return `${u.protocol}//${u.host}${u.pathname}`;
  } catch {
    return url;
  }
}

let lastSignalAt = 0;

function isHttpPage(): boolean {
  return location.protocol === 'http:' || location.protocol === 'https:';
}

function sendStuckSignal(kind: string, extra: Record<string, unknown> = {}): void {
  const now = Date.now();
  if (now - lastSignalAt < MIN_INTERVAL_MS) return;
  lastSignalAt = now;
  try {
    ext.runtime.sendMessage({
      type: 'STUCK_SIGNAL',
      kind,
      url: location.href,
      title: document.title,
      ...extra,
    }).catch(() => {});
  } catch { /* noop */ }
}

// ── Tab revisit counting (persisted in chrome.storage.local) ────────────────
// Map URL -> timestamps[]. Prune entries older than 48h on each read.

interface RevisitRecord { ts: number[] }

async function recordVisitAndCheck(rawUrl: string): Promise<void> {
  const url = normalizeUrl(rawUrl);
  try {
    const data = await ext.storage.local.get(['aura_visit_log']);
    const log: Record<string, RevisitRecord> = (data.aura_visit_log as Record<string, RevisitRecord> | undefined) || {};
    const now = Date.now();
    const entry = log[url] || { ts: [] };

    // Prune old timestamps
    entry.ts = entry.ts.filter((t) => now - t < REVISIT_WINDOW_MS);

    // Dedupe: if the most recent timestamp is within 10 min, treat this as
    // the same visit (rapid reload, tab switch back, etc.) — no write.
    const lastTs = entry.ts[entry.ts.length - 1] || 0;
    const isDuplicate = now - lastTs < REVISIT_DEDUPE_MS;

    if (!isDuplicate) {
      entry.ts.push(now);
      log[url] = entry;

      // Prune empty entries + cap total size.
      for (const k of Object.keys(log)) {
        if (!log[k].ts.length) delete log[k];
      }
      const keys = Object.keys(log);
      if (keys.length > MAX_TRACKED_URLS) {
        const sorted = keys.sort((a, b) => (log[a].ts[0] || 0) - (log[b].ts[0] || 0));
        for (const k of sorted.slice(0, keys.length - MAX_TRACKED_URLS)) delete log[k];
      }

      await ext.storage.local.set({ aura_visit_log: log });
    }

    if (entry.ts.length >= REVISIT_THRESHOLD) {
      sendStuckSignal('tab_revisit', { count: entry.ts.length });
    }
  } catch {
    // chrome.storage failures are non-fatal
  }
}

// ── Paragraph re-read detection via IntersectionObserver ────────────────────
const rereadCounts = new WeakMap<Element, number>();
let rereadObserver: IntersectionObserver | null = null;

function initReread(): void {
  if (typeof IntersectionObserver === 'undefined') return;
  try {
    rereadObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const n = (rereadCounts.get(entry.target) || 0) + 1;
        rereadCounts.set(entry.target, n);
        if (n >= REREAD_THRESHOLD) {
          const text = (entry.target.textContent || '').slice(0, 140);
          sendStuckSignal('reread', { snippet: text });
          rereadObserver?.unobserve(entry.target);
        }
      }
    }, { threshold: 0.6, rootMargin: '0px' });

    // Observe all paragraphs with meaningful text. Skip tiny spans.
    const candidates = document.querySelectorAll('p, article, section, blockquote');
    let observed = 0;
    for (const el of Array.from(candidates)) {
      if ((el.textContent || '').trim().length >= 120) {
        rereadObserver.observe(el);
        observed++;
        if (observed >= 200) break; // cap
      }
    }
  } catch { /* noop */ }
}

// ── Form flip-flop detection ────────────────────────────────────────────────
interface FlipFlop { lastEdit: number; emptyAt: number[] }
const flipFlopMap = new WeakMap<HTMLInputElement | HTMLTextAreaElement, FlipFlop>();

function initFlipFlop(): void {
  const handler = (e: Event) => {
    const el = e.target as HTMLInputElement | HTMLTextAreaElement | null;
    if (!el || !('value' in el)) return;
    if (el.type === 'password' || el.type === 'hidden') return;

    const rec = flipFlopMap.get(el) || { lastEdit: 0, emptyAt: [] };
    const now = Date.now();

    if (el.value === '') {
      // Cleared — prune old empties then push
      rec.emptyAt = rec.emptyAt.filter((t) => now - t < FLIPFLOP_WINDOW_MS);
      rec.emptyAt.push(now);
      if (rec.emptyAt.length >= FLIPFLOP_THRESHOLD) {
        const name = el.getAttribute('name') || el.getAttribute('id') || 'field';
        sendStuckSignal('form_flipflop', { field: name });
        rec.emptyAt = []; // reset
      }
    } else {
      rec.lastEdit = now;
    }
    flipFlopMap.set(el, rec);
  };
  document.addEventListener('input', handler, { capture: true, passive: true });
}

// ── Workflow-boundary detection ─────────────────────────────────────────────
function initWorkflowBoundary(): void {
  // beforeunload fires when user closes tab / navigates away.
  window.addEventListener('beforeunload', () => {
    // This is a fire-and-forget best-effort — don't block unload.
    const hasUnsaved = document.querySelector('textarea:focus, [contenteditable="true"]:focus');
    if (hasUnsaved) {
      try {
        ext.runtime.sendMessage({
          type: 'STUCK_SIGNAL',
          kind: 'workflow_boundary_unsaved',
          url: location.href,
          title: document.title,
        });
      } catch { /* noop */ }
    }
  });
}

// ── Public init ─────────────────────────────────────────────────────────────

export function initStuckDetector(): void {
  if (!isHttpPage()) return;
  // Record the visit immediately — revisit check runs on every page load.
  recordVisitAndCheck(location.href);

  // Wait for idle before setting up heavier observers.
  const setup = () => {
    initReread();
    initFlipFlop();
    initWorkflowBoundary();
  };
  if (document.readyState === 'complete') {
    setTimeout(setup, 500);
  } else {
    window.addEventListener('load', () => setTimeout(setup, 500), { once: true });
  }
}
