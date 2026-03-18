/**
 * AURA New Tab — Vanilla JS (no React, no heavy deps)
 * Keeps bundle under 50KB.
 */

const CONV_LIST_KEY = 'aura_conversations';
const BACKEND_URL = 'http://localhost:8000';

const QUOTES = [
  'The best way to predict the future is to invent it. \u2014 Alan Kay',
  'Simplicity is the ultimate sophistication. \u2014 Leonardo da Vinci',
  'First, solve the problem. Then, write the code. \u2014 John Johnson',
  'Code is like humor. When you have to explain it, it is bad. \u2014 Cory House',
  'The only way to do great work is to love what you do. \u2014 Steve Jobs',
  'Stay hungry, stay foolish. \u2014 Steve Jobs',
  'Think different.',
  'Move fast and break things. Then fix them.',
  'The computer was born to solve problems that did not exist before. \u2014 Bill Gates',
  'Any sufficiently advanced technology is indistinguishable from magic. \u2014 Arthur C. Clarke',
  'Intelligence is the ability to adapt to change. \u2014 Stephen Hawking',
  'Not all those who wander are lost. \u2014 J.R.R. Tolkien',
  'Imagination is more important than knowledge. \u2014 Albert Einstein',
  'Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away. \u2014 Saint-Exup\u00E9ry',
  'The question of whether a computer can think is no more interesting than the question of whether a submarine can swim. \u2014 Dijkstra',
];

interface ConversationMeta {
  id: string;
  title: string;
  timestamp: number;
  messageCount: number;
}

/* ── Helpers ── */

function formatTime(d: Date): string {
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

function formatDate(d: Date): string {
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  const days = Math.floor(hrs / 24);
  if (days < 7) return days + 'd ago';
  return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function getDayQuote(): string {
  const dayIndex = Math.floor(Date.now() / 86400000) % QUOTES.length;
  return QUOTES[dayIndex];
}

function esc(s: string): string {
  const el = document.createElement('span');
  el.textContent = s;
  return el.innerHTML;
}

/* ── Chrome storage ── */

const ext: typeof chrome | null =
  typeof chrome !== 'undefined' && chrome?.storage ? chrome : null;

function storageGet(keys: string[]): Promise<Record<string, any>> {
  return new Promise((resolve) => {
    if (ext?.storage?.local) {
      ext.storage.local.get(keys, (d: any) => resolve(d || {}));
    } else {
      resolve({});
    }
  });
}

/* ── Send message to background ── */

function sendToBackground(msg: Record<string, unknown>) {
  if (ext?.runtime?.sendMessage) {
    ext.runtime.sendMessage(msg);
  }
}

/* ── Build UI ── */

function mount() {
  const root = document.getElementById('root')!;
  const now = new Date();

  root.innerHTML = `
    <div class="nt-clock">
      <div class="nt-clock-time" id="clock-time">${esc(formatTime(now))}</div>
      <div class="nt-clock-date" id="clock-date">${esc(formatDate(now))}</div>
    </div>

    <div class="nt-logo">
      <div class="nt-logo-dot"></div>
      <span class="nt-logo-text">AURA</span>
    </div>

    <form class="nt-search-wrap" id="search-form">
      <input class="nt-search" id="search-input" type="text" placeholder="Search Google..." autofocus />
      <svg class="nt-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
    </form>

    <div class="nt-actions">
      <button class="nt-action" data-panel="chat">
        <span class="nt-action-icon">\u{1F4AC}</span>Chat
      </button>
      <button class="nt-action" data-panel="search">
        <span class="nt-action-icon">\u{1F50D}</span>Search
      </button>
      <button class="nt-action" data-panel="translate">
        <span class="nt-action-icon">\u{1F310}</span>Translate
      </button>
      <button class="nt-action" data-panel="write">
        <span class="nt-action-icon">\u270D\uFE0F</span>Write
      </button>
    </div>

    <div class="nt-recent" id="recent-section" style="display:none">
      <div class="nt-recent-title">Recent conversations</div>
      <div class="nt-recent-list" id="recent-list"></div>
    </div>

    <div class="nt-quote">
      <div class="nt-quote-text">${esc(getDayQuote())}</div>
    </div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `;

  // ── Clock tick ──
  setInterval(() => {
    const d = new Date();
    const timeEl = document.getElementById('clock-time');
    const dateEl = document.getElementById('clock-date');
    if (timeEl) timeEl.textContent = formatTime(d);
    if (dateEl) dateEl.textContent = formatDate(d);
  }, 10000);

  // ── Search form ──
  const form = document.getElementById('search-form') as HTMLFormElement;
  const input = document.getElementById('search-input') as HTMLInputElement;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;

    if (q.startsWith('!')) {
      // Send to AURA chat via sidebar
      sendToBackground({ type: 'OPEN_SIDEBAR', panel: 'chat', message: q.slice(1).trim() });
    } else {
      window.location.href = `https://www.google.com/search?q=${encodeURIComponent(q)}`;
    }
  });

  // ── Quick action buttons ──
  document.querySelectorAll<HTMLButtonElement>('.nt-action[data-panel]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const panel = btn.dataset.panel;
      if (panel) sendToBackground({ type: 'OPEN_SIDEBAR', panel });
    });
  });

  // ── Load recent conversations from storage (cached, no backend fetch) ──
  storageGet([CONV_LIST_KEY]).then((data) => {
    const convs: ConversationMeta[] = (data[CONV_LIST_KEY] || []).slice(0, 5);
    if (convs.length === 0) return;

    const section = document.getElementById('recent-section')!;
    const list = document.getElementById('recent-list')!;
    section.style.display = '';

    list.innerHTML = convs
      .map(
        (c) =>
          `<button class="nt-recent-item" data-conv-id="${esc(c.id)}">
            <span class="nt-recent-item-text">${esc(c.title)}</span>
            <span class="nt-recent-item-time">${esc(timeAgo(c.timestamp))}</span>
          </button>`
      )
      .join('');

    list.querySelectorAll<HTMLButtonElement>('.nt-recent-item').forEach((item) => {
      item.addEventListener('click', () => {
        const convId = item.dataset.convId;
        if (convId) sendToBackground({ type: 'OPEN_SIDEBAR', panel: 'chat', conversationId: convId });
      });
    });
  });

  // ── Backend health check (lightweight, no blocking) ──
  const ctrl = new AbortController();
  fetch(`${BACKEND_URL}/api/health`, { signal: ctrl.signal, method: 'GET' })
    .then((r) => {
      if (r.ok) {
        const statusEl = document.getElementById('status-indicator');
        if (statusEl) {
          statusEl.innerHTML = '<div class="nt-status-dot online"></div>AURA online';
        }
        // Update search placeholder
        input.placeholder = 'Search Google or type ! to ask AURA...';
      }
    })
    .catch(() => {});
  // Auto-abort after 3s to not hang
  setTimeout(() => ctrl.abort(), 3000);
}

// ── Init ──
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
