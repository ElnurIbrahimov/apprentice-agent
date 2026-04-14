/**
 * AURA New Tab — Vanilla JS (no React, no heavy deps)
 * Keeps bundle under 50KB.
 */

const CONV_LIST_KEY = 'aura_conversations';
const BACKEND_URL = 'https://aura-elnur.duckdns.org';
const MOOD_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

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

/* ── Mood → Color mapping (ALMA emotions) ── */

interface MoodColors {
  core: string;
  glow: string;
  highlight: string;
}

const MOOD_COLOR_MAP: Record<string, MoodColors> = {
  neutral:    { core: '#7c3aed', glow: '#a78bfa', highlight: '#e0d6ff' },
  calm:       { core: '#7c3aed', glow: '#a78bfa', highlight: '#e0d6ff' },
  content:    { core: '#7c3aed', glow: '#a78bfa', highlight: '#e0d6ff' },
  thoughtful: { core: '#7c3aed', glow: '#a78bfa', highlight: '#e0d6ff' },
  happy:         { core: '#f59e0b', glow: '#fbbf24', highlight: '#fef3c7' },
  excited:       { core: '#f59e0b', glow: '#fbbf24', highlight: '#fef3c7' },
  playful:       { core: '#f59e0b', glow: '#fbbf24', highlight: '#fef3c7' },
  gratification: { core: '#f59e0b', glow: '#fbbf24', highlight: '#fef3c7' },
  joy:           { core: '#f59e0b', glow: '#fbbf24', highlight: '#fef3c7' },
  curious:    { core: '#3b82f6', glow: '#60a5fa', highlight: '#dbeafe' },
  surprised:  { core: '#3b82f6', glow: '#60a5fa', highlight: '#dbeafe' },
  engaged:    { core: '#3b82f6', glow: '#60a5fa', highlight: '#dbeafe' },
  admiration: { core: '#3b82f6', glow: '#60a5fa', highlight: '#dbeafe' },
  confident:    { core: '#10b981', glow: '#34d399', highlight: '#d1fae5' },
  satisfaction: { core: '#10b981', glow: '#34d399', highlight: '#d1fae5' },
  pride:        { core: '#10b981', glow: '#34d399', highlight: '#d1fae5' },
  gratitude:    { core: '#10b981', glow: '#34d399', highlight: '#d1fae5' },
  sad:            { core: '#64748b', glow: '#94a3b8', highlight: '#cbd5e1' },
  distress:       { core: '#64748b', glow: '#94a3b8', highlight: '#cbd5e1' },
  disappointment: { core: '#64748b', glow: '#94a3b8', highlight: '#cbd5e1' },
  sorry_for:      { core: '#64748b', glow: '#94a3b8', highlight: '#cbd5e1' },
  remorse:        { core: '#64748b', glow: '#94a3b8', highlight: '#cbd5e1' },
  anxious:    { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  frustrated: { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  angry:      { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  fearful:    { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  fear:       { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  hate:       { core: '#ef4444', glow: '#f87171', highlight: '#fecaca' },
  empathetic: { core: '#8b5cf6', glow: '#a78bfa', highlight: '#ede9fe' },
  concerned:  { core: '#8b5cf6', glow: '#a78bfa', highlight: '#ede9fe' },
};

function getMoodColors(emotion: string): MoodColors {
  return MOOD_COLOR_MAP[emotion] || MOOD_COLOR_MAP['neutral'];
}

function applyMoodColors(colors: MoodColors): void {
  const root = document.documentElement;
  root.style.setProperty('--dot-core', colors.core);
  root.style.setProperty('--dot-glow', colors.glow);
  root.style.setProperty('--dot-highlight', colors.highlight);
}

let _moodCache: { emotion: string; ts: number } | null = null;

async function fetchAndApplyMood(): Promise<void> {
  if (_moodCache && (Date.now() - _moodCache.ts) < MOOD_CACHE_TTL) {
    applyMoodColors(getMoodColors(_moodCache.emotion));
    return;
  }
  const stored = await storageGet(['apiKey']);
  const apiKey = stored?.apiKey?.trim?.() || '';
  if (!apiKey) return;
  try {
    const headers: Record<string, string> = { 'X-API-Key': apiKey };
    const r = await fetch(`${BACKEND_URL}/api/status`, {
      signal: AbortSignal.timeout(4000),
      headers,
    });
    if (!r.ok) return;
    const data = await r.json();
    const emotion: string = data?.mood?.emotion || 'neutral';
    _moodCache = { emotion, ts: Date.now() };
    applyMoodColors(getMoodColors(emotion));
  } catch {
    // Keep default purple
  }
}

/* ── Helpers ── */

function formatTime(d: Date): string {
  const h = String(d.getHours()).padStart(2, '0');
  const m = String(d.getMinutes()).padStart(2, '0');
  return `${h}<span class="nt-clock-colon">:</span>${m}`;
}

function formatDate(d: Date): string {
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
}

function getGreeting(d: Date): string {
  const h = d.getHours();
  if (h >= 5 && h < 12) return 'Good morning';
  if (h >= 12 && h < 17) return 'Good afternoon';
  if (h >= 17 && h < 21) return 'Good evening';
  return 'Good night';
}

/* ── Weather ── */

const WEATHER_CACHE_KEY = 'aura_weather_cache';
const WEATHER_CACHE_TTL = 30 * 60 * 1000; // 30 minutes

interface WeatherData {
  temp: string;
  condition: string;
  icon: string;
  city: string;
  cachedAt: number;
}

function weatherIcon(condition: string): string {
  const lower = condition.toLowerCase();
  if (lower.includes('thunder')) return '\u26C8\uFE0F';
  if (lower.includes('snow') || lower.includes('blizzard')) return '\u2744\uFE0F';
  if (lower.includes('rain')) return '\uD83C\uDF27\uFE0F';
  if (lower.includes('drizzle')) return '\uD83C\uDF26\uFE0F';
  if (lower.includes('mist') || lower.includes('fog')) return '\uD83C\uDF2B\uFE0F';
  if (lower.includes('overcast') || lower.includes('cloudy')) return '\u2601\uFE0F';
  if (lower.includes('partly')) return '\u26C5';
  if (lower.includes('sunny') || lower.includes('clear')) return '\u2600\uFE0F';
  return '\uD83C\uDF24\uFE0F';
}

async function fetchWeather(): Promise<WeatherData | null> {
  const cached = await storageGet([WEATHER_CACHE_KEY]);
  const data = cached[WEATHER_CACHE_KEY] as WeatherData | undefined;
  if (data && Date.now() - data.cachedAt < WEATHER_CACHE_TTL) {
    return data;
  }

  try {
    const ctrl = new AbortController();
    setTimeout(() => ctrl.abort(), 5000);
    const res = await fetch('https://wttr.in/?format=j1', { signal: ctrl.signal });
    if (!res.ok) return null;
    const json = await res.json();

    const current = json.current_condition?.[0];
    const area = json.nearest_area?.[0];
    if (!current) return null;

    const weather: WeatherData = {
      temp: current.temp_C,
      condition: current.weatherDesc?.[0]?.value || '',
      icon: weatherIcon(current.weatherDesc?.[0]?.value || ''),
      city: area?.areaName?.[0]?.value || '',
      cachedAt: Date.now(),
    };

    if (ext?.storage?.local) {
      ext.storage.local.set({ [WEATHER_CACHE_KEY]: weather });
    }

    return weather;
  } catch {
    return null;
  }
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

/* ── Top Sites ── */

function loadTopSites() {
  if (!(ext as any)?.topSites?.get) return;

  (ext as any).topSites.get((sites: Array<{ title: string; url: string }>) => {
    if (!sites || sites.length === 0) return;

    const section = document.getElementById('topsites-section');
    const row = document.getElementById('topsites-row');
    if (!section || !row) return;

    section.style.display = '';

    row.innerHTML = sites.slice(0, 8).map((site) => {
      let domain: string;
      try { domain = new URL(site.url).hostname.replace('www.', ''); } catch { domain = site.url; }
      const label = domain.split('.')[0] || domain;
      return `<a class="nt-topsite" href="${esc(site.url)}" title="${esc(site.title || domain)}">
        <div class="nt-topsite-icon">
          <img src="https://www.google.com/s2/favicons?domain=${esc(domain)}&sz=32" alt="" width="20" height="20" loading="lazy" />
        </div>
        <span class="nt-topsite-label">${esc(label)}</span>
      </a>`;
    }).join('');
  });
}

/* ── SVG Icon helpers ── */

const ICONS = {
  chat: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
  search: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
  translate: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 8l6 6"></path><path d="M4 14l6-6 2-3"></path><path d="M2 5h12"></path><path d="M7 2v3"></path><path d="M22 22l-5-10-5 10"></path><path d="M14 18h6"></path></svg>',
  write: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.85 2.85 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"></path><path d="m15 5 4 4"></path></svg>',
  code: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
  research: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path><circle cx="11" cy="10" r="3"></circle><path d="m14 13-1.5-1.5"></path></svg>',
  ocr: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 2H2v5"></path><path d="M17 2h5v5"></path><path d="M7 22H2v-5"></path><path d="M17 22h5v-5"></path><line x1="5" y1="12" x2="19" y2="12"></line></svg>',
  grammar: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline><line x1="4" y1="21" x2="20" y2="21"></line></svg>',
  chatBubble: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>',
  plus: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>',
  x: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>',
};

/* ── Build UI ── */

function mount() {
  const root = document.getElementById('root')!;
  const now = new Date();

  root.innerHTML = `
    <div class="nt-clock nt-fade nt-fade-d1">
      <div class="nt-greeting" id="clock-greeting">${esc(getGreeting(now))}</div>
      <div class="nt-clock-time" id="clock-time">${formatTime(now)}</div>
      <div class="nt-clock-date" id="clock-date">${esc(formatDate(now))}<span class="nt-weather" id="weather"></span></div>
    </div>

    <div class="nt-logo nt-fade nt-fade-d2">
      <div class="nt-logo-dot"></div>
      <span class="nt-logo-text">AURA</span>
    </div>

    <form class="nt-search-wrap nt-fade nt-fade-d3" id="search-form">
      <button type="button" class="nt-mode-toggle disabled" id="mode-toggle" title="Click or press Tab to switch mode">
        <span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span>
        <span class="nt-mode-label">Google</span>
      </button>
      <input class="nt-search" id="search-input" type="text" placeholder="Search Google..." autofocus />
      <svg class="nt-search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
      <div class="nt-mode-hint" id="mode-hint">Tab to switch to AURA</div>
    </form>

    <div class="nt-actions nt-fade nt-fade-d4">
      <button class="nt-action" data-panel="chat">
        <span class="nt-action-icon">${ICONS.chat}</span>Chat
      </button>
      <button class="nt-action" data-panel="search">
        <span class="nt-action-icon">${ICONS.search}</span>Search
      </button>
      <button class="nt-action" data-panel="translate">
        <span class="nt-action-icon">${ICONS.translate}</span>Translate
      </button>
      <button class="nt-action" data-panel="write">
        <span class="nt-action-icon">${ICONS.write}</span>Write
      </button>
      <button class="nt-action" data-panel="code">
        <span class="nt-action-icon">${ICONS.code}</span>Code
      </button>
      <button class="nt-action" data-panel="research">
        <span class="nt-action-icon">${ICONS.research}</span>Research
      </button>
      <button class="nt-action" data-panel="ocr">
        <span class="nt-action-icon">${ICONS.ocr}</span>OCR
      </button>
      <button class="nt-action" data-panel="grammar">
        <span class="nt-action-icon">${ICONS.grammar}</span>Grammar
      </button>
    </div>

    <div class="nt-topsites nt-fade nt-fade-d5" id="topsites-section" style="display:none">
      <div class="nt-topsites-row" id="topsites-row"></div>
    </div>

    <div class="nt-recent nt-fade nt-fade-d6" id="recent-section">
      <div class="nt-recent-header">
        <div class="nt-recent-title">Recent conversations</div>
        <button class="nt-new-chat-btn" id="new-chat-btn">
          ${ICONS.plus} New chat
        </button>
      </div>
      <div class="nt-recent-list" id="recent-list">
        <div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>
      </div>
    </div>

    <div class="nt-quote">
      <div class="nt-quote-text">${esc(getDayQuote())}</div>
    </div>

    <div class="nt-brand">AURA</div>

    <div class="nt-status" id="status-indicator">
      <div class="nt-status-dot"></div>
      AURA offline
    </div>
  `;

  // ── Clock tick (1s for colon blink) ──
  setInterval(() => {
    const d = new Date();
    const timeEl = document.getElementById('clock-time');
    const greetEl = document.getElementById('clock-greeting');
    const dateEl = document.getElementById('clock-date');
    if (timeEl) timeEl.innerHTML = formatTime(d);
    if (greetEl) greetEl.textContent = getGreeting(d);
    // Update date every tick but preserve weather span
    if (dateEl) {
      const weatherEl = document.getElementById('weather');
      const weatherHTML = weatherEl ? weatherEl.outerHTML : '<span class="nt-weather" id="weather"></span>';
      dateEl.innerHTML = esc(formatDate(d)) + weatherHTML;
    }
  }, 1000);

  // ── Weather (non-blocking async) ──
  fetchWeather().then((w) => {
    if (!w) return;
    const weatherEl = document.getElementById('weather');
    if (weatherEl) {
      weatherEl.innerHTML = ` &middot; ${w.icon} ${esc(w.temp)}\u00B0C ${esc(w.condition)}`;
    }
  });

  // ── Search form + mode toggle ──
  const form = document.getElementById('search-form') as HTMLFormElement;
  const input = document.getElementById('search-input') as HTMLInputElement;
  const modeToggle = document.getElementById('mode-toggle') as HTMLButtonElement;
  const modeHint = document.getElementById('mode-hint') as HTMLElement;

  let searchMode: 'google' | 'aura' = 'google';
  let auraOnline = false;

  function setSearchMode(mode: 'google' | 'aura') {
    searchMode = mode;
    if (mode === 'aura') {
      modeToggle.classList.add('aura-mode');
      modeToggle.innerHTML = '<span class="nt-mode-icon"><span class="nt-mode-icon-aura"></span></span><span class="nt-mode-label">AURA</span>';
      input.placeholder = 'Ask AURA anything...';
      modeHint.textContent = 'Tab to switch to Google';
    } else {
      modeToggle.classList.remove('aura-mode');
      modeToggle.innerHTML = '<span class="nt-mode-icon"><span class="nt-mode-icon-google">G</span></span><span class="nt-mode-label">Google</span>';
      input.placeholder = 'Search Google...';
      modeHint.textContent = auraOnline ? 'Tab to switch to AURA' : 'AURA offline';
    }
    input.focus();
  }

  modeToggle.addEventListener('click', () => {
    if (!auraOnline) return;
    setSearchMode(searchMode === 'google' ? 'aura' : 'google');
  });

  // Show/hide hint on focus/blur
  input.addEventListener('focus', () => form.classList.add('focused'));
  input.addEventListener('blur', () => form.classList.remove('focused'));

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;

    if (searchMode === 'aura') {
      sendToBackground({ type: 'OPEN_SIDEBAR', panel: 'chat', message: q });
      input.value = '';
    } else {
      window.location.href = `https://www.google.com/search?q=${encodeURIComponent(q)}`;
    }
  });

  // ── Keyboard shortcuts ──
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Tab' && document.activeElement === input && auraOnline) {
      e.preventDefault();
      setSearchMode(searchMode === 'google' ? 'aura' : 'google');
    }
    if (e.key === '/' && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
    }
    if (e.key === 'Escape' && document.activeElement === input) {
      input.value = '';
      input.blur();
    }
  });

  // ── Quick action buttons ──
  document.querySelectorAll<HTMLButtonElement>('.nt-action[data-panel]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const panel = btn.dataset.panel;
      if (panel) sendToBackground({ type: 'OPEN_SIDEBAR', panel });
    });
  });

  // ── New chat button ──
  document.getElementById('new-chat-btn')?.addEventListener('click', () => {
    sendToBackground({ type: 'OPEN_SIDEBAR', panel: 'chat', newConversation: true });
  });

  // ── Load top sites ──
  loadTopSites();

  // ── Load recent conversations from storage ──
  storageGet([CONV_LIST_KEY]).then((data) => {
    const convs: ConversationMeta[] = (data[CONV_LIST_KEY] || []).slice(0, 5);
    if (convs.length === 0) return;

    const list = document.getElementById('recent-list')!;

    list.innerHTML = convs
      .map(
        (c) =>
          `<button class="nt-recent-item" data-conv-id="${esc(c.id)}">
            <span class="nt-recent-item-icon">${ICONS.chatBubble}</span>
            <span class="nt-recent-item-text">${esc(c.title)}</span>
            <span class="nt-recent-item-time">${esc(timeAgo(c.timestamp))}</span>
            <button class="nt-recent-item-delete" title="Remove">${ICONS.x}</button>
          </button>`
      )
      .join('');

    list.querySelectorAll<HTMLButtonElement>('.nt-recent-item').forEach((item) => {
      item.addEventListener('click', (e) => {
        if ((e.target as HTMLElement).closest('.nt-recent-item-delete')) return;
        const convId = item.dataset.convId;
        if (convId) sendToBackground({ type: 'OPEN_SIDEBAR', panel: 'chat', conversationId: convId });
      });
    });

    // Delete conversation handlers
    list.querySelectorAll<HTMLButtonElement>('.nt-recent-item-delete').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const item = btn.closest('.nt-recent-item') as HTMLElement;
        const convId = item?.dataset?.convId;
        if (!convId) return;

        item.remove();

        // Update storage
        storageGet([CONV_LIST_KEY]).then((d) => {
          const remaining = (d[CONV_LIST_KEY] || []).filter((c: ConversationMeta) => c.id !== convId);
          if (ext?.storage?.local) {
            ext.storage.local.set({ [CONV_LIST_KEY]: remaining });
          }
          if (remaining.length === 0) {
            list.innerHTML = '<div class="nt-recent-empty">No conversations yet. Click Chat or switch to AURA mode to start.</div>';
          }
        });
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
        // Enable the mode toggle
        auraOnline = true;
        modeToggle.classList.remove('disabled');
        modeHint.textContent = 'Tab to switch to AURA';

        // Fetch mood data for breathing dot color
        fetchAndApplyMood();
      }
    })
    .catch(() => {});
  setTimeout(() => ctrl.abort(), 3000);
}

// ── Init ──
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
