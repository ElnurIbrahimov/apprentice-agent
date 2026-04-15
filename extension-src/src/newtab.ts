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

/**
 * HTML entity escape covering all 5 characters that matter in both text-node
 * AND attribute contexts. The prior implementation used textContent→innerHTML
 * which does not escape " or ' (irrelevant in text nodes, critical in
 * quoted attribute values). All server-originated strings interpolated into
 * template literals MUST go through this.
 */
function esc(s: string): string {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Allowlist-only URL check for anywhere we load or navigate to a server-provided URL.
 * Rejects javascript:, data:, file:, vbscript:, and anything with control chars or spaces.
 */
function isSafeHttpUrl(s: unknown): s is string {
  if (typeof s !== 'string' || !s) return false;
  if (/[\x00-\x1f\s]/.test(s)) return false;
  return /^https?:\/\/[^\s]+$/i.test(s);
}

/**
 * Allowlist-only color check for anywhere a server value is interpolated into
 * a CSS style attribute. Even quote-safe values can inject via
 * `background-image:url(...)` — the only safe approach is a whitelist.
 */
const COLOR_RE = /^(#[0-9a-fA-F]{3,8}|rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*(?:\d*\.)?\d+\s*)?\))$/;
function isSafeColor(s: unknown): s is string {
  return typeof s === 'string' && COLOR_RE.test(s);
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

    <div class="nt-cockpit nt-fade nt-fade-d5" id="cockpit" style="display:none">
      <div class="nt-card nt-card-span2" id="card-ticker">
        <div class="nt-card-title">💭 Aura is thinking</div>
        <div class="nt-card-body"><div class="nt-ticker" id="ticker-text">—</div><div class="nt-ticker-meta" id="ticker-meta"></div></div>
      </div>
      <div class="nt-card nt-card-span2" id="card-heatmap">
        <div class="nt-card-title">🔥 Context focus</div>
        <div class="nt-card-body"><div class="nt-heatmap" id="heatmap-chips"><span class="nt-card-empty">No active topics</span></div></div>
      </div>
      <div class="nt-card" id="card-starter">
        <div class="nt-card-title">✨ Starter</div>
        <div class="nt-card-body" id="starter-body"><span class="nt-card-empty">Quiet.</span></div>
      </div>
      <div class="nt-card" id="card-hands">
        <div class="nt-card-title">🤖 Hands</div>
        <div class="nt-card-body" id="hands-body"><span class="nt-card-empty">No hands running</span></div>
      </div>
      <div class="nt-card" id="card-activity">
        <div class="nt-card-title">📊 Activity</div>
        <div class="nt-card-body"><div class="nt-list" id="activity-list"><span class="nt-card-empty">No events</span></div></div>
      </div>
      <div class="nt-card" id="card-memories">
        <div class="nt-card-title">🧠 Memories</div>
        <div class="nt-card-body"><div class="nt-list" id="memories-list"><span class="nt-card-empty">No memories</span></div></div>
      </div>
      <div class="nt-card nt-card-span2" id="card-feed">
        <div class="nt-card-title">📎 Recent captures</div>
        <div class="nt-card-body"><div class="nt-feed" id="feed-row"><span class="nt-card-empty">No captures</span></div></div>
      </div>
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

  // ── Clock tick ──
  // Time updates every second (colon blinks); greeting + date only need to
  // refresh on the minute boundary. Splitting the two avoids replacing the
  // weather span's innerHTML every second — previously the weather node was
  // re-serialized 60× per minute.
  setInterval(() => {
    const d = new Date();
    const timeEl = document.getElementById('clock-time');
    if (timeEl) timeEl.innerHTML = formatTime(d);
  }, 1000);
  setInterval(() => {
    const d = new Date();
    const greetEl = document.getElementById('clock-greeting');
    const dateEl = document.getElementById('clock-date');
    if (greetEl) greetEl.textContent = getGreeting(d);
    if (dateEl) {
      // Preserve the existing weather span instead of re-creating it.
      const weatherEl = document.getElementById('weather');
      const weatherHTML = weatherEl ? weatherEl.outerHTML : '<span class="nt-weather" id="weather"></span>';
      dateEl.innerHTML = esc(formatDate(d)) + weatherHTML;
    }
  }, 60_000);

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

        // Mount cockpit dashboard (gated on backend reachable + user opt-in)
        storageGet(['cockpitDisabled']).then((d) => {
          if (d?.cockpitDisabled) return;
          mountCockpit();
        });
      }
    })
    .catch(() => {});
  setTimeout(() => ctrl.abort(), 3000);
}

// ═════════════════════════════════════════════════════════════════════════
// Cockpit dashboard — live cards polled from the backend.
// ═════════════════════════════════════════════════════════════════════════

async function authHeaders(): Promise<Record<string, string>> {
  const d = await storageGet(['apiKey']);
  const key = d?.apiKey?.trim?.() || '';
  return key ? { 'X-API-Key': key } : {};
}

async function fetchJson<T>(path: string, timeoutMs = 4000): Promise<T | null> {
  try {
    const headers = await authHeaders();
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    const r = await fetch(`${BACKEND_URL}${path}`, { headers, signal: ctrl.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

function fmtTimeShort(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Coerce ISO string or epoch-ms/epoch-s into unix seconds. */
function normalizeTimestamp(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value > 1e12 ? Math.floor(value / 1000) : value;
  }
  if (typeof value === 'string' && value) {
    const ms = Date.parse(value);
    if (!Number.isNaN(ms)) return Math.floor(ms / 1000);
  }
  return 0;
}

function openSidebar(panel: string, extras: Record<string, unknown> = {}) {
  sendToBackground({ type: 'OPEN_SIDEBAR', panel, ...extras });
}

/**
 * Cockpit polling — all 7 intervals are tracked so we can pause them when
 * the new-tab page is not visible. Without this, a backgrounded newtab keeps
 * firing 7 intervals against Hetzner forever, which adds up fast across many
 * open tabs.
 */
let cockpitIntervalHandles: number[] = [];
let cockpitMounted = false;

function startCockpitPolling(): void {
  if (cockpitIntervalHandles.length > 0) return; // already running
  cockpitIntervalHandles = [
    window.setInterval(refreshTicker,   8000),
    window.setInterval(refreshHeatmap, 30000),
    window.setInterval(refreshStarter, 20000),
    window.setInterval(refreshHands,   15000),
    window.setInterval(refreshActivity, 60000),
    window.setInterval(refreshMemories, 60000),
    window.setInterval(refreshFeed,     60000),
  ];
}

function stopCockpitPolling(): void {
  for (const id of cockpitIntervalHandles) clearInterval(id);
  cockpitIntervalHandles = [];
}

function refreshCockpitCards(): void {
  refreshTicker();
  refreshHeatmap();
  refreshStarter();
  refreshHands();
  refreshActivity();
  refreshMemories();
  refreshFeed();
}

function mountCockpit(): void {
  const cockpit = document.getElementById('cockpit');
  if (!cockpit) return;
  cockpit.style.display = '';
  cockpitMounted = true;

  refreshCockpitCards();
  startCockpitPolling();

  // Pause polling when the tab is hidden. Chrome already throttles intervals
  // on hidden tabs, but it still fires them periodically — pausing avoids
  // the network calls entirely.
  document.addEventListener('visibilitychange', () => {
    if (!cockpitMounted) return;
    if (document.hidden) {
      stopCockpitPolling();
    } else {
      // Fresh fetch on visibility-gained, then resume the intervals.
      refreshCockpitCards();
      startCockpitPolling();
    }
  });
}

async function refreshTicker(): Promise<void> {
  const data = await fetchJson<{ has_teaser: boolean; teaser?: { content: string; topics?: string[]; intensity?: number } }>('/api/thinking/teaser');
  const textEl = document.getElementById('ticker-text');
  const metaEl = document.getElementById('ticker-meta');
  if (!textEl || !metaEl) return;
  if (!data?.has_teaser || !data.teaser) {
    textEl.textContent = '—';
    metaEl.textContent = '';
    return;
  }
  textEl.textContent = data.teaser.content;
  const topics = data.teaser.topics?.slice(0, 4).join(' · ') || '';
  metaEl.textContent = topics;
  const card = document.getElementById('card-ticker');
  if (card) {
    card.onclick = () => openSidebar('multi-agent');
    card.style.cursor = 'pointer';
  }
}

async function refreshHeatmap(): Promise<void> {
  const data = await fetchJson<{ items: Array<{ name: string; color: string; opacity: number; size: number; weight: number }> }>('/api/context/heatmap');
  const host = document.getElementById('heatmap-chips');
  if (!host) return;
  if (!data?.items || data.items.length === 0) {
    host.innerHTML = '<span class="nt-card-empty">No active topics</span>';
    return;
  }
  host.innerHTML = data.items.slice(0, 12).map((it) => {
    const op = Math.max(0.35, Number(it.opacity) || 0.5);
    const fs = Math.round(9 + (Number(it.size) || 0) * 3);
    // Colors go into a `style` attribute — esc() alone isn't sufficient for CSS
    // injection (e.g. `red;background-image:url(evil)` has no HTML special chars).
    // Strict allowlist is the only safe approach.
    const safeColor = isSafeColor(it.color) ? it.color : '#7c3aed';
    return `<span class="nt-heatchip" style="background:${safeColor};opacity:${op};font-size:${fs}px">${esc(it.name)}</span>`;
  }).join('');
  const card = document.getElementById('card-heatmap');
  if (card) {
    card.onclick = () => openSidebar('context-heatmap');
    card.style.cursor = 'pointer';
  }
}

async function refreshStarter(): Promise<void> {
  const data = await fetchJson<{ has_starter: boolean; starter?: { content: string; topic?: string } }>('/api/conversation/starter/pending');
  const body = document.getElementById('starter-body');
  if (!body) return;
  if (!data?.has_starter || !data.starter) {
    body.innerHTML = '<span class="nt-card-empty">Quiet.</span>';
    return;
  }
  const content = data.starter.content;
  const topic = data.starter.topic || '';
  body.innerHTML = `
    <div class="nt-starter">${esc(content)}</div>
    <div class="nt-starter-actions">
      <button class="nt-starter-btn primary" id="starter-accept">Reply</button>
      <button class="nt-starter-btn" id="starter-dismiss">Dismiss</button>
    </div>
  `;
  document.getElementById('starter-accept')?.addEventListener('click', () => {
    openSidebar('chat', { message: content });
  });
  document.getElementById('starter-dismiss')?.addEventListener('click', async () => {
    if (topic) {
      const headers = await authHeaders();
      fetch(`${BACKEND_URL}/api/conversation/starter/dismiss?topic=${encodeURIComponent(topic)}`, {
        method: 'POST',
        headers,
      }).catch(() => {});
    }
    body.innerHTML = '<span class="nt-card-empty">Dismissed.</span>';
  });
}

async function refreshHands(): Promise<void> {
  const data = await fetchJson<{ hands: Array<{ name: string; state?: string }> }>('/api/hands');
  const body = document.getElementById('hands-body');
  if (!body) return;
  const hands = data?.hands ?? [];
  if (hands.length === 0) {
    body.innerHTML = '<span class="nt-card-empty">No hands running</span>';
    return;
  }
  const active = hands.filter((h) => h.state === 'running' || h.state === 'active' || h.state === 'paused').slice(0, 6);
  if (active.length === 0) {
    body.innerHTML = '<span class="nt-card-empty">All hands idle</span>';
    return;
  }
  body.innerHTML = active.map((h) => {
    const cls = h.state === 'running' || h.state === 'active' ? 'running' : h.state === 'paused' ? 'paused' : 'idle';
    return `<div class="nt-hand"><span class="nt-hand-dot ${cls}"></span>${esc(h.name)}</div>`;
  }).join('');
  const card = document.getElementById('card-hands');
  if (card) {
    card.onclick = () => openSidebar('hands');
    card.style.cursor = 'pointer';
  }
}

async function refreshActivity(): Promise<void> {
  const since = Math.floor(Date.now() / 1000) - 86400;
  // Server field names: event_type/summary/payload/timestamp(ISO).
  // Normalize here because newtab.ts bypasses api/client.ts.
  const data = await fetchJson<{ events: Array<any> }>(`/api/activity/events?limit=15&after=${since}`);
  const list = document.getElementById('activity-list');
  if (!list) return;
  const rawEvents = Array.isArray(data?.events) ? data!.events : [];
  const events = rawEvents.map((e) => ({
    timestamp: normalizeTimestamp(e?.timestamp),
    title: e?.title ?? e?.summary ?? e?.description ?? '',
    url: e?.url ?? '',
  }));
  if (events.length === 0) {
    list.innerHTML = '<span class="nt-card-empty">No events</span>';
    return;
  }
  list.innerHTML = events.slice(0, 10).map((e) => {
    const label = e.title || '(event)';
    // Only carry the URL through data-url if it's a real http(s) link.
    // The click handler also revalidates before calling window.open().
    const safeUrl = isSafeHttpUrl(e.url) ? e.url : '';
    return `<div class="nt-li" data-url="${esc(safeUrl)}">
      <span class="nt-li-time">${fmtTimeShort(e.timestamp)}</span>
      <span class="nt-li-text">${esc(label)}</span>
    </div>`;
  }).join('');
  list.querySelectorAll<HTMLElement>('.nt-li').forEach((el) => {
    const url = el.dataset.url;
    if (url && isSafeHttpUrl(url)) {
      el.onclick = () => window.open(url, '_blank', 'noopener');
    }
  });
}

async function refreshMemories(): Promise<void> {
  // Server returns ISO-string timestamps; fetchJson itself is untyped here so
  // we just pull content and ignore timestamp (the newtab memories card
  // doesn't render it).
  const data = await fetchJson<{ memories: Array<any> }>('/api/memory/recent?limit=8');
  const list = document.getElementById('memories-list');
  if (!list) return;
  const mems = Array.isArray(data?.memories) ? data!.memories : [];
  if (mems.length === 0) {
    list.innerHTML = '<span class="nt-card-empty">No memories</span>';
    return;
  }
  list.innerHTML = mems.slice(0, 8).map((m) => {
    const snippet = String(m?.content ?? '').slice(0, 80);
    return `<div class="nt-li"><span class="nt-li-text">${esc(snippet)}</span></div>`;
  }).join('');
  const card = document.getElementById('card-memories');
  if (card) {
    card.onclick = () => openSidebar('memory-browser');
    card.style.cursor = 'pointer';
  }
}

async function refreshFeed(): Promise<void> {
  const data = await fetchJson<{ items: Array<{ id: string; thumbnail?: string; title?: string }> }>('/api/feed/list?limit=6&offset=0');
  const row = document.getElementById('feed-row');
  if (!row) return;
  const items = data?.items ?? [];
  if (items.length === 0) {
    row.innerHTML = '<span class="nt-card-empty">No captures</span>';
    return;
  }
  row.innerHTML = items.map((it) => {
    const title = it.title || 'Capture';
    // Skip the <img> entirely unless the thumbnail is a real http(s) URL.
    // A `javascript:` or `data:` URL here could fire network requests from
    // the new-tab page, which has access to the extension's API key in storage.
    const thumb = isSafeHttpUrl(it.thumbnail) ? `<img src="${esc(it.thumbnail)}" alt="">` : '';
    return `<div class="nt-feed-item" title="${esc(title)}" data-id="${esc(it.id)}">
      ${thumb}
    </div>`;
  }).join('');
  row.querySelectorAll<HTMLElement>('.nt-feed-item').forEach((el) => {
    el.onclick = () => openSidebar('feed');
  });
}

// ── Init ──
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}
