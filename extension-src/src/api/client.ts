/**
 * Typed client for backend endpoints added after the last extension update.
 *
 * All calls route through the existing `apiFetch()` helper in `../api.ts`,
 * which handles API-key injection, timeout (30s default), and JSON parsing.
 *
 * Panels should import from this module rather than building URLs inline.
 * One file = single point to fix if backend shapes drift.
 */

import { HTTP, apiFetch, getAuthHeaders } from '../api';

// ─── normalizers ───────────────────────────────────────────────────────────
// The backend drifted away from the TypeScript types in a few places. Rather
// than changing the panels (which would push knowledge of the drift into every
// consumer), we normalize inside the client so panels see consistent shapes.

/**
 * Convert ISO-string-or-epoch-number into unix seconds. Used wherever the
 * backend inconsistently returns a timestamp as either a string or a number.
 */
function toEpochSeconds(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    // Heuristic: if it looks like ms (> year 2500 if interpreted as seconds),
    // convert; otherwise trust as seconds.
    return value > 1e12 ? Math.floor(value / 1000) : value;
  }
  if (typeof value === 'string' && value) {
    const ms = Date.parse(value);
    if (!Number.isNaN(ms)) return Math.floor(ms / 1000);
  }
  return 0;
}

/**
 * Map categorical intensity labels to a 0-1 numeric scale for anything that
 * expects a real number. Unknown strings fall to 0.5. Pass-through for numbers.
 */
const INTENSITY_LABELS: Record<string, number> = {
  fresh: 1.0,
  hot: 1.0,
  active: 0.8,
  warm: 0.7,
  fading: 0.5,
  cooling: 0.35,
  weak: 0.25,
  dormant: 0.1,
  cold: 0.05,
};
function toNumericIntensity(v: unknown): number {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const key = v.toLowerCase().trim();
    if (key in INTENSITY_LABELS) return INTENSITY_LABELS[key];
  }
  return 0.5;
}
import type {
  ReasoningResponse, TreeVisualization, ReasoningSessionListItem,
  ThinkingState, ThinkingTeaser, ThinkingStats, ThinkingMode, ThinkingModeState,
  MultiAgentStatus, MultiAgentChatResponse, RoutePreviewResponse, SpecialistAgent,
  GhostCompletionResponse, BanditState,
  EvolutionRunResponse, EvolutionStatusResponse, EvolutionRunRequest,
  SelfImprovementParams, SelfImprovementStatus,
  FocusResponse, HeatmapResponse, ContextStats,
  ConversationStarterResponse,
  RoutingFeedbackBody, RoutingStats,
  MemoryListResponse, MemorySearchResponse, MemoryItem, MemoryRecallStats, MemoryRecallEvent,
  LifelogListResponse,
  ActivityResponse,
  FeedListResponse,
  ServerConversation, ConversationSearchHit,
  CalendarEvent, FlashcardsDue, EmailMessage,
  ShareItem,
  AlmaState, AlmaPersonality,
} from './types';

// ─── helpers ────────────────────────────────────────────────────────────────

function url(path: string): string {
  return `${HTTP}${path}`;
}

function jsonBody(obj: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(obj),
  };
}

// ─── reasoning tree ─────────────────────────────────────────────────────────

export const reasoning = {
  think: (problem: string, opts: { context?: string; max_iterations?: number; max_depth?: number } = {}): Promise<ReasoningResponse> =>
    apiFetch(url('/api/reasoning-tree/think'), jsonBody({ problem, ...opts })),

  explore: (question: string, opts: { num_options?: number; context?: string } = {}): Promise<ReasoningResponse> =>
    apiFetch(url('/api/reasoning-tree/explore'), jsonBody({ question, ...opts })),

  tree: (sessionId: string): Promise<TreeVisualization> =>
    apiFetch(url(`/api/reasoning-tree/tree/${encodeURIComponent(sessionId)}`)),

  path: (sessionId: string): Promise<{ success: boolean; path: unknown[] }> =>
    apiFetch(url(`/api/reasoning-tree/path/${encodeURIComponent(sessionId)}`)),

  reflections: (sessionId: string): Promise<{ success: boolean; reflections: Record<string, unknown> }> =>
    apiFetch(url(`/api/reasoning-tree/reflections/${encodeURIComponent(sessionId)}`)),

  status: (): Promise<{ success: boolean; enabled: boolean }> =>
    apiFetch(url('/api/reasoning-tree/status')),

  sessions: (): Promise<{ success: boolean; sessions: ReasoningSessionListItem[]; total: number }> =>
    apiFetch(url('/api/reasoning-tree/sessions')),
};

// ─── thinking mode (system 1 / system 2) ───────────────────────────────────

export const thinkingMode = {
  get: async (): Promise<ThinkingModeState> => {
    const raw: any = await apiFetch(url('/api/thinking-mode/state'));
    // Backend returns cognitive_load as {load_score, window_size, suggestion};
    // TS type expects a plain number. Flatten to the score.
    let load: number | undefined;
    if (raw?.cognitive_load !== undefined) {
      if (typeof raw.cognitive_load === 'number') {
        load = raw.cognitive_load;
      } else if (typeof raw.cognitive_load === 'object' && raw.cognitive_load) {
        const n = raw.cognitive_load.load_score;
        if (typeof n === 'number') load = n;
      }
    }
    return {
      success: !!raw?.success,
      mode: raw?.mode || 'auto',
      cognitive_load: load,
      auto_switches: raw?.auto_switches,
    };
  },

  set: (mode: ThinkingMode): Promise<{ success: boolean; mode: ThinkingMode }> =>
    apiFetch(url('/api/thinking-mode/set'), jsonBody({ mode })),

  resetLoad: (): Promise<{ success: boolean; message: string }> =>
    apiFetch(url('/api/thinking-mode/reset-load'), { method: 'POST' }),
};

// ─── thinking (background thoughts) ────────────────────────────────────────

export const thinking = {
  state: (since?: number): Promise<ThinkingState> => {
    const qs = since ? `?since=${since}` : '';
    return apiFetch(url(`/api/thinking/state${qs}`));
  },

  teaser: (): Promise<ThinkingTeaser> =>
    apiFetch(url('/api/thinking/teaser')),

  generate: (force = false): Promise<{ generated: boolean; thought?: Record<string, unknown>; reason?: string }> =>
    apiFetch(url(`/api/thinking/generate?force=${force}`), { method: 'POST' }),

  add: (thought_type: string, topic: string, intensity = 0.5): Promise<{ status: string; topic: string }> =>
    apiFetch(url('/api/thinking/add'), jsonBody({ thought_type, topic, intensity })),

  resolve: (thought_id: string, resolution: 'dismissed' | 'spoke' | 'merged' = 'dismissed'): Promise<{ status: string }> =>
    apiFetch(url(`/api/thinking/resolve/${encodeURIComponent(thought_id)}?resolution=${resolution}`), { method: 'POST' }),

  stats: (): Promise<ThinkingStats> =>
    apiFetch(url('/api/thinking/stats')),

  clear: (): Promise<{ status: string }> =>
    apiFetch(url('/api/thinking/clear'), { method: 'POST' }),
};

// ─── multi-agent ────────────────────────────────────────────────────────────

export const multiAgent = {
  status: (session_id = 'default'): Promise<MultiAgentStatus> =>
    apiFetch(url(`/api/multi-agent/status?session_id=${encodeURIComponent(session_id)}`)),

  agents: (session_id = 'default'): Promise<{ agents: SpecialistAgent[]; count: number }> =>
    apiFetch(url(`/api/multi-agent/agents?session_id=${encodeURIComponent(session_id)}`)),

  chat: (message: string, context?: Record<string, unknown>, session_id = 'default'): Promise<MultiAgentChatResponse> =>
    apiFetch(
      url(`/api/multi-agent/chat?session_id=${encodeURIComponent(session_id)}`),
      jsonBody({ message, context }),
    ),

  route: (query: string): Promise<RoutePreviewResponse> =>
    apiFetch(url('/api/multi-agent/route'), jsonBody({ query })),

  clear: (session_id = 'default'): Promise<{ status: string; session_id: string }> =>
    apiFetch(url(`/api/multi-agent/clear?session_id=${encodeURIComponent(session_id)}`), { method: 'POST' }),

  history: (session_id = 'default'): Promise<{ history: unknown[]; total_turns: number }> =>
    apiFetch(url(`/api/multi-agent/history?session_id=${encodeURIComponent(session_id)}`)),
};

// ─── ghost text ─────────────────────────────────────────────────────────────

export const ghost = {
  complete: (text: string, meta?: { url?: string; title?: string }): Promise<GhostCompletionResponse> =>
    apiFetch(url('/api/ghost/complete'), jsonBody({ text, ...(meta || {}) })),
};

// ─── bandit ─────────────────────────────────────────────────────────────────

export const bandit = {
  state: (): Promise<BanditState> =>
    apiFetch(url('/api/bandit/state')),
};

// ─── evolution + self-improvement ──────────────────────────────────────────

export const evolution = {
  run: (body: EvolutionRunRequest): Promise<EvolutionRunResponse> =>
    apiFetch(url('/api/evolution/run'), jsonBody(body)),

  status: (): Promise<EvolutionStatusResponse> =>
    apiFetch(url('/api/evolution/status')),

  preview: (body: EvolutionRunRequest): Promise<{ status: string; preview: Record<string, unknown> }> =>
    apiFetch(url('/api/evolution/preview'), jsonBody(body)),
};

export const selfImprovement = {
  status: (): Promise<SelfImprovementStatus> =>
    apiFetch(url('/api/self-improvement/status')),

  report: (): Promise<{ status: string; report: Record<string, unknown> }> =>
    apiFetch(url('/api/self-improvement/report')),

  params: async (): Promise<SelfImprovementParams> => {
    const raw: any = await apiFetch(url('/api/self-improvement/params'));
    // Server returns each param value as a full descriptor
    // {name, path, current_value, min_value, max_value, ...} rather than a
    // scalar. Flatten to current_value for the simple scalar renderer, while
    // preserving the descriptor in `descriptors` for panels that want min/max.
    const flat: Record<string, number | string | boolean> = {};
    const descriptors: Record<string, any> = {};
    if (raw?.params && typeof raw.params === 'object') {
      for (const [key, val] of Object.entries(raw.params)) {
        if (val !== null && typeof val === 'object') {
          const v: any = val;
          descriptors[key] = v;
          const cur = v.current_value ?? v.value;
          if (typeof cur === 'number' || typeof cur === 'string' || typeof cur === 'boolean') {
            flat[key] = cur;
          }
        } else if (typeof val === 'number' || typeof val === 'string' || typeof val === 'boolean') {
          flat[key] = val;
        }
      }
    }
    return {
      status: raw?.status ?? 'ok',
      params: flat,
      ...(Object.keys(descriptors).length > 0 ? { descriptors } : {}),
    } as SelfImprovementParams;
  },

  cycle: (): Promise<{ status: string; message: string; result: Record<string, unknown> | null }> =>
    apiFetch(url('/api/self-improvement/cycle'), { method: 'POST' }),

  tune: (name: string, value: number): Promise<{ status: string; success: boolean; error?: string }> =>
    apiFetch(url('/api/self-improvement/tune'), jsonBody({ name, value })),
};

// ─── context awareness ─────────────────────────────────────────────────────

export const context = {
  focus: async (limit = 15): Promise<FocusResponse> => {
    const raw: any = await apiFetch(url(`/api/context/focus?limit=${limit}`));
    // Server returns `intensity` as a categorical string ("fading", etc.);
    // normalize to 0-1 number so heatmap/progress bars work.
    const normalizeItem = (it: any) => ({
      name: it?.name ?? '',
      category: it?.category ?? '',
      intensity: toNumericIntensity(it?.intensity),
      weight: typeof it?.weight === 'number' ? it.weight : undefined,
      activated: typeof it?.activated === 'number'
        ? it.activated
        : toEpochSeconds(it?.last_updated) || undefined,
    });
    const items = Array.isArray(raw?.items) ? raw.items.map(normalizeItem) : [];
    const byCategory: Record<string, any[]> = {};
    if (raw?.by_category && typeof raw.by_category === 'object') {
      for (const [cat, arr] of Object.entries(raw.by_category)) {
        byCategory[cat] = Array.isArray(arr) ? (arr as any[]).map(normalizeItem) : [];
      }
    }
    return {
      items,
      by_category: byCategory,
      total_focus: raw?.total_focus ?? 0,
      average_intensity: typeof raw?.average_intensity === 'number' ? raw.average_intensity : 0,
      active_count: raw?.active_count ?? items.length,
      category_colors: raw?.category_colors ?? undefined,
    };
  },

  heatmap: async (): Promise<HeatmapResponse> => {
    const raw: any = await apiFetch(url('/api/context/heatmap'));
    return {
      items: Array.isArray(raw?.items) ? raw.items : [],
      timestamp: toEpochSeconds(raw?.timestamp),
    };
  },

  stats: (): Promise<ContextStats> =>
    apiFetch(url('/api/context/stats')),

  trackMessage: (message: string, is_user = true, source = 'extension'): Promise<{ status: string }> =>
    apiFetch(
      url(`/api/context/track/message?message=${encodeURIComponent(message)}&is_user=${is_user}&source=${source}`),
      { method: 'POST' },
    ),
};

// ─── conversation starters ─────────────────────────────────────────────────

export const starter = {
  pending: (): Promise<ConversationStarterResponse> =>
    apiFetch(url('/api/conversation/starter/pending')),

  check: (): Promise<{ has_pending: boolean }> =>
    apiFetch(url('/api/conversation/starter/check')),

  generate: (opts: { focus_topics?: string[]; emotion?: string; idle_seconds?: number; force?: boolean } = {}): Promise<{ generated: boolean }> =>
    apiFetch(url('/api/conversation/starter/generate'), jsonBody(opts)),

  dismiss: (topic: string): Promise<{ status: string }> =>
    apiFetch(url(`/api/conversation/starter/dismiss?topic=${encodeURIComponent(topic)}`), { method: 'POST' }),
};

// ─── routing feedback ──────────────────────────────────────────────────────

export const routing = {
  feedback: (body: RoutingFeedbackBody): Promise<{ ok: boolean; signal: string }> =>
    apiFetch(url('/api/routing/feedback'), jsonBody(body)),

  stats: (): Promise<RoutingStats> =>
    apiFetch(url('/api/routing/stats')),

  forConversation: (conversation_id: string): Promise<Record<string, unknown>> =>
    apiFetch(url(`/api/routing/conversation/${encodeURIComponent(conversation_id)}`)),
};

// ─── memory ─────────────────────────────────────────────────────────────────

/** Coerces a raw memory item to the expected TS shape, normalizing timestamps. */
function normalizeMemoryItem(m: any): MemoryItem {
  return {
    id: String(m?.id ?? ''),
    content: String(m?.content ?? ''),
    timestamp: toEpochSeconds(m?.timestamp),
    source: m?.source,
    category: m?.category,
    importance: typeof m?.importance === 'number' ? m.importance : undefined,
    tags: Array.isArray(m?.tags) ? m.tags : undefined,
    score: typeof m?.score === 'number' ? m.score : undefined,
    relevance: typeof m?.relevance === 'number' ? m.relevance : undefined,
  };
}

export const memory = {
  recent: async (limit = 20): Promise<MemoryListResponse> => {
    const raw: any = await apiFetch(url(`/api/memory/recent?limit=${limit}`));
    return { memories: Array.isArray(raw?.memories) ? raw.memories.map(normalizeMemoryItem) : [] };
  },

  search: async (q: string): Promise<MemorySearchResponse> => {
    const raw: any = await apiFetch(url(`/api/memory/search?q=${encodeURIComponent(q)}`));
    return { results: Array.isArray(raw?.results) ? raw.results.map(normalizeMemoryItem) : [] };
  },

  browse: async (limit = 50): Promise<MemoryListResponse> => {
    try {
      const raw: any = await apiFetch(url(`/api/memory/browse?limit=${limit}`));
      return { memories: Array.isArray(raw?.memories) ? raw.memories.map(normalizeMemoryItem) : [] };
    } catch {
      return { memories: [] };
    }
  },

  item: async (id: string): Promise<MemoryItem> => {
    const raw: any = await apiFetch(url(`/api/memory/item/${encodeURIComponent(id)}`));
    return normalizeMemoryItem(raw);
  },

  remove: (id: string): Promise<{ success: boolean }> =>
    apiFetch(url(`/api/memory/item/${encodeURIComponent(id)}`), { method: 'DELETE' }),

  recalls: {
    recent: async (limit = 20, session_id = 'default'): Promise<{ count: number; events: MemoryRecallEvent[] }> => {
      const raw: any = await apiFetch(url(`/api/memory/recalls/recent?limit=${limit}&session_id=${session_id}`));
      const events: MemoryRecallEvent[] = Array.isArray(raw?.events)
        ? raw.events.map((e: any) => ({
            timestamp: toEpochSeconds(e?.timestamp),
            query: String(e?.query ?? ''),
            source: e?.source ?? 'amem',
            memories_retrieved: typeof e?.memories_retrieved === 'number' ? e.memories_retrieved : 0,
          }))
        : [];
      return { count: events.length, events };
    },

    stats: async (session_id = 'default'): Promise<MemoryRecallStats> => {
      const raw: any = await apiFetch(url(`/api/memory/recalls/stats?session_id=${session_id}`));
      return {
        total_recalls: raw?.total_recalls ?? 0,
        amem_recalls: raw?.amem_recalls ?? 0,
        rag_recalls: raw?.rag_recalls ?? 0,
        kg_recalls: raw?.kg_recalls ?? 0,
        total_memories_retrieved: raw?.total_memories_retrieved ?? 0,
        last_recall: raw?.last_recall !== undefined ? toEpochSeconds(raw.last_recall) : undefined,
        recent_count: raw?.recent_count ?? 0,
      };
    },

    status: (session_id = 'default'): Promise<{ is_active: boolean; recent_count: number }> =>
      apiFetch(url(`/api/memory/recalls/status?session_id=${session_id}`)),
  },
};

// ─── lifelog (in addition to background.ts batch ingest) ───────────────────

export const lifelog = {
  recent: (limit = 20): Promise<LifelogListResponse> =>
    apiFetch(url(`/api/lifelog/recent?limit=${limit}`)),

  search: (q: string, limit = 20): Promise<LifelogListResponse> =>
    apiFetch(url(`/api/lifelog/search?q=${encodeURIComponent(q)}&limit=${limit}`)),
};

// ─── activity timeline ─────────────────────────────────────────────────────

export const activity = {
  events: async (opts: { limit?: number; after?: number; before?: number; categories?: string[] } = {}): Promise<ActivityResponse> => {
    const params = new URLSearchParams();
    if (opts.limit) params.set('limit', String(opts.limit));
    if (opts.after) params.set('after', String(opts.after));
    if (opts.before) params.set('before', String(opts.before));
    if (opts.categories?.length) params.set('categories', opts.categories.join(','));
    const qs = params.toString();
    const raw: any = await apiFetch(url(`/api/activity/events${qs ? '?' + qs : ''}`));
    // Server field names drifted: event_type/summary/payload instead of
    // kind/title/metadata. Also id is a number, not string.
    const events: any[] = Array.isArray(raw?.events)
      ? raw.events.map((e: any) => ({
          id: e?.id != null ? String(e.id) : undefined,
          timestamp: toEpochSeconds(e?.timestamp),
          category: e?.category,
          kind: e?.kind ?? e?.event_type,
          title: e?.title ?? e?.summary,
          description: e?.description ?? (typeof e?.payload === 'string' ? e.payload : undefined),
          url: e?.url,
          metadata: e?.metadata ?? (typeof e?.payload === 'object' ? e.payload : undefined),
        }))
      : [];
    return { events, count: typeof raw?.count === 'number' ? raw.count : events.length };
  },
};

// ─── feed ───────────────────────────────────────────────────────────────────

export const feed = {
  list: (limit = 50, offset = 0): Promise<FeedListResponse> =>
    apiFetch(url(`/api/feed/list?limit=${limit}&offset=${offset}`)),

  get: (id: string): Promise<Record<string, unknown>> =>
    apiFetch(url(`/api/feed/${encodeURIComponent(id)}`)),

  remove: (id: string): Promise<{ ok: boolean; id: string }> =>
    apiFetch(url(`/api/feed/${encodeURIComponent(id)}`), { method: 'DELETE' }),
};

// ─── chat conversations (server-side CRUD — replaces local IDB source) ─────

export const conversations = {
  list: (): Promise<ServerConversation[]> =>
    apiFetch(url('/api/chat/conversations')),

  create: (title?: string): Promise<ServerConversation> =>
    apiFetch(url('/api/chat/conversations'), jsonBody({ title })),

  rename: (id: string, title: string): Promise<{ success: boolean }> =>
    apiFetch(url(`/api/chat/conversations/${encodeURIComponent(id)}`), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    }),

  remove: (id: string): Promise<{ success: boolean; error?: string }> =>
    apiFetch(url(`/api/chat/conversations/${encodeURIComponent(id)}`), { method: 'DELETE' }),

  switch: (id: string): Promise<ServerConversation> =>
    apiFetch(url(`/api/chat/conversations/${encodeURIComponent(id)}/switch`), { method: 'POST' }),

  search: (q: string, limit = 30): Promise<{ results: ConversationSearchHit[]; query: string }> =>
    apiFetch(url(`/api/chat/conversations/search?q=${encodeURIComponent(q)}&limit=${limit}`)),
};

// ─── tools (calendar, flashcards, email) ───────────────────────────────────

export const tools = {
  calendar: {
    today: (): Promise<{ events: CalendarEvent[]; success: boolean }> =>
      apiFetch(url('/api/calendar/today')),

    upcoming: (days = 7): Promise<{ events: CalendarEvent[] }> =>
      apiFetch(url(`/api/calendar/upcoming?days=${days}`)),

    add: (event: Partial<CalendarEvent> & { title: string; start: string }): Promise<{ success: boolean; event_id?: string }> =>
      apiFetch(url('/api/calendar/add'), jsonBody(event)),

    remove: (event_id: string): Promise<{ success: boolean }> =>
      apiFetch(url(`/api/calendar/${encodeURIComponent(event_id)}`), { method: 'DELETE' }),
  },

  flashcards: {
    due: (): Promise<FlashcardsDue> =>
      apiFetch(url('/api/flashcards/due')),

    answer: (card_id: string, quality: number): Promise<{ success: boolean; interval?: number }> =>
      apiFetch(url('/api/flashcards/answer'), jsonBody({ card_id, quality })),

    stats: (): Promise<{ decks: unknown[]; total_cards: number; due_today: number }> =>
      apiFetch(url('/api/flashcards/stats')),
  },

  email: {
    status: (): Promise<{ configured: boolean; provider?: string; account?: string }> =>
      apiFetch(url('/api/email/status')),

    inbox: (limit = 20): Promise<{ emails: EmailMessage[] }> =>
      apiFetch(url(`/api/email/inbox?limit=${limit}`)),
  },
};

// ─── share ──────────────────────────────────────────────────────────────────

export const share = {
  create: (body: { project_name: string; files: Record<string, string>; entry_point: string; expires_days?: number }): Promise<{ url: string; id: string; expires_at: number }> =>
    apiFetch(url('/api/share'), jsonBody(body)),

  list: (): Promise<ShareItem[]> =>
    apiFetch(url('/api/shares')),

  remove: (id: string): Promise<{ deleted: boolean }> =>
    apiFetch(url(`/api/shares/${encodeURIComponent(id)}`), { method: 'DELETE' }),
};

// ─── status / ALMA (convenience wrappers for dashboard + header) ───────────

export const status = {
  alma: async (): Promise<AlmaState> => {
    const raw: any = await apiFetch(url('/api/alma/state'));
    // Server uses `mood.label`; TS expects `mood.emotion`. Backfill.
    let mood: AlmaState['mood'] = raw?.mood;
    if (mood && typeof mood === 'object' && !('emotion' in mood)) {
      const label = (mood as any).label;
      if (typeof label === 'string') {
        mood = { ...(mood as any), emotion: label };
      }
    }
    return {
      dominant_emotion: raw?.dominant_emotion ?? 'neutral',
      intensity: typeof raw?.intensity === 'number' ? raw.intensity : 0,
      pad: raw?.pad ?? { pleasure: 0, arousal: 0, dominance: 0 },
      neuromodulators: raw?.neuromodulators,
      mood,
      active_emotions: raw?.active_emotions,
    };
  },

  personality: (): Promise<AlmaPersonality> =>
    apiFetch(url('/api/alma/personality')),
};

// ─── hands (convenience import so new dashboard card doesn't dup code) ─────

export const hands = {
  list: (): Promise<{ hands: unknown[] }> =>
    apiFetch(url('/api/hands')),
};

// ─── fetch headers helper (for places that can't use apiFetch, e.g. SSE) ──

export { getAuthHeaders };
