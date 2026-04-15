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
  get: (): Promise<ThinkingModeState> =>
    apiFetch(url('/api/thinking-mode/state')),

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

  params: (): Promise<SelfImprovementParams> =>
    apiFetch(url('/api/self-improvement/params')),

  cycle: (): Promise<{ status: string; message: string; result: Record<string, unknown> | null }> =>
    apiFetch(url('/api/self-improvement/cycle'), { method: 'POST' }),

  tune: (name: string, value: number): Promise<{ status: string; success: boolean; error?: string }> =>
    apiFetch(url('/api/self-improvement/tune'), jsonBody({ name, value })),
};

// ─── context awareness ─────────────────────────────────────────────────────

export const context = {
  focus: (limit = 15): Promise<FocusResponse> =>
    apiFetch(url(`/api/context/focus?limit=${limit}`)),

  heatmap: (): Promise<HeatmapResponse> =>
    apiFetch(url('/api/context/heatmap')),

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

export const memory = {
  recent: (limit = 20): Promise<MemoryListResponse> =>
    apiFetch(url(`/api/memory/recent?limit=${limit}`)),

  search: (q: string): Promise<MemorySearchResponse> =>
    apiFetch(url(`/api/memory/search?q=${encodeURIComponent(q)}`)),

  browse: (limit = 50): Promise<MemoryListResponse> =>
    apiFetch(url(`/api/memory/browse?limit=${limit}`)).catch(() => ({ memories: [] })),

  item: (id: string): Promise<MemoryItem> =>
    apiFetch(url(`/api/memory/item/${encodeURIComponent(id)}`)),

  remove: (id: string): Promise<{ success: boolean }> =>
    apiFetch(url(`/api/memory/item/${encodeURIComponent(id)}`), { method: 'DELETE' }),

  recalls: {
    recent: (limit = 20, session_id = 'default'): Promise<{ count: number; events: MemoryRecallEvent[] }> =>
      apiFetch(url(`/api/memory/recalls/recent?limit=${limit}&session_id=${session_id}`)),

    stats: (session_id = 'default'): Promise<MemoryRecallStats> =>
      apiFetch(url(`/api/memory/recalls/stats?session_id=${session_id}`)),

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
  events: (opts: { limit?: number; after?: number; before?: number; categories?: string[] } = {}): Promise<ActivityResponse> => {
    const params = new URLSearchParams();
    if (opts.limit) params.set('limit', String(opts.limit));
    if (opts.after) params.set('after', String(opts.after));
    if (opts.before) params.set('before', String(opts.before));
    if (opts.categories?.length) params.set('categories', opts.categories.join(','));
    const qs = params.toString();
    return apiFetch(url(`/api/activity/events${qs ? '?' + qs : ''}`));
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
  alma: (): Promise<AlmaState> =>
    apiFetch(url('/api/alma/state')),

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
