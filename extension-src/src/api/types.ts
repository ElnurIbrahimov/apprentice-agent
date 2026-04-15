/**
 * Typed response shapes mirroring Pydantic models for the ~15 new backend routes
 * that the sidebar started consuming in the SOTA upgrade.
 *
 * Kept deliberately narrow — only fields the UI actually reads are typed.
 * Anything extra is carried through `Record<string, unknown>` escape hatches.
 */

// ─── Reasoning Tree ────────────────────────────────────────────────────────

export interface ReasoningStep {
  step: number;
  thought: string;
  score?: number;
}

export interface ReasoningResponse {
  success: boolean;
  session_id?: string;
  answer?: string;
  confidence?: number;
  reasoning_steps?: ReasoningStep[];
  summary?: string;
  metadata?: Record<string, unknown>;
  error?: string;
}

export interface TreeNode {
  id: string;
  thought: string;
  score?: number;
  visits?: number;
  children?: TreeNode[];
}

export interface TreeVisualization {
  success: boolean;
  tree?: TreeNode;
  stats?: {
    total_nodes?: number;
    max_depth?: number;
    best_score?: number;
  };
  error?: string;
}

export interface ReasoningSessionListItem {
  session_id: string;
  problem: string;
  timestamp: number;
  confidence?: number;
}

// ─── Thinking (background thoughts) ────────────────────────────────────────

export interface ActiveThought {
  id: string;
  type: string;
  topic: string;
  content: string;
  intensity: number;
  timestamp: number;
}

export interface ThinkingState {
  is_thinking: boolean;
  active_thoughts: ActiveThought[];
  thought_count: number;
  primary_thought?: ActiveThought | null;
  has_new_since?: boolean;
}

export interface ThinkingTeaser {
  has_teaser: boolean;
  teaser?: {
    content: string;
    type: string;
    icon?: string;
    intensity?: number;
    topics?: string[];
    thought_id?: string;
  };
}

export interface ThinkingStats {
  total_thoughts: number;
  real_thoughts: number;
  template_thoughts: number;
  thoughts_spoken: number;
  active_thoughts: number;
  history_size: number;
}

export type ThinkingMode = 'auto' | 'system1' | 'system2';

export interface ThinkingModeState {
  success: boolean;
  mode: ThinkingMode;
  cognitive_load?: number;
  auto_switches?: number;
}

// ─── Multi-Agent ────────────────────────────────────────────────────────────

export interface SpecialistAgent {
  name: string;
  description: string;
  tools: string[];
  triggers: string[];
}

export interface MultiAgentStatus {
  enabled: boolean;
  specialists: string[];
  specialist_details: Record<string, SpecialistAgent>;
  conversation_turns: number;
}

export interface MultiAgentChatResponse {
  response: string;
  agents_used: string[];
  routing_mode: string;
  confidence?: number;
}

export interface RoutePreviewResponse {
  query: string;
  selected_agents: string[];
  mode: string;
  reasoning: string;
  confidence: number;
  all_scores?: Record<string, number>;
}

// ─── Ghost text ─────────────────────────────────────────────────────────────

export interface GhostCompletionResponse {
  continuation: string;
}

// ─── Bandit ─────────────────────────────────────────────────────────────────

export interface BanditArm {
  strategy: string;
  alpha: number;
  beta: number;
  mean_reward: number;
  total_pulls: number;
  total_reward: number;
  last_updated?: number;
}

export interface BanditState {
  categories: Record<string, BanditArm[]>;
  summary: {
    total_arms: number;
    total_outcomes: number;
    category_counts: Record<string, number>;
  };
}

// ─── Evolution (GEPA) ───────────────────────────────────────────────────────

export interface EvolutionRunResponse {
  started: boolean;
  run_id: string;
  message?: string;
}

export interface EvolutionStatusResponse {
  status: 'idle' | 'running' | 'complete' | 'error';
  run_id?: string;
  result?: Record<string, unknown>;
}

export interface EvolutionRunRequest {
  skill_ids?: string[];
  max_iterations?: number;
  dry_run?: boolean;
  reflection_model?: string;
  eval_model?: string;
  timeout_seconds?: number;
}

// ─── Self-improvement ──────────────────────────────────────────────────────

export interface SelfImprovementParams {
  status: string;
  params?: Record<string, number | string | boolean>;
  /** Optional raw descriptors (min/max/path) — populated when the backend
   *  returns full descriptor objects rather than plain scalars. Panels can
   *  consume `params` for quick scalar display and `descriptors` for sliders. */
  descriptors?: Record<string, {
    name?: string;
    path?: string;
    current_value?: number | string | boolean;
    min_value?: number;
    max_value?: number;
    description?: string;
  }>;
}

export interface SelfImprovementStatus {
  status: string;
  outcomes?: Record<string, unknown>;
  cycle_info?: Record<string, unknown>;
}

// ─── Context ────────────────────────────────────────────────────────────────

export interface FocusItem {
  name: string;
  category: string;
  intensity: number;
  weight?: number;
  activated?: number;
}

export interface FocusResponse {
  items: FocusItem[];
  by_category?: Record<string, FocusItem[]>;
  total_focus: number;
  average_intensity: number;
  active_count: number;
  category_colors?: Record<string, string>;
}

export interface HeatmapChip {
  name: string;
  category: string;
  weight: number;
  size: number;
  color: string;
  opacity: number;
}

export interface HeatmapResponse {
  items: HeatmapChip[];
  timestamp: number;
}

export interface ContextStats {
  total_activations: number;
  topics_tracked: number;
  decay_cycles: number;
  current_items: number;
  conversation_depth: number;
}

// ─── Conversation starters ─────────────────────────────────────────────────

export interface ConversationStarter {
  type: string;
  content: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
  topic?: string;
}

export interface ConversationStarterResponse {
  has_starter: boolean;
  starter?: ConversationStarter;
}

// ─── Routing ────────────────────────────────────────────────────────────────

export interface RoutingFeedbackBody {
  signal: string;
  model: string;
  conversation_id?: string;
  task_dimensions?: Record<string, unknown>;
  switched_to?: string;
}

export interface RoutingStats {
  profiles: Record<string, Record<string, unknown>>;
  total_models: number;
}

// ─── Memory ─────────────────────────────────────────────────────────────────

export interface MemoryItem {
  id: string;
  content: string;
  timestamp: number;
  source?: string;
  category?: string;
  importance?: number;
  tags?: string[];
  score?: number;
  relevance?: number;
}

export interface MemoryListResponse {
  memories: MemoryItem[];
}

export interface MemorySearchResponse {
  results: MemoryItem[];
}

export interface MemoryRecallEvent {
  timestamp: number;
  query: string;
  source: 'amem' | 'rag' | 'kg' | string;
  memories_retrieved: number;
}

export interface MemoryRecallStats {
  total_recalls: number;
  amem_recalls: number;
  rag_recalls: number;
  kg_recalls: number;
  total_memories_retrieved: number;
  last_recall?: number;
  recent_count: number;
}

// ─── Lifelog ────────────────────────────────────────────────────────────────

export interface LifelogEvent {
  url: string;
  title?: string;
  dwell_ms: number;
  scroll_max_pct?: number;
  selection?: string;
  timestamp: number;
}

export interface LifelogBatchResponse {
  stored: number;
  skipped: number;
}

export interface LifelogItem {
  url: string;
  title?: string;
  dwell_ms: number;
  ts_ms: number;
  snippet?: string;
}

export interface LifelogListResponse {
  items: LifelogItem[];
  count: number;
}

// ─── Activity ──────────────────────────────────────────────────────────────

export interface ActivityEvent {
  id?: string;
  timestamp: number;
  category?: string;
  kind?: string;
  title?: string;
  description?: string;
  url?: string;
  metadata?: Record<string, unknown>;
}

export interface ActivityResponse {
  events: ActivityEvent[];
  count: number;
}

// ─── Feed ───────────────────────────────────────────────────────────────────

export interface FeedItemSummary {
  id: string;
  type: 'component' | 'page';
  timestamp: number;
  source_url?: string;
  thumbnail?: string;
  title?: string;
}

export interface FeedListResponse {
  items: FeedItemSummary[];
  total: number;
}

// ─── Chat conversations ────────────────────────────────────────────────────

export interface ServerConversation {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface ConversationSearchHit {
  conversation_id: string;
  conversation_title: string;
  role: 'user' | 'ai';
  snippet: string;
  timestamp: number;
}

// ─── Tools (calendar, flashcards, email) ───────────────────────────────────

export interface CalendarEvent {
  id?: string;
  title: string;
  start: string;
  end?: string;
  description?: string;
  location?: string;
}

export interface Flashcard {
  id: string;
  front: string;
  back: string;
  ease_factor?: number;
  interval_days?: number;
  due_date?: string;
}

export interface FlashcardsDue {
  due_count: number;
  next_card?: Flashcard;
}

export interface EmailMessage {
  from: string;
  subject: string;
  snippet: string;
  timestamp: number;
}

// ─── Share ──────────────────────────────────────────────────────────────────

export interface ShareItem {
  id: string;
  project_name: string;
  entry_point: string;
  file_count: number;
  total_bytes: number;
  created_at: number;
  expires_at: number;
  url: string;
}

// ─── Status / ALMA (used in newtab + header) ───────────────────────────────

export interface AlmaState {
  dominant_emotion: string;
  intensity: number;
  pad: { pleasure: number; arousal: number; dominance: number };
  neuromodulators?: Record<string, number>;
  mood?: { emotion: string };
  active_emotions?: Array<{ emotion: string; intensity: number }>;
}

export interface AlmaPersonality {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}
