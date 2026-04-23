// Type definitions for AURA Web UI

export interface MoodState {
  emotion: string | null;
  confidence: number;
  valence: number;      // PAD: Pleasure (-1 to 1)
  arousal: number;      // PAD: Arousal (-1 to 1)
  dominance?: number;   // PAD: Dominance (-1 to 1)
  emoji?: string;       // Mood emoji from ALMA
  session_dominant?: string;
  readings?: number;
}

// File attachment types
export type AttachmentType = 'image' | 'document' | 'code' | 'archive';

export interface FileAttachment {
  id: string;
  filename: string;
  mimeType: string;
  size: number;
  type: AttachmentType;
  preview?: string;  // Base64 data URL for images
  uploading?: boolean;
  error?: string;
  path?: string;  // Server path (after upload)
}

export interface Citation {
  id: number;
  title: string;
  url: string;
  snippet?: string;
  score?: number;
}

export interface ToolTrace {
  event: 'start' | 'done' | 'error';
  tool: string;
  detail?: string;
  elapsed_ms?: number;
  timestamp: number;
}

export interface ModelResult {
  model: string;
  response: string;
  time_ms: number;
  error?: string | null;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
  edited?: boolean;
  model_used?: string | null;
  attachments?: FileAttachment[];
  citations?: Citation[];
  toolTrace?: ToolTrace[];
  compareResults?: ModelResult[];
  memoriesUsed?: string[];  // Memory snippets that influenced this response
  actionMode?: string | null;  // e.g. 'delegate', 'research', 'compare' — preserved from composer
  proactive?: {
    action: string;       // e.g., 'notify', 'suggest', 'remind', 'ask'
    trigger?: string;     // What triggered this message
    confidence?: number;  // How confident the daemon was
  };
}

export interface ProactiveMessage {
  action: string;
  content: string;
  priority: string;
  timestamp: string;
  delivered: boolean;
  metadata: Record<string, unknown>;
}

// ─── Typed agent events (mirror of Python LoopEvent) ─────────────────────
// Emitted by Telegram/API agent threads via websocket_hub.broadcast_agent_event
// and consumed by the Mini App chat tab + web SPA for live tool-progress UI.

export type AgentEventKind =
  | 'tool_start'
  | 'tool_result'
  | 'chunk'
  | 'response'
  | 'done'
  | 'error'
  | 'run_finished';

export interface AgentEvent {
  type: 'agent_event';
  kind: AgentEventKind;
  run_id: string;
  iteration: number;
  payload: Record<string, unknown>;
  timestamp: number;
  user_id?: string | null;
  conversation_id?: string | null;
}

// ─── Rich tool-output cards for the Mini App chat ────────────────────────
// Each assistant ChatMessage can carry a list of ToolResults that render
// above the markdown bubble as interactive cards (CodeCard, ImageCard,
// ResearchCard, SearchCard, SummaryCard, GenericToolCard).

export type ToolStatus = 'running' | 'done' | 'error';

export interface ToolResultBase {
  id: string;           // unique — typically `${run_id}:${iteration}:${tool_name}`
  tool: string;         // raw tool name from the agent
  status: ToolStatus;
  iconOverride?: string;
}

export type ToolResult =
  | (ToolResultBase & {
      kind: 'code';
      language: string;
      source: string;
      output?: string;
    })
  | (ToolResultBase & {
      kind: 'image';
      imageUrl?: string;
      imageB64?: string;
      prompt: string;
    })
  | (ToolResultBase & {
      kind: 'research';
      query: string;
      report?: string;
      sources?: Citation[];
    })
  | (ToolResultBase & {
      kind: 'search';
      query: string;
      results?: { url: string; title: string; snippet?: string }[];
    })
  | (ToolResultBase & {
      kind: 'summary';
      title?: string;
      summary?: string;
      highlights?: string[];
    })
  | (ToolResultBase & {
      kind: 'generic';
      args?: Record<string, unknown>;
      result?: string;
    })
  | (ToolResultBase & {
      kind: 'document';
      doc_id: string;
      filename: string;
      chunks_count?: number;
      size_chars?: number;
      summary?: string;
      facts?: string[];
      questions?: string[];
    });

// ─── Proactive action card (from Hands + webhooks) ───────────────────────

export interface ProactiveCardAction {
  id: string;           // e.g. "ack", "more", "snooze_3600"
  label: string;        // e.g. "✓ Acknowledge", "💬 Tell me more", "⏰ Snooze 1h"
  style?: 'primary' | 'secondary' | 'danger';
}

// ─── Memory browser (Mini App Brain tab) ─────────────────────────────────

export interface MemoryItem {
  id: string;
  content: string;
  title?: string;
  source: string;
  memory_type?: string;
  importance?: number;
  tags?: string[];
  pinned?: boolean;
  category?: string;
  lifecycle_state?: string;
  access_count?: number;
  strength?: number;
  created_at: string;
  updated_at?: string;
}

export interface MemoryKgNode {
  id: string;
  label: string;
  type: string;
  confidence: number;
  access_count: number;
}

export interface ProactiveCard {
  type: 'proactive_card';
  hand_name: string;
  title: string;
  summary: string;
  severity?: 'info' | 'success' | 'warning' | 'error';
  metadata?: {
    iterations?: number;
    duration_seconds?: number;
    cost_usd?: number;
    source?: string;  // e.g. "github:ci_failure"
  };
  actions?: ProactiveCardAction[];
  timestamp?: number;
}

export interface ChatResponse {
  response: string;
  fast_path: boolean;
  mood: MoodState | null;
  model_used: string | null;
}

export interface StatusResponse {
  online: boolean;
  model: string;
  aura_enabled: boolean;
  mood: MoodState | null;
  memory_count: number;
  query_count: number;
  last_model_used: string | null;
}

// Research Progress
export type ResearchStage = 'plan' | 'search' | 'source' | 'finding' | 'synthesis';

export interface ResearchProgressStep {
  stage: ResearchStage;
  data: Record<string, unknown>;
  timestamp: number;
}

export interface ResearchProgress {
  active: boolean;
  stage: ResearchStage;
  steps: ResearchProgressStep[];
}

export interface WebSocketMessage {
  type: 'chat' | 'chunk' | 'done' | 'error' | 'ping' | 'pong' | 'stopped' | 'proactive' | 'tool_status' | 'citations' | 'tool_trace' | 'action_trace' | 'research_progress' | 'hand_event' | 'hand_approval_request' | 'conv_sync';
  content?: string;
  message?: string;
  response?: string;
  mood?: MoodState;
  model_used?: string | null;
  audio_url?: string | null;
  id?: string | number;
  error?: string;
  action?: string;
  priority?: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
  tool_name?: string;
  tool_action?: string;
  citations?: Citation[];
  event?: 'start' | 'done' | 'error';
  tool?: string;
  detail?: string;
  elapsed_ms?: number;
  // research_progress fields
  stage?: ResearchStage;
  data?: Record<string, unknown>;
  // hand_event fields
  hand?: string;
  success?: boolean;
  summary?: string;
  // hand_approval_request fields
  hand_name?: string;
  request_id?: string;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// Multi-conversation support
export interface Conversation {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
  preview: string;
  is_active: boolean;
}

// AURA ALIVE
export interface AuraStatus {
  enabled: boolean;
  mood: string;
  energy: number;
  warmth: number;
  engagement: number;
  soul_name: string;
  patterns_learned: number;
  turns: number;
}

// Thoughts / Inner Monologue
export interface Thought {
  type: string;
  content: string;
  confidence?: number;
  timestamp?: string;
}

export interface ThoughtsResponse {
  thoughts: Thought[];
  verbosity: number;
  think_aloud: boolean;
  thought_count: number;
}

// Knowledge Graph
export interface KGNode {
  id: string;
  label: string;
  type: string;
  confidence: number;
  access_count: number;
}

export interface KGEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
}

export interface KnowledgeGraphData {
  nodes: KGNode[];
  edges: KGEdge[];
  stats: {
    total_nodes?: number;
    total_edges?: number;
    clusters?: number;
    avg_confidence?: number;
  };
}

// Guardian
export interface GuardianStatus {
  enabled: boolean;
  monitoring_level: string;
  interventions: number;
  patterns_learned: number;
  session_predictions: number;
  recent_predictions: Array<{
    type: string;
    probability: number;
    action: string;
  }>;
}

// NeuroDream
export interface NeuroDreamStatus {
  enabled: boolean;
  loading?: boolean;
  is_sleeping: boolean;
  current_phase?: string;
  total_sessions: number;
  total_insights: number;
  dream_journal: Array<{
    phase: string;
    timestamp: string;
    content: string;
  }>;
  insights: Array<{
    type: string;
    content: string;
    confidence: number;
  }>;
}

// FluxMind
export interface FluxMindStatus {
  enabled: boolean;
  version: string;
  accuracy: number;
  calibration: string;
}

// Voice
export interface VoiceStatus {
  available: boolean;
  engine: string;
  sesame_loaded: boolean;
}

// Tools
export interface Tool {
  name: string;
  description: string;
  category: string;
}

export interface SessionCosts {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  cost_usd: number;
  queries: number;
}

// Metacognition
export interface MetacognitionStats {
  total_actions: number;
  success_rate: number;
  avg_confidence: number;
  tool_usage: Record<string, number>;
}

// Activity Timeline
export interface ActivityEvent {
  id: number;
  timestamp: number;
  category: 'tool' | 'memory' | 'emotion' | 'proactive' | 'strategy' | 'system';
  event_type: string;
  summary: string;
  payload?: Record<string, unknown> | null;
  duration_ms?: number | null;
}

// Fleet Dashboard
export interface FleetTask {
  id: string;
  description: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  elapsed: number;
  result?: string;
  error?: string;
}

// Tab types
export type TabId = 'chat' | 'create' | 'tools' | 'insights' | 'settings';

// ─── Consciousness Dashboard ───
export interface DriveState {
  curiosity: number;
  competence: number;
  social: number;
  coherence: number;
  [key: string]: number;
}

export interface CognitiveLoad {
  breath_rate: number;
  glow_intensity: number;
  total_load?: number;
  breakdown?: Record<string, number>;
}

export interface ToMSummary {
  emotional_state?: { valence: number; arousal: number; engagement: number; frustration: number };
  style?: { verbosity: number; formality: number; technical_depth: number };
  top_topics?: Array<{ topic: string; level: string }>;
  full_model?: string;
  style_guidance?: string;
}

// ─── Insights Feed ───
export interface CuriosityTarget {
  entity_id: string;
  label: string;
  gap_type: string;
  question?: string;
  urgency?: number;
}

export interface DriveAction {
  drive: string;
  action: string;
  description: string;
  priority: number;
}

export interface ProactiveSuggestion {
  suggestion: string;
  has_suggestion: boolean;
  beliefs?: Record<string, number>;
}

// ─── World Model ───
export interface WorldProject {
  id: string;
  label: string;
  type: string;
  confidence?: number;
  access_count?: number;
  last_accessed?: string;
}

export interface WorldGoal {
  id: string;
  label: string;
  type: string;
  confidence?: number;
  access_count?: number;
}

// A-MEM (Agentic Memory)
export interface AMEMNote {
  id: string;
  content: string;
  keywords: string[];
  tags: string[];
  context: string;
  category: string;
  importance: number;
  links: number;
  created_at: string;
}

export interface AMEMStats {
  total_notes: number;
  total_links: number;
  total_boxes: number;
  categories: Record<string, number>;
  has_embeddings: number;
  evolution_enabled: boolean;
}

export interface AMEMSearchResult {
  id: string;
  content: string;
  keywords: string[];
  tags: string[];
  context: string;
  relevance: number;
  hop: number;
}

export interface HybridResult {
  content: string;
  source: 'amem' | 'kg';
  score: number;
  id: string;
  keywords: string[];
  tags: string[];
  context: string;
  node_type: string;
  relationships: string[];
}

// ── Hands ──

export interface HandTemplate {
  name: string;
  description: string;
  goal: string;
  interval_minutes: number;
  trigger_on_drive: string | null;
  search_queries?: string[];
}

export interface HandStats {
  name: string;
  description: string;
  state: string;
  total_runs: number;
  total_tokens: number;
  total_cost: number;
  consecutive_failures: number;
  last_run: string | null;
  last_run_ts: number;
  last_error: string | null;
  model_preference: string;
  idle_only: boolean;
  trigger_on_drive: string;
  is_custom?: boolean;
  goal?: string;
  adaptive_multiplier?: number;
}

export interface HandHistoryEntry {
  timestamp: string;
  action_type: string;
  action_data: Record<string, unknown>;
  agent_id: string;
}

export interface ApprovalRequest {
  request_id: string;
  hand_name: string;
  tool_name: string;
  args: Record<string, unknown>;
  timestamp: string;
  age_seconds: number;
}
