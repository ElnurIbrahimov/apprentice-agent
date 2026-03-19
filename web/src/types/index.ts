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
  model_used?: string | null;
  attachments?: FileAttachment[];
  citations?: Citation[];
  toolTrace?: ToolTrace[];
  compareResults?: ModelResult[];
  memoriesUsed?: string[];  // Memory snippets that influenced this response
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

export interface WebSocketMessage {
  type: 'chat' | 'chunk' | 'done' | 'error' | 'ping' | 'pong' | 'stopped' | 'proactive' | 'tool_status' | 'citations' | 'tool_trace';
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
export type TabId = 'chat' | 'monitoring' | 'tools' | 'advanced' | 'activity';

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
