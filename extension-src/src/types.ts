export type ThinkingLevel = 'low' | 'medium' | 'high';

export interface ConversationMeta {
  id: string;
  title: string;
  timestamp: number;
  messageCount: number;
  folder?: string;
  pinned?: boolean;
}

export interface FileAttachment {
  id: string;
  name: string;
  type: 'image' | 'pdf' | 'text' | 'code';
  mimeType: string;
  /** base64 data for images (NOT persisted to history) */
  data?: string;
  /** text content for text/code files */
  textContent?: string;
  /** file size in bytes */
  size: number;
}

// ─── Hands (autonomous background agents) ─────────────────────────────────

export interface HandStats {
  name: string;
  description: string;
  state: string; // inactive | active | running | paused | cooldown | error
  total_runs: number;
  total_tokens: number;
  total_cost: number;
  consecutive_failures: number;
  last_run: string | null;
  last_run_ts: number;
  last_error: string | null;
  model_preference: string;
  idle_only: boolean;
  trigger_on_drive: string | null;
  is_custom?: boolean;
  goal?: string;
  adaptive_multiplier?: number;
}

export interface HandTemplate {
  name: string;
  description: string;
  goal: string;
  interval_minutes: number;
  trigger_on_drive: string | null;
  search_queries?: string[];
}

export interface HandHistoryEntry {
  timestamp: string;
  action_type: string;
  action_data: Record<string, unknown>;
  agent_id: string;
}

export interface HandApprovalRequest {
  request_id: string;
  hand_name: string;
  tool_name: string;
  args: Record<string, unknown>;
  timestamp: number | string;
  age_seconds?: number;
}

export interface McpServerInfo {
  name: string;
  transport: 'stdio' | 'http';
  enabled: boolean;
  connected: boolean;
  tool_count: number;
  tools: string[];
  error?: string;
}

export interface McpServerCreate {
  name: string;
  transport: 'stdio' | 'http';
  url?: string;
  command?: string[];
  env?: Record<string, string>;
  headers?: Record<string, string>;
  enabled?: boolean;
}

export interface HandLiveTrace {
  hand: string;
  step: number;
  description: string;
  timestamp: number;
}

export interface AgentStep {
  stepNum: number;
  action: 'click' | 'type' | 'scroll' | 'navigate' | 'done';
  selector?: string;
  description: string;
  result: 'ok' | 'error';
  error?: string;
  timestamp: number;
}

export interface Message {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
  thinkingContent?: string;
  /** Present when this assistant message represents an agent run. */
  agentSteps?: AgentStep[];
  /** The originating task text for an agent run. */
  agentTask?: string;
  /** False while the agent loop is running, true when finished or stopped. */
  agentDone?: boolean;
}

export interface StreamState {
  type: 'chat' | 'translate' | 'search' | 'grammar' | 'summary' | 'agent';
  rawText: string;
  thinkingText?: string;
  isThinkingPhase?: boolean;
  thinkingStartTime?: number;
  onFirstChunk?: (() => void) | null;
  onDone?: ((rawText: string, thinkingContent?: string) => void) | null;
  /** For agent-type streams: the message ID that accumulates agentSteps. */
  agentMessageId?: string;
  /** For agent-type streams: abort controller for the in-flight fetch. */
  agentAbortController?: AbortController | null;
}

export interface Context {
  text: string;
  title?: string;
  url?: string;
  action?: string;
}

export interface ProactiveMessage {
  id: string;
  text: string;
  /** ISO timestamp or unix ms — when the message was generated */
  timestamp: number;
}

export interface SendOpts {
  override?: string;
  modelKey?: string;
}

export type PanelId =
  | 'chat'
  | 'search'
  | 'translate'
  | 'grammar'
  | 'wisebase'
  | 'ask'
  | 'summary'
  | 'tools'
  | 'pdf'
  | 'voice'
  | 'record'
  | 'ocr'
  | 'youtube'
  | 'research'
  | 'math'
  | 'artifacts'
  | 'image'
  | 'compare'
  | 'capture'
  | 'agent'
  | 'hands'
  | 'aura-status'
  | 'models'
  | 'mcp'
  | 'settings'
  // SOTA upgrade — new panels wired to recent backend routes
  | 'reasoning-tree'
  | 'multi-agent'
  | 'bandit'
  | 'context-heatmap'
  | 'memory-browser'
  | 'activity'
  | 'evolution'
  | 'calendar'
  | 'flashcards'
  | 'email'
  | 'feed'
  | 'share';

export interface FeatureDef {
  key: string;
  label: string;
  icon: string;
  desc: string;
}

export const FEATURE_DEFS: FeatureDef[] = [
  { key: 'chat', label: 'Chat', icon: '💬', desc: 'Main conversation' },
  { key: 'search', label: 'Search', icon: '🔍', desc: 'Web search answer' },
  { key: 'translate', label: 'Translate', icon: '🌐', desc: 'Language translation' },
  { key: 'grammar', label: 'Grammar', icon: '✅', desc: 'Grammar & style check' },
  { key: 'ask', label: 'Ask / Explain', icon: '⚡', desc: 'Quick-action context prompts' },
  { key: 'pdf', label: 'PDF Chat', icon: '📄', desc: 'Chat with PDF content' },
  { key: 'voice', label: 'Voice Notes', icon: '🎤', desc: 'Transcript summarization' },
  { key: 'record', label: 'REC Note', icon: '🔴', desc: 'Tab/mic recording with AI transcription' },
  { key: 'agent', label: 'Browser Agent', icon: '🤖', desc: 'Page action planning' },
  { key: 'summary', label: 'Page Summary', icon: '📋', desc: 'One-click page summarization' },
  { key: 'youtube', label: 'YouTube', icon: '▶️', desc: 'Summarize YouTube videos' },
  { key: 'research', label: 'Deep Research', icon: '🔬', desc: 'Multi-source web research' },
  { key: 'math', label: 'Math Solver', icon: '➗', desc: 'Step-by-step math solving' },
  { key: 'artifacts', label: 'Artifacts', icon: '⌨️', desc: 'Generate runnable code/HTML/SVG' },
  { key: 'wisebase', label: 'Wisebase', icon: '📚', desc: 'Highlights & knowledge base' },
  { key: 'tools', label: 'Tools', icon: '🔧', desc: 'Utility tools & actions' },
  { key: 'ocr', label: 'OCR', icon: '👁️', desc: 'Extract text from images' },
  { key: 'image', label: 'Image', icon: '🖼️', desc: 'Image generation & editing' },
  { key: 'compare', label: 'Compare', icon: '⚖️', desc: 'Compare model responses' },
  { key: 'capture', label: 'Capture', icon: '🎯', desc: 'Capture & recreate UI components' },
  { key: 'models', label: 'Models', icon: '🧠', desc: 'Model selection & management' },
];
