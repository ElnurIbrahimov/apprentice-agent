// Type definitions for AURA Web UI

export interface MoodState {
  emotion: string | null;
  confidence: number;
  valence: number;
  arousal: number;
  session_dominant?: string;
  readings?: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
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
  type: 'chat' | 'chunk' | 'done' | 'error' | 'ping' | 'pong';
  content?: string;
  message?: string;
  response?: string;
  mood?: MoodState;
  error?: string;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

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
}

// Metacognition
export interface MetacognitionStats {
  total_actions: number;
  success_rate: number;
  avg_confidence: number;
  tool_usage: Record<string, number>;
}

// Tab types
export type TabId = 'chat' | 'monitoring' | 'tools' | 'advanced';
