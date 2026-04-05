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

export interface Message {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
  thinkingContent?: string;
}

export interface StreamState {
  type: 'chat' | 'translate' | 'write' | 'search' | 'grammar' | 'summary' | 'slides' | 'compose' | 'chat-draft' | 'research-write';
  rawText: string;
  thinkingText?: string;
  isThinkingPhase?: boolean;
  thinkingStartTime?: number;
  onFirstChunk?: (() => void) | null;
  onDone?: ((rawText: string, thinkingContent?: string) => void) | null;
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
  | 'write'
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
  | 'code'
  | 'artifacts'
  | 'webcreator'
  | 'image'
  | 'compare'
  | 'capture'
  | 'agent'
  | 'slides'
  | 'models'
  | 'settings';

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
  { key: 'write', label: 'Write', icon: '✍️', desc: 'Writing assistant' },
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
  { key: 'code', label: 'Code Interpreter', icon: '🖥️', desc: 'Run Python code & analyze data' },
  { key: 'artifacts', label: 'Artifacts', icon: '⌨️', desc: 'Generate runnable code/HTML/SVG' },
  { key: 'webcreator', label: 'Web Creator', icon: '🌐', desc: 'Build websites with AI chat' },
  { key: 'wisebase', label: 'Wisebase', icon: '📚', desc: 'Highlights & knowledge base' },
  { key: 'tools', label: 'Tools', icon: '🔧', desc: 'Utility tools & actions' },
  { key: 'ocr', label: 'OCR', icon: '👁️', desc: 'Extract text from images' },
  { key: 'image', label: 'Image', icon: '🖼️', desc: 'Image generation & editing' },
  { key: 'compare', label: 'Compare', icon: '⚖️', desc: 'Compare model responses' },
  { key: 'capture', label: 'Capture', icon: '🎯', desc: 'Capture & recreate UI components' },
  { key: 'slides', label: 'Slides', icon: '📊', desc: 'AI-powered slide deck generator' },
  { key: 'models', label: 'Models', icon: '🧠', desc: 'Model selection & management' },
];
