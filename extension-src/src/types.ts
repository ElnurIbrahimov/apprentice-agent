export interface Message {
  id: string;
  role: 'user' | 'ai';
  text: string;
  timestamp: number;
}

export interface StreamState {
  type: 'chat' | 'translate' | 'write' | 'search';
  rawText: string;
  onFirstChunk?: (() => void) | null;
  onDone?: ((rawText: string) => void) | null;
}

export interface Context {
  text: string;
  title?: string;
  url?: string;
  action?: string;
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
  | 'ocr'
  | 'youtube'
  | 'research'
  | 'math'
  | 'artifacts'
  | 'image'
  | 'compare'
  | 'agent'
  | 'models';

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
  { key: 'agent', label: 'Browser Agent', icon: '🤖', desc: 'Page action planning' },
  { key: 'summary', label: 'Page Summary', icon: '📋', desc: 'One-click page summarization' },
  { key: 'youtube', label: 'YouTube', icon: '▶️', desc: 'Summarize YouTube videos' },
  { key: 'research', label: 'Deep Research', icon: '🔬', desc: 'Multi-source web research' },
  { key: 'math', label: 'Math Solver', icon: '➗', desc: 'Step-by-step math solving' },
  { key: 'artifacts', label: 'Artifacts', icon: '⌨️', desc: 'Generate runnable code/HTML/SVG' },
];
