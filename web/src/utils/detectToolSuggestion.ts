import type { ToolId } from '../components/ToolLauncher';

interface ToolSuggestion {
  toolId: ToolId;
  label: string;
  reason: string;
}

const RULES: Array<{ pattern: RegExp; suggestion: ToolSuggestion }> = [
  { pattern: /\b(write code|python|javascript|function|algorithm|debug|implement|coding)\b/i,
    suggestion: { toolId: 'ask', label: 'Code Interpreter', reason: 'Coding task detected' } },
  { pattern: /\b(search for|find me|look up|what is the latest|current news|google)\b/i,
    suggestion: { toolId: 'search', label: 'Search', reason: 'Search query detected' } },
  { pattern: /\b(research|deep dive|analyze|literature review|comprehensive analysis)\b/i,
    suggestion: { toolId: 'research', label: 'Research Panel', reason: 'Research task detected' } },
  { pattern: /\b(compare|vs|versus|difference between|which is better)\b/i,
    suggestion: { toolId: 'compare', label: 'Compare Models', reason: 'Comparison detected' } },
  { pattern: /\b(summarize|summarise|tldr|key points|condense|shorten)\b/i,
    suggestion: { toolId: 'summary', label: 'Summary Tool', reason: 'Summary task detected' } },
  { pattern: /\b(translate|in spanish|in french|in german|in japanese|in chinese)\b/i,
    suggestion: { toolId: 'translate', label: 'Translate', reason: 'Translation detected' } },
  { pattern: /\b(write|draft|essay|email|blog post|article|report|compose)\b/i,
    suggestion: { toolId: 'write', label: 'Write Tool', reason: 'Writing task detected' } },
  { pattern: /\b(grammar|proofread|check spelling|fix my writing)\b/i,
    suggestion: { toolId: 'grammar', label: 'Grammar Check', reason: 'Grammar task detected' } },
  { pattern: /\b(solve|equation|integral|derivative|calculate|math|formula)\b/i,
    suggestion: { toolId: 'math', label: 'Math Solver', reason: 'Math problem detected' } },
  { pattern: /\b(pdf|extract from document|analyze document)\b/i,
    suggestion: { toolId: 'pdf', label: 'PDF Analyzer', reason: 'Document task detected' } },
  { pattern: /\b(ocr|extract text|image text|screenshot text)\b/i,
    suggestion: { toolId: 'ocr', label: 'OCR Tool', reason: 'Text extraction detected' } },
  { pattern: /\b(youtube|video|transcript|this video)\b/i,
    suggestion: { toolId: 'youtube', label: 'YouTube Analyzer', reason: 'Video task detected' } },
];

export function detectToolSuggestion(message: string): ToolSuggestion | null {
  if (message.length < 10) return null;
  const text = message.toLowerCase();
  for (const rule of RULES) {
    if (rule.pattern.test(text)) return rule.suggestion;
  }
  return null;
}
