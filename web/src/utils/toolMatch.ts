/**
 * Capability search — fuzzy-rank tools against a natural-language intent.
 *
 * Pure client-side; uses simple token overlap with per-field weights. Good
 * enough for a "what are you trying to do?" assistant without spinning up
 * embeddings. Returns tools sorted by score (desc), filtered to a minimum.
 */

import type { Tool } from '../types';

const NAME_WEIGHT = 3;
const DESC_WEIGHT = 2;
const CAT_WEIGHT = 1;

const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'do', 'for', 'from',
  'has', 'i', 'in', 'is', 'it', 'me', 'my', 'of', 'on', 'or', 'our', 'some',
  'that', 'the', 'this', 'to', 'want', 'was', 'we', 'what', 'with', 'would',
  'you', 'your', 'find', 'need', 'using', 'help',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOPWORDS.has(w));
}

export interface ToolMatch {
  tool: Tool;
  score: number;
  matchedTerms: string[];
}

export function scoreTools(query: string, tools: Tool[]): ToolMatch[] {
  const q = tokenize(query);
  if (q.length === 0) return [];

  const results: ToolMatch[] = [];
  for (const tool of tools) {
    const nameTokens = tokenize(tool.name);
    const descTokens = tokenize(tool.description);
    const catTokens = tokenize(tool.category);

    let score = 0;
    const matched: string[] = [];

    for (const term of q) {
      // Exact token match
      if (nameTokens.includes(term)) { score += NAME_WEIGHT; matched.push(term); continue; }
      if (descTokens.includes(term)) { score += DESC_WEIGHT; matched.push(term); continue; }
      if (catTokens.includes(term)) { score += CAT_WEIGHT; matched.push(term); continue; }

      // Substring fallback (short queries, partial word matches)
      const nameStr = tool.name.toLowerCase();
      const descStr = tool.description.toLowerCase();
      if (nameStr.includes(term)) { score += NAME_WEIGHT * 0.5; matched.push(term); continue; }
      if (descStr.includes(term)) { score += DESC_WEIGHT * 0.5; matched.push(term); continue; }
    }

    if (score > 0) {
      results.push({ tool, score, matchedTerms: matched });
    }
  }

  return results.sort((a, b) => b.score - a.score);
}

/** Top-N matches, skipping very weak scores (< threshold * top-score). */
export function topMatches(query: string, tools: Tool[], n = 5): ToolMatch[] {
  const ranked = scoreTools(query, tools);
  if (ranked.length === 0) return [];
  const topScore = ranked[0].score;
  return ranked.filter((m) => m.score >= topScore * 0.35).slice(0, n);
}
