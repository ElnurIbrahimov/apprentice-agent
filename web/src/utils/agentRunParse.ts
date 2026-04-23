/**
 * Parse a past agent conversation into structured replay data.
 *
 * Runs are stored as normal conversations with titles starting with "Agent: ".
 * The first user message typically contains the EXECUTOR_PROMPT_PREFIX +
 * a JSON plan (from buildExecutorMessage). We reverse-engineer that to get
 * the original plan back for the replay view.
 */

import type { Message, ToolTrace } from '../types';
import { type AgentPlan, parsePlan, EXECUTOR_PROMPT_PREFIX } from './agentPlan';

export interface ParsedRun {
  conversationId: string;
  title: string;
  goal: string;
  plan: AgentPlan | null;
  messages: Message[];
  toolTraces: ToolTrace[];
  elapsedMs: number;
  totalChars: number;
}

/** Fetch a conversation's messages by ID. */
export async function fetchConversationMessages(conversationId: string): Promise<Message[]> {
  const res = await fetch(
    `/api/chat/conversations/${encodeURIComponent(conversationId)}/messages`,
    { credentials: 'same-origin' },
  );
  if (!res.ok) throw new Error(`Failed to load messages (HTTP ${res.status})`);
  const data = await res.json();
  const raw = Array.isArray(data.messages) ? data.messages : [];
  // Normalize to the shape our UI expects. Server field names may differ slightly.
  return raw.map((m: Record<string, unknown>, i: number): Message => ({
    id: String(m.id ?? `mparsed_${i}`),
    role: (m.role === 'user' || m.role === 'assistant' || m.role === 'system' ? m.role : 'assistant') as Message['role'],
    content: String(m.content ?? ''),
    timestamp: typeof m.timestamp === 'number'
      ? (m.timestamp > 1e12 ? m.timestamp : m.timestamp * 1000)
      : Date.now() - (raw.length - i) * 1000,
    model_used: typeof m.model_used === 'string' ? m.model_used : null,
    toolTrace: Array.isArray(m.tool_trace)
      ? (m.tool_trace as unknown[]).map((t): ToolTrace => {
          const tr = t as Record<string, unknown>;
          return {
            event: (tr.event === 'start' || tr.event === 'done' || tr.event === 'error' ? tr.event : 'done') as ToolTrace['event'],
            tool: String(tr.tool ?? 'unknown'),
            detail: typeof tr.detail === 'string' ? tr.detail : undefined,
            elapsed_ms: typeof tr.elapsed_ms === 'number' ? tr.elapsed_ms : undefined,
            timestamp: typeof tr.timestamp === 'number' ? tr.timestamp : Date.now(),
          };
        })
      : undefined,
    citations: Array.isArray(m.citations) ? (m.citations as Message['citations']) : undefined,
  }));
}

/**
 * Extract the plan that was sent to the executor. The executor message
 * is built by `buildExecutorMessage(plan)` which prepends EXECUTOR_PROMPT_PREFIX
 * and then serializes the plan as JSON. We look for that prefix and parse
 * the JSON that follows.
 */
export function extractPlan(userMessages: Message[]): AgentPlan | null {
  for (const m of userMessages) {
    if (m.role !== 'user') continue;
    const idx = m.content.indexOf(EXECUTOR_PROMPT_PREFIX);
    if (idx < 0) continue;
    const tail = m.content.slice(idx + EXECUTOR_PROMPT_PREFIX.length);
    const plan = parsePlan(tail);
    if (plan) return plan;
  }
  // Fall back to parsing the first user message as a raw plan JSON (in case the
  // user loaded a recipe — the first message might already contain the plan).
  for (const m of userMessages) {
    if (m.role !== 'user') continue;
    const plan = parsePlan(m.content);
    if (plan) return plan;
  }
  return null;
}

/** Extract the goal from the title (strip "Agent: " prefix) or the first user message. */
export function extractGoal(title: string, messages: Message[]): string {
  const fromTitle = title.replace(/^Agent:\s*/, '').trim();
  if (fromTitle) return fromTitle;
  const firstUser = messages.find((m) => m.role === 'user');
  if (!firstUser) return '(untitled run)';
  // Strip planner/executor boilerplate if present.
  const content = firstUser.content;
  const stripped = content
    .replace(/^You are in PLANNING mode[\s\S]*?Do not emit.*$/gim, '')
    .replace(EXECUTOR_PROMPT_PREFIX, '')
    .trim();
  return stripped.slice(0, 120) || '(untitled run)';
}

/** Roll up a conversation into the replay data structure. */
export function parseRun(conversationId: string, title: string, messages: Message[]): ParsedRun {
  const plan = extractPlan(messages);
  const goal = extractGoal(title, messages);

  const toolTraces: ToolTrace[] = [];
  for (const m of messages) {
    if (m.toolTrace) toolTraces.push(...m.toolTrace);
  }
  toolTraces.sort((a, b) => a.timestamp - b.timestamp);

  const first = messages[0]?.timestamp ?? 0;
  const last = messages[messages.length - 1]?.timestamp ?? 0;
  const elapsedMs = Math.max(0, last - first);

  const totalChars = messages.reduce((sum, m) => sum + (m.content?.length ?? 0), 0);

  return {
    conversationId,
    title,
    goal,
    plan,
    messages,
    toolTraces,
    elapsedMs,
    totalChars,
  };
}
