/**
 * Agent plan schema + parser/validator.
 *
 * The plan phase asks the LLM to output a JSON object that the UI then
 * renders as editable cards. Parsing is tolerant:
 *   - Accepts fenced ```json blocks or raw JSON
 *   - Accepts mild schema drift (missing optional fields, extra fields)
 *   - Falls back to null on malformed input so the caller can show a retry UI
 */

export interface PlanStep {
  id: string;
  title: string;
  rationale: string;
  tool?: string;
  inputs?: Record<string, unknown>;
  estTokens?: number;
  /** Runtime-only status — not part of the LLM's output. */
  status?: 'pending' | 'running' | 'done' | 'failed' | 'skipped';
}

export interface AgentPlan {
  goal: string;
  steps: PlanStep[];
}

export const PLANNER_SYSTEM_PROMPT = `You are in PLANNING mode. Do NOT execute anything yet.

Output ONLY a JSON object (no prose, no markdown fences) matching this schema:

{
  "goal": "<restated goal in one sentence>",
  "steps": [
    {
      "id": "s1",
      "title": "<short verb-led title, e.g. 'Search for framework docs'>",
      "rationale": "<one-sentence why this step is needed>",
      "tool": "<optional tool name the step likely needs, e.g. 'web_search' | 'code_executor' | 'filesystem'>",
      "inputs": { "<optional structured inputs>": "..." },
      "estTokens": 500
    }
  ]
}

Rules:
- 3 to 8 steps maximum. Prefer fewer, higher-leverage steps.
- Each step must be independently verifiable (you can tell if it succeeded).
- If the goal is trivial (single tool call), plan 1-2 steps, not 8.
- Do not emit pleasantries, explanation, or markdown. JSON only.`;

export const EXECUTOR_PROMPT_PREFIX = `You have an APPROVED plan. Execute it step-by-step using tools as needed.
For each step, announce "Step <id>: <title>" before starting, then do the work, then report a short outcome. If a step requires a tool, call it. If a step fails, report the failure and either retry with a different approach or skip.

APPROVED PLAN:`;

/** Parse the LLM's plan response. Tolerant to fencing and trailing commentary. */
export function parsePlan(raw: string): AgentPlan | null {
  if (!raw) return null;
  let text = raw.trim();

  // Strip ```json ... ``` or ``` ... ``` fences if present.
  const fenceMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch) text = fenceMatch[1].trim();

  // Find the outermost { ... } block.
  const first = text.indexOf('{');
  const last = text.lastIndexOf('}');
  if (first < 0 || last < first) return null;
  const jsonSlice = text.slice(first, last + 1);

  let obj: unknown;
  try {
    obj = JSON.parse(jsonSlice);
  } catch {
    return null;
  }
  if (!obj || typeof obj !== 'object') return null;
  const rec = obj as Record<string, unknown>;
  if (typeof rec.goal !== 'string' || !Array.isArray(rec.steps)) return null;

  const steps: PlanStep[] = [];
  for (const [i, raw] of (rec.steps as unknown[]).entries()) {
    if (!raw || typeof raw !== 'object') continue;
    const s = raw as Record<string, unknown>;
    const title = typeof s.title === 'string' ? s.title : '';
    const rationale = typeof s.rationale === 'string' ? s.rationale : '';
    if (!title) continue;
    steps.push({
      id: typeof s.id === 'string' && s.id ? s.id : `s${i + 1}`,
      title,
      rationale,
      tool: typeof s.tool === 'string' ? s.tool : undefined,
      inputs: s.inputs && typeof s.inputs === 'object' ? (s.inputs as Record<string, unknown>) : undefined,
      estTokens: typeof s.estTokens === 'number' ? s.estTokens : undefined,
      status: 'pending',
    });
  }
  if (!steps.length) return null;

  return { goal: rec.goal, steps };
}

/** Build the executor message for an approved plan. */
export function buildExecutorMessage(plan: AgentPlan): string {
  const planJson = JSON.stringify(
    { goal: plan.goal, steps: plan.steps.map(({ status: _s, ...rest }) => rest) },
    null,
    2,
  );
  return `${EXECUTOR_PROMPT_PREFIX}\n\n${planJson}`;
}

/** Move a step up (delta=-1) or down (delta=+1). No-op at edges. */
export function moveStep(plan: AgentPlan, id: string, delta: -1 | 1): AgentPlan {
  const i = plan.steps.findIndex((s) => s.id === id);
  if (i < 0) return plan;
  const j = i + delta;
  if (j < 0 || j >= plan.steps.length) return plan;
  const next = [...plan.steps];
  [next[i], next[j]] = [next[j], next[i]];
  return { ...plan, steps: next };
}

export function removeStep(plan: AgentPlan, id: string): AgentPlan {
  return { ...plan, steps: plan.steps.filter((s) => s.id !== id) };
}

export function updateStep(plan: AgentPlan, id: string, patch: Partial<PlanStep>): AgentPlan {
  return {
    ...plan,
    steps: plan.steps.map((s) => (s.id === id ? { ...s, ...patch } : s)),
  };
}

export function addStep(plan: AgentPlan): AgentPlan {
  const id = `s${plan.steps.length + 1}-${Math.random().toString(36).slice(2, 6)}`;
  return {
    ...plan,
    steps: [
      ...plan.steps,
      { id, title: 'New step', rationale: '', status: 'pending' },
    ],
  };
}
