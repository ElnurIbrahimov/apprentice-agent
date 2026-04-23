/**
 * Agent recipes — user-saved reusable plans, stored locally.
 *
 * A recipe is a snapshot of an approved plan (goal + steps) that the user
 * can re-invoke later to skip the Planning phase. All storage is in
 * localStorage under a single key; no backend state.
 */

import type { AgentPlan } from './agentPlan';

const STORAGE_KEY = 'aura-agent-recipes';

export interface AgentRecipe {
  id: string;
  name: string;
  description?: string;
  plan: AgentPlan;
  createdAt: number;
}

function readRaw(): AgentRecipe[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((r): r is AgentRecipe =>
      r && typeof r.id === 'string' && typeof r.name === 'string' && r.plan && Array.isArray(r.plan.steps)
    );
  } catch {
    return [];
  }
}

function writeRaw(recipes: AgentRecipe[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(recipes));
  } catch { /* private mode */ }
}

export function listRecipes(): AgentRecipe[] {
  return readRaw().sort((a, b) => b.createdAt - a.createdAt);
}

export function saveRecipe(name: string, plan: AgentPlan, description?: string): AgentRecipe {
  const recipes = readRaw();
  const recipe: AgentRecipe = {
    id: `rcp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    name: name.trim() || plan.goal.slice(0, 60),
    description: description?.trim() || undefined,
    // Strip runtime status before saving.
    plan: {
      goal: plan.goal,
      steps: plan.steps.map(({ status: _s, ...rest }) => rest),
    },
    createdAt: Date.now(),
  };
  writeRaw([recipe, ...recipes]);
  return recipe;
}

export function deleteRecipe(id: string): void {
  writeRaw(readRaw().filter((r) => r.id !== id));
}

export function renameRecipe(id: string, name: string): void {
  writeRaw(readRaw().map((r) => (r.id === id ? { ...r, name } : r)));
}
