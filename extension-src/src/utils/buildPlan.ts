export interface BuildPlanItem {
  path: string;
  purpose: string;
  priority: number;
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, '/').replace(/^\.\//, '').replace(/^\/+/, '').trim();
}

function sanitizePlanItem(item: unknown, index: number): BuildPlanItem | null {
  if (!item || typeof item !== 'object') return null;
  const raw = item as Record<string, unknown>;
  const path = normalizePath(typeof raw.path === 'string' ? raw.path : '');
  if (!path) return null;

  return {
    path,
    purpose: typeof raw.purpose === 'string' && raw.purpose.trim()
      ? raw.purpose.trim()
      : 'Generated file',
    priority: typeof raw.priority === 'number' && Number.isFinite(raw.priority)
      ? raw.priority
      : index + 1,
  };
}

function sortAndDedupe(items: BuildPlanItem[]): BuildPlanItem[] {
  const seen = new Set<string>();
  return items
    .sort((a, b) => a.priority - b.priority || a.path.localeCompare(b.path))
    .filter((item) => {
      if (seen.has(item.path)) return false;
      seen.add(item.path);
      return true;
    })
    .map((item, index) => ({
      ...item,
      priority: index + 1,
    }));
}

function extractJsonArray(raw: string): unknown[] | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const candidates = [trimmed];
  const firstBracket = trimmed.indexOf('[');
  const lastBracket = trimmed.lastIndexOf(']');
  if (firstBracket !== -1 && lastBracket > firstBracket) {
    candidates.push(trimmed.slice(firstBracket, lastBracket + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) return parsed;
    } catch {
      // Ignore parse failures and try the next candidate.
    }
  }

  return null;
}

export function parseBuildPlan(response: string): BuildPlanItem[] {
  const parsedArray = extractJsonArray(response);
  if (parsedArray) {
    const items = parsedArray
      .map((item, index) => sanitizePlanItem(item, index))
      .filter((item): item is BuildPlanItem => item != null);
    if (items.length > 0) {
      return sortAndDedupe(items);
    }
  }

  const lineMatches = response.matchAll(/(?:^|\n)\s*(?:[-*]\s*)?([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)\s*(?:[-:]\s*(.+))?/g);
  const fallbackItems: BuildPlanItem[] = [];
  for (const match of lineMatches) {
    const path = normalizePath(match[1] || '');
    if (!path) continue;
    fallbackItems.push({
      path,
      purpose: (match[2] || 'Generated file').trim(),
      priority: fallbackItems.length + 1,
    });
  }

  return sortAndDedupe(fallbackItems);
}

export function formatBuildPlan(items: BuildPlanItem[]): string {
  return sortAndDedupe(items)
    .map((item) => `${item.priority}. ${item.path} — ${item.purpose}`)
    .join('\n');
}
