/**
 * Shared deduplication set for proactive messages.
 * Both useProactiveMessages (polling) and useWebSocket (push)
 * import this single Set so duplicates are caught across both paths.
 */

const MAX_SEEN_PROACTIVE = 500;

const seenProactiveIds = new Set<string>();

export function hasSeenProactive(id: string): boolean {
  return seenProactiveIds.has(id);
}

export function markProactiveSeen(id: string): void {
  seenProactiveIds.add(id);
  pruneSet(seenProactiveIds, MAX_SEEN_PROACTIVE);
}

function pruneSet(set: Set<string>, max: number): void {
  if (set.size <= max) return;
  const iter = set.values();
  for (let i = 0, n = set.size - max; i < n; i++) set.delete(iter.next().value!);
}
