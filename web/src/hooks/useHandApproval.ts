/**
 * Pending Hand approval state.
 *
 * Listens for the `aura:hand_approval` CustomEvent dispatched by
 * `useWebSocket.ts` when the server broadcasts a `hand_approval_request`,
 * and exposes a resolve() helper that POSTs to /api/hands/{name}/approve.
 *
 * Used by AgentPanel to render an inline approval prompt during a live run.
 */

import { useCallback, useEffect, useState } from 'react';

export interface PendingApproval {
  request_id: string;
  hand_name: string;
  tool_name: string;
  args: Record<string, unknown>;
  timestamp: string;
  age_seconds?: number;
}

export function useHandApproval() {
  const [pending, setPending] = useState<PendingApproval | null>(null);
  const [resolving, setResolving] = useState(false);

  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (!detail) return;
      setPending({
        request_id: String(detail.request_id ?? detail.id ?? ''),
        hand_name: String(detail.hand_name ?? detail.hand ?? 'unknown'),
        tool_name: String(detail.tool_name ?? detail.tool ?? 'unknown'),
        args: (detail.args && typeof detail.args === 'object') ? detail.args : {},
        timestamp: String(detail.timestamp ?? new Date().toISOString()),
        age_seconds: typeof detail.age_seconds === 'number' ? detail.age_seconds : undefined,
      });
    };
    window.addEventListener('aura:hand_approval', handler);
    return () => window.removeEventListener('aura:hand_approval', handler);
  }, []);

  const resolve = useCallback(async (approved: boolean) => {
    const target = pending;
    if (!target) return false;
    setResolving(true);
    try {
      const res = await fetch(`/api/hands/${encodeURIComponent(target.hand_name)}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ approved }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPending(null);
      return true;
    } catch (e: any) {
      console.warn('[useHandApproval] resolve failed:', e);
      return false;
    } finally {
      setResolving(false);
    }
  }, [pending]);

  const dismiss = useCallback(() => setPending(null), []);

  return { pending, resolve, resolving, dismiss };
}
