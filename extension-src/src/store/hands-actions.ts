import { HTTP, apiFetch } from '../api';
import type {
  HandStats,
  HandApprovalRequest,
  HandHistoryEntry,
  HandTemplate,
  HandLiveTrace,
} from '../types';

const HAND_LIVE_TRACE_MAX = 20;

async function withRetry<T>(fn: () => Promise<T>, retries = 2): Promise<T | null> {
  let lastErr: any;
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (err) {
      lastErr = err;
      if (i < retries) await new Promise(r => setTimeout(r, 1000 * (i + 1)));
    }
  }
  return null;
}

type SetFn = (partial: any) => void;
type GetFn = () => any;

export function createHandsActions(set: SetFn, get: GetFn) {
  return {
    loadHands: async () => {
      set({ handsError: null });
      const data = await withRetry(() => apiFetch(`${HTTP}/api/hands`));
      if (data) {
        set({ hands: (data?.hands || []) as HandStats[], handsLoaded: true, handsError: null });
      } else {
        set({ handsError: 'Failed to load hands after retries', handsLoaded: true });
      }
    },

    loadHandApprovals: async () => {
      try {
        const data = await apiFetch(`${HTTP}/api/hands/approvals`);
        set({ handApprovals: (data?.approvals || []) as HandApprovalRequest[] });
      } catch { /* non-fatal */ }
    },

    loadHandHistory: async (limit = 30) => {
      try {
        const data = await apiFetch(`${HTTP}/api/hands/history?limit=${limit}`);
        set({ handHistory: (data?.history || []) as HandHistoryEntry[] });
      } catch { /* non-fatal */ }
    },

    loadHandTemplates: async () => {
      try {
        const data = await apiFetch(`${HTTP}/api/hands/templates`);
        set({ handTemplates: (data?.templates || []) as HandTemplate[] });
      } catch { /* non-fatal */ }
    },

    runHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/run`, { method: 'POST' });
      await get().loadHands();
    },

    pauseHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/pause`, { method: 'POST' });
      await get().loadHands();
    },

    activateHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/activate`, { method: 'POST' });
      await get().loadHands();
    },

    deactivateHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/deactivate`, { method: 'POST' });
      await get().loadHands();
    },

    deleteHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}`, { method: 'DELETE' });
      await get().loadHands();
    },

    approveHand: async (name: string, _requestId: string, approved: boolean) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      });
      set((s: any) => ({ handApprovals: s.handApprovals.filter((a: HandApprovalRequest) => a.request_id !== _requestId) }));
    },

    createHand: async (description: string) => {
      await apiFetch(`${HTTP}/api/hands/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description }),
      });
      await get().loadHands();
    },

    createHandFromTemplate: async (templateName: string, variables?: Record<string, string>) => {
      await apiFetch(`${HTTP}/api/hands/from-template`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template_name: templateName, variables: variables || {} }),
      });
      await get().loadHands();
    },

    applyHandEvent: (ev: any) => {
      set((s: any) => {
        const handName = ev.hand || ev.hand_name;
        const next = s.hands.map((h: HandStats) =>
          h.name === handName
            ? { ...h, total_runs: h.total_runs + 1, last_run_ts: Date.now() / 1000, state: ev.success === false ? 'error' : 'cooldown' }
            : h,
        );
        const historyEntry: HandHistoryEntry = {
          timestamp: new Date().toISOString(),
          action_type: 'hand_complete',
          action_data: {
            success: ev.success !== false,
            hand: handName,
            summary: ev.summary || '',
            duration_ms: Math.round((ev.duration_seconds || 0) * 1000),
          },
          agent_id: `hand:${handName}`,
        };
        return { hands: next, handHistory: [historyEntry, ...s.handHistory].slice(0, 50) };
      });
    },

    applyHandApprovalRequest: (req: any) => {
      const entry: HandApprovalRequest = {
        request_id: String(req.request_id),
        hand_name: String(req.hand_name),
        tool_name: String(req.tool_name),
        args: (req.args && typeof req.args === 'object') ? req.args : {},
        timestamp: req.timestamp || Date.now(),
        age_seconds: Number(req.age_seconds || 0),
      };
      set((s: any) => {
        if (s.handApprovals.some((a: HandApprovalRequest) => a.request_id === entry.request_id)) return s;
        return { handApprovals: [...s.handApprovals, entry] };
      });
    },

    applyHandActionTrace: (trace: any) => {
      const entry: HandLiveTrace = {
        hand: String(trace.hand || ''),
        step: Number(trace.step || 0),
        description: String(trace.description || ''),
        timestamp: Number(trace.timestamp || Date.now()),
      };
      set((s: any) => ({
        handLiveTrace: [...s.handLiveTrace, entry].slice(-HAND_LIVE_TRACE_MAX),
      }));
    },
  };
}
