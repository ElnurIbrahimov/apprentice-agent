import { HTTP, apiFetch } from '../api';
import type {
  HandStats,
  HandApprovalRequest,
  HandHistoryEntry,
  HandTemplate,
  HandLiveTrace,
} from '../types';

const HAND_LIVE_TRACE_MAX = 20;

type SetFn = (partial: any) => void;
type GetFn = () => any;

export function createHandsActions(set: SetFn, get: GetFn) {
  return {
    loadHands: async () => {
      try {
        const data = await apiFetch(`${HTTP}/api/hands`);
        set({ hands: (data?.hands || []) as HandStats[], handsLoaded: true, handsError: null });
      } catch (err: any) {
        set({ handsError: err?.message || 'Failed to load hands', handsLoaded: true });
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
      get().loadHands();
    },

    pauseHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/pause`, { method: 'POST' });
      get().loadHands();
    },

    activateHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/activate`, { method: 'POST' });
      get().loadHands();
    },

    deactivateHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/deactivate`, { method: 'POST' });
      get().loadHands();
    },

    deleteHand: async (name: string) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}`, { method: 'DELETE' });
      set((s: any) => ({ hands: s.hands.filter((h: HandStats) => h.name !== name) }));
    },

    approveHand: async (name: string, _requestId: string, approved: boolean) => {
      await apiFetch(`${HTTP}/api/hands/${encodeURIComponent(name)}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ approved }),
      });
      set((s: any) => ({ handApprovals: s.handApprovals.filter((a: HandApprovalRequest) => a.hand_name !== name) }));
    },

    createHand: async (description: string) => {
      await apiFetch(`${HTTP}/api/hands/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description }),
      });
      get().loadHands();
    },

    createHandFromTemplate: async (templateName: string, variables?: Record<string, string>) => {
      await apiFetch(`${HTTP}/api/hands/from-template`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template_name: templateName, variables: variables || {} }),
      });
      get().loadHands();
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
