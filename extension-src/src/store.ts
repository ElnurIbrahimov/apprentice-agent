import { create } from 'zustand';
import { FEATURE_DEFS } from './types';
import type { Message, StreamState, Context, PanelId } from './types';
import { HTTP } from './api';
import ext from './ext';

interface AuraStore {
  // WebSocket
  ws: WebSocket | null;
  wsReady: boolean;
  conversationId: string | null;

  // Status
  mood: string;
  modelName: string;
  backendStatus: 'online' | 'offline' | 'connecting';
  lastBackendCheck: number;

  // UI
  activePanel: PanelId;
  moreOpen: boolean;
  theme: 'dark' | 'light';

  // Chat
  messages: Message[];
  activeStream: StreamState | true | null;
  pendingCtx: Context | null;

  // Modes
  thinkingMode: boolean;
  deepResearch: boolean;
  autoSpeak: boolean;

  // Models
  featureModels: Record<string, string>;
  mdlCloudList: string[];
  mdlLocalList: string[];
  mdlChatgptList: string[];
  mdlListsLoaded: boolean;

  // Actions
  setWs: (ws: WebSocket | null) => void;
  setWsReady: (ready: boolean) => void;
  setConversationId: (id: string | null) => void;
  setMood: (mood: string) => void;
  setModelName: (name: string) => void;
  setBackendStatus: (status: 'online' | 'offline' | 'connecting') => void;
  setPanel: (panel: PanelId) => void;
  setMoreOpen: (open: boolean) => void;
  toggleTheme: () => void;
  addMessage: (msg: Message) => void;
  setMessages: (msgs: Message[]) => void;
  setActiveStream: (stream: StreamState | true | null) => void;
  updateStreamText: (text: string) => void;
  setPendingCtx: (ctx: Context | null) => void;
  setThinkingMode: (on: boolean) => void;
  setDeepResearch: (on: boolean) => void;
  setAutoSpeak: (on: boolean) => void;
  setModel: (feature: string, model: string | null) => void;
  setMdlLists: (cloud: string[], local: string[], chatgpt?: string[]) => void;
  setAllModels: (model: string) => void;
  loadModels: () => Promise<void>;
  clearAll: () => void;
  getModel: (feature: string) => string | null;
}

export const useStore = create<AuraStore>((set, get) => {
  // Load saved model prefs + cached model lists
  ext?.storage?.local?.get(['featureModels', 'cachedModelLists'], (d: any) => {
    const cached = d?.cachedModelLists;
    const savedFeatureModels = d?.featureModels || {};
    // If cached lists exist, pre-populate so UI doesn't flash empty
    if (cached) {
      const allAvailable = [...(cached.cloud || []), ...(cached.local || []), ...(cached.chatgpt || [])];
      // Prune stale feature assignments — remove models no longer available
      const pruned: Record<string, string> = {};
      for (const [k, v] of Object.entries(savedFeatureModels)) {
        if (allAvailable.includes(v as string)) pruned[k] = v as string;
      }
      set({
        featureModels: pruned,
        mdlCloudList: cached.cloud || [],
        mdlLocalList: cached.local || [],
        mdlChatgptList: cached.chatgpt || [],
        mdlListsLoaded: true,
      });
      // Persist pruned if different
      if (Object.keys(pruned).length !== Object.keys(savedFeatureModels).length) {
        ext?.storage?.local?.set({ featureModels: pruned });
      }
    } else {
      set({ featureModels: savedFeatureModels });
    }
  });

  // Load saved autoSpeak pref
  ext?.storage?.local?.get(['autoSpeak'], (d: any) => {
    set({ autoSpeak: !!d?.autoSpeak });
  });

  // Load saved theme
  ext?.storage?.local?.get(['theme'], (d: any) => {
    const saved = d?.theme === 'light' ? 'light' : 'dark';
    set({ theme: saved });
    document.documentElement.classList.toggle('light', saved === 'light');
  });

  return {
    ws: null,
    wsReady: false,
    conversationId: null,
    mood: '',
    modelName: '',
    backendStatus: 'connecting' as 'online' | 'offline' | 'connecting',
    lastBackendCheck: 0,
    activePanel: 'chat',
    moreOpen: false,
    theme: 'dark' as 'dark' | 'light',
    messages: [],
    activeStream: null,
    pendingCtx: null,
    thinkingMode: false,
    deepResearch: false,
    autoSpeak: false,
    featureModels: {},
    mdlCloudList: [],
    mdlLocalList: [],
    mdlChatgptList: [],
    mdlListsLoaded: false,

    setWs: (ws) => set({ ws }),
    setWsReady: (wsReady) => set({ wsReady }),
    setConversationId: (conversationId) => set({ conversationId }),
    setMood: (mood) => set({ mood }),
    setModelName: (modelName) => set({ modelName }),
    setBackendStatus: (backendStatus) => set({ backendStatus, lastBackendCheck: Date.now() }),
    setPanel: (activePanel) => set({ activePanel }),
    setMoreOpen: (moreOpen) => set({ moreOpen }),
    toggleTheme: () => {
      const next = get().theme === 'dark' ? 'light' : 'dark';
      set({ theme: next });
      document.documentElement.classList.toggle('light', next === 'light');
      ext?.storage?.local?.set({ theme: next });
    },
    addMessage: (msg) => set(s => {
      const msgs = [...s.messages, msg];
      // Cap at 500 messages to prevent unbounded growth
      return { messages: msgs.length > 500 ? msgs.slice(-500) : msgs };
    }),
    setMessages: (messages) => set({ messages }),
    setActiveStream: (activeStream) => set({ activeStream }),
    updateStreamText: (text) =>
      set(s => {
        const stream = s.activeStream;
        if (!stream || stream === true) return s;
        return { activeStream: { ...stream, rawText: stream.rawText + text } };
      }),
    setPendingCtx: (pendingCtx) => set({ pendingCtx }),
    setThinkingMode: (thinkingMode) => set({ thinkingMode }),
    setDeepResearch: (deepResearch) => set({ deepResearch }),
    setAutoSpeak: (autoSpeak) => {
      set({ autoSpeak });
      ext?.storage?.local?.set({ autoSpeak });
    },

    setModel: (feature, model) => {
      const featureModels = { ...get().featureModels };
      if (model) featureModels[feature] = model;
      else delete featureModels[feature];
      ext?.storage?.local?.set({ featureModels });
      set({ featureModels });
    },

    setMdlLists: (mdlCloudList, mdlLocalList, mdlChatgptList = []) => {
      set({ mdlCloudList, mdlLocalList, mdlChatgptList, mdlListsLoaded: true });
      // Cache to storage for fast restore on next open
      ext?.storage?.local?.set({
        cachedModelLists: { cloud: mdlCloudList, local: mdlLocalList, chatgpt: mdlChatgptList },
      });
      // Prune stale feature model assignments
      const allAvailable = [...mdlCloudList, ...mdlLocalList, ...mdlChatgptList];
      const fm = { ...get().featureModels };
      let changed = false;
      for (const [k, v] of Object.entries(fm)) {
        if (!allAvailable.includes(v)) {
          delete fm[k];
          changed = true;
        }
      }
      if (changed) {
        set({ featureModels: fm });
        ext?.storage?.local?.set({ featureModels: fm });
      }
    },

    setAllModels: (model: string) => {
      const fm: Record<string, string> = {};
      for (const def of FEATURE_DEFS) {
        if (model) fm[def.key] = model;
      }
      set({ featureModels: fm });
      ext?.storage?.local?.set({ featureModels: fm });
    },

    loadModels: async () => {
      const s = get();
      if (s.mdlListsLoaded && (s.mdlCloudList.length || s.mdlLocalList.length || s.mdlChatgptList.length)) return;
      let cloud: string[] = [];
      let local: string[] = [];
      let chatgpt: string[] = [];
      try {
        const d = await fetch('http://localhost:11434/api/tags').then(r => r.json());
        const all: string[] = (d.models || []).map((m: any) => m.name);
        cloud = all.filter(n => n.includes(':cloud'));
        local = all.filter(n => !n.includes(':cloud'));
      } catch {
        try {
          const d = await fetch(`${HTTP}/api/models/available`).then(r => r.json());
          cloud = (d.cloud || []).map((m: any) => m.name || m);
          local = (d.local || []).map((m: any) => m.name || m);
          chatgpt = (d.chatgpt || []).map((m: any) => m.name || m);
        } catch { /* leave empty */ }
      }
      // Try fetching ChatGPT models separately if not already loaded
      if (chatgpt.length === 0) {
        try {
          const authStatus = await fetch(`${HTTP}/api/auth/chatgpt/status`).then(r => r.json());
          if (authStatus.authenticated) {
            const modelsResp = await fetch(`${HTTP}/api/models/available`).then(r => r.json());
            chatgpt = (modelsResp.chatgpt || []).map((m: any) => m.name || m);
          }
        } catch { /* no chatgpt */ }
      }
      get().setMdlLists(cloud, local, chatgpt);
    },

    clearAll: () => {
      const s = get();
      // Unblock stream if running
      if (s.activeStream && s.activeStream !== true) {
        // stream was active — just null it
      }
      set({
        messages: [],
        activeStream: null,
        conversationId: null,
        pendingCtx: null,
        thinkingMode: false,
        deepResearch: false,
      });
      // Tell backend to clear
      if (s.wsReady && s.ws?.readyState === WebSocket.OPEN) {
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST' }).catch(() => {});
      }
    },

    getModel: (feature) => get().featureModels[feature] || null,
  };
});
