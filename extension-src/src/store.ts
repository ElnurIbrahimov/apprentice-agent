import { create } from 'zustand';
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

  // UI
  activePanel: PanelId;
  moreOpen: boolean;

  // Chat
  messages: Message[];
  activeStream: StreamState | true | null;
  pendingCtx: Context | null;

  // Modes
  thinkingMode: boolean;
  deepResearch: boolean;

  // Models
  featureModels: Record<string, string>;
  mdlCloudList: string[];
  mdlLocalList: string[];

  // Actions
  setWs: (ws: WebSocket | null) => void;
  setWsReady: (ready: boolean) => void;
  setConversationId: (id: string | null) => void;
  setMood: (mood: string) => void;
  setModelName: (name: string) => void;
  setPanel: (panel: PanelId) => void;
  setMoreOpen: (open: boolean) => void;
  addMessage: (msg: Message) => void;
  setMessages: (msgs: Message[]) => void;
  setActiveStream: (stream: StreamState | true | null) => void;
  updateStreamText: (text: string) => void;
  setPendingCtx: (ctx: Context | null) => void;
  setThinkingMode: (on: boolean) => void;
  setDeepResearch: (on: boolean) => void;
  setModel: (feature: string, model: string | null) => void;
  setMdlLists: (cloud: string[], local: string[]) => void;
  clearAll: () => void;
  getModel: (feature: string) => string | null;
}

export const useStore = create<AuraStore>((set, get) => {
  // Load saved model prefs
  ext?.storage?.local?.get(['featureModels'], (d: any) => {
    set({ featureModels: d?.featureModels || {} });
  });

  return {
    ws: null,
    wsReady: false,
    conversationId: null,
    mood: '',
    modelName: '',
    activePanel: 'chat',
    moreOpen: false,
    messages: [],
    activeStream: null,
    pendingCtx: null,
    thinkingMode: false,
    deepResearch: false,
    featureModels: {},
    mdlCloudList: [],
    mdlLocalList: [],

    setWs: (ws) => set({ ws }),
    setWsReady: (wsReady) => set({ wsReady }),
    setConversationId: (conversationId) => set({ conversationId }),
    setMood: (mood) => set({ mood }),
    setModelName: (modelName) => set({ modelName }),
    setPanel: (activePanel) => set({ activePanel }),
    setMoreOpen: (moreOpen) => set({ moreOpen }),
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

    setModel: (feature, model) => {
      const featureModels = { ...get().featureModels };
      if (model) featureModels[feature] = model;
      else delete featureModels[feature];
      ext?.storage?.local?.set({ featureModels });
      set({ featureModels });
    },

    setMdlLists: (mdlCloudList, mdlLocalList) => set({ mdlCloudList, mdlLocalList }),

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
