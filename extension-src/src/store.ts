import { create } from 'zustand';
import type { Message, StreamState, Context, PanelId, ThinkingLevel, ConversationMeta } from './types';
import { HTTP, API_KEY } from './api';
import ext from './ext';

// --- Conversation history constants ---
const MAX_CONVERSATIONS = 50;
const MAX_MESSAGES_PER_CONVERSATION = 100;
const CONV_LIST_KEY = 'aura_conversations';
const ACTIVE_CONV_KEY = 'aura_active_conversation';
const convStorageKey = (id: string) => `aura_chat_${id}`;

// --- Storage helpers (chrome.storage.local) ---
function storageGet(keys: string[]): Promise<Record<string, any>> {
  return new Promise((resolve) => {
    ext?.storage?.local?.get(keys, (d: any) => resolve(d || {}));
  });
}
function storageSet(data: Record<string, any>): Promise<void> {
  return new Promise((resolve) => {
    ext?.storage?.local?.set(data, () => resolve());
  });
}
function storageRemove(keys: string[]): Promise<void> {
  return new Promise((resolve) => {
    ext?.storage?.local?.remove(keys, () => resolve());
  });
}

interface AuraStore {
  // WebSocket
  ws: WebSocket | null;
  wsReady: boolean;
  conversationId: string | null;

  // Status
  mood: string;
  modelName: string;
  backendStatus: 'online' | 'offline' | 'connecting';

  // UI
  activePanel: PanelId;
  moreOpen: boolean;
  theme: 'dark' | 'light';

  // Chat
  messages: Message[];
  activeStream: StreamState | true | null;
  pendingCtx: Context | null;

  // Conversation History
  conversations: ConversationMeta[];
  activeConversationId: string | null;
  historyLoaded: boolean;

  // Modes
  thinkingMode: boolean;
  thinkingLevel: ThinkingLevel;
  deepResearch: boolean;
  autoSpeak: boolean;

  // Custom Instructions / Persona
  customInstructions: string;
  userName: string;

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
  setActiveStream: (stream: StreamState | true | null) => void;
  setPendingCtx: (ctx: Context | null) => void;
  setThinkingMode: (on: boolean) => void;
  setThinkingLevel: (level: ThinkingLevel) => void;
  setDeepResearch: (on: boolean) => void;
  setAutoSpeak: (on: boolean) => void;
  setCustomInstructions: (text: string) => void;
  setUserName: (name: string) => void;
  setModel: (feature: string, model: string | null) => void;
  setMdlLists: (cloud: string[], local: string[], chatgpt?: string[]) => void;
  loadModels: () => Promise<void>;
  clearAll: () => void;
  getModel: (feature: string) => string | null;

  // Conversation History Actions
  loadConversationList: () => Promise<void>;
  saveCurrentConversation: () => Promise<void>;
  loadConversation: (id: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  clearAllHistory: () => Promise<void>;
  newConversation: () => Promise<void>;
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

  // Load saved custom instructions & user name
  ext?.storage?.local?.get(['customInstructions', 'userName'], (d: any) => {
    set({
      customInstructions: d?.customInstructions || '',
      userName: d?.userName || '',
    });
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
    activePanel: 'chat',
    moreOpen: false,
    theme: 'dark' as 'dark' | 'light',
    messages: [],
    activeStream: null,
    pendingCtx: null,
    conversations: [],
    activeConversationId: null,
    historyLoaded: false,
    thinkingMode: false,
    thinkingLevel: 'medium' as ThinkingLevel,
    deepResearch: false,
    autoSpeak: false,
    customInstructions: '',
    userName: '',
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
    setBackendStatus: (backendStatus) => set({ backendStatus }),
    setPanel: (activePanel) => set({ activePanel }),
    setMoreOpen: (moreOpen) => set({ moreOpen }),
    toggleTheme: () => {
      const next = get().theme === 'dark' ? 'light' : 'dark';
      set({ theme: next });
      document.documentElement.classList.toggle('light', next === 'light');
      ext?.storage?.local?.set({ theme: next });
    },
    addMessage: (msg) => {
      set(s => {
        const msgs = [...s.messages, msg];
        // Cap at 500 messages to prevent unbounded growth
        return { messages: msgs.length > 500 ? msgs.slice(-500) : msgs };
      });
      // Debounced auto-save: save after each user+ai pair settles
      // We do it on a microtask so streaming doesn't hammer storage
      if (msg.role === 'user') {
        const s = get();
        // Create conversation on first user message if none active
        if (!s.activeConversationId) {
          const newId = crypto.randomUUID();
          set({ activeConversationId: newId });
        }
      }
    },
    setActiveStream: (activeStream) => set({ activeStream }),
    setPendingCtx: (pendingCtx) => set({ pendingCtx }),
    setThinkingMode: (thinkingMode) => set({ thinkingMode }),
    setThinkingLevel: (thinkingLevel) => set({ thinkingLevel }),
    setDeepResearch: (deepResearch) => set({ deepResearch }),
    setAutoSpeak: (autoSpeak) => {
      set({ autoSpeak });
      ext?.storage?.local?.set({ autoSpeak });
    },

    setCustomInstructions: (customInstructions) => {
      set({ customInstructions });
      ext?.storage?.local?.set({ customInstructions });
    },
    setUserName: (userName) => {
      set({ userName });
      ext?.storage?.local?.set({ userName });
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

    loadModels: async () => {
      const s = get();
      if (s.mdlListsLoaded && (s.mdlCloudList.length || s.mdlLocalList.length || s.mdlChatgptList.length)) return;
      let cloud: string[] = [];
      let local: string[] = [];
      let chatgpt: string[] = [];

      // 1. Try Ollama for local/cloud models
      try {
        const d = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(3000) }).then(r => r.json());
        const all: string[] = (d.models || []).map((m: any) => m.name);
        cloud = all.filter(n => n.includes(':cloud'));
        local = all.filter(n => !n.includes(':cloud'));
      } catch { /* Ollama not available */ }

      // 2. Try backend for models (also gets ChatGPT list)
      try {
        const d = await fetch(`${HTTP}/api/models/available`, { signal: AbortSignal.timeout(3000), headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).then(r => r.json());
        // Merge backend models with Ollama models (Ollama may have more)
        const backendCloud = (d.cloud || []).map((m: any) => m.name || m);
        const backendLocal = (d.local || []).map((m: any) => m.name || m);
        chatgpt = (d.chatgpt || []).map((m: any) => m.name || m);
        // Add any backend models not already in Ollama list
        for (const m of backendCloud) if (!cloud.includes(m)) cloud.push(m);
        for (const m of backendLocal) if (!local.includes(m)) local.push(m);
      } catch { /* backend not available */ }

      // 3. Always include ChatGPT models — auth is checked at request time, not listing time.
      // Users should always SEE the models are available. If they select one without auth,
      // the backend returns a clear "run aura --login chatgpt" error.
      if (chatgpt.length === 0) {
        chatgpt = [
          'chatgpt:gpt-5.4', 'chatgpt:gpt-5.4-thinking', 'chatgpt:gpt-5.4-pro',
          'chatgpt:gpt-5.3', 'chatgpt:gpt-5.3-codex', 'chatgpt:gpt-5.3-codex-spark',
          'chatgpt:gpt-5.2', 'chatgpt:gpt-5.2-codex',
          'chatgpt:gpt-5.1', 'chatgpt:gpt-5.1-codex', 'chatgpt:gpt-5.1-codex-mini', 'chatgpt:gpt-5.1-codex-max',
        ];
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
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).catch(() => {});
      }
    },

    getModel: (feature) => get().featureModels[feature] || null,

    // --- Conversation History ---

    loadConversationList: async () => {
      const data = await storageGet([CONV_LIST_KEY, ACTIVE_CONV_KEY]);
      const convs: ConversationMeta[] = data[CONV_LIST_KEY] || [];
      const activeId: string | null = data[ACTIVE_CONV_KEY] || null;
      set({ conversations: convs, historyLoaded: true });

      // Restore last active conversation
      if (activeId && convs.some(c => c.id === activeId)) {
        await get().loadConversation(activeId);
      }
    },

    saveCurrentConversation: async () => {
      const s = get();
      if (s.messages.length === 0) return;

      const convId = s.activeConversationId || crypto.randomUUID();
      if (!s.activeConversationId) set({ activeConversationId: convId });

      // Build title from first user message, truncated
      const firstUserMsg = s.messages.find(m => m.role === 'user');
      const title = firstUserMsg
        ? firstUserMsg.text.slice(0, 60) + (firstUserMsg.text.length > 60 ? '...' : '')
        : 'New conversation';

      // Strip base64 image data from messages before persisting
      const messagesToSave = s.messages.slice(-MAX_MESSAGES_PER_CONVERSATION).map(m => ({
        id: m.id,
        role: m.role,
        text: m.text,
        timestamp: m.timestamp,
        thinkingContent: m.thinkingContent,
      }));

      const meta: ConversationMeta = {
        id: convId,
        title,
        timestamp: Date.now(),
        messageCount: s.messages.length,
      };

      // Update conversations list
      let convs = [...s.conversations];
      const existingIdx = convs.findIndex(c => c.id === convId);
      if (existingIdx >= 0) {
        convs[existingIdx] = meta;
      } else {
        convs.unshift(meta);
      }
      // Cap at MAX_CONVERSATIONS — remove oldest
      if (convs.length > MAX_CONVERSATIONS) {
        const removed = convs.splice(MAX_CONVERSATIONS);
        // Clean up storage for removed conversations
        const keysToRemove = removed.map(c => convStorageKey(c.id));
        storageRemove(keysToRemove).catch(() => {});
      }
      // Sort by timestamp descending
      convs.sort((a, b) => b.timestamp - a.timestamp);

      set({ conversations: convs });
      await storageSet({
        [CONV_LIST_KEY]: convs,
        [ACTIVE_CONV_KEY]: convId,
        [convStorageKey(convId)]: messagesToSave,
      });
    },

    loadConversation: async (id: string) => {
      const data = await storageGet([convStorageKey(id)]);
      const messages: Message[] = data[convStorageKey(id)] || [];
      set({
        messages,
        activeConversationId: id,
        activeStream: null,
        conversationId: null,
        pendingCtx: null,
      });
      await storageSet({ [ACTIVE_CONV_KEY]: id });
      // Tell backend to clear since we're loading a different conversation
      const s = get();
      if (s.wsReady && s.ws?.readyState === WebSocket.OPEN) {
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).catch(() => {});
      }
    },

    deleteConversation: async (id: string) => {
      const s = get();
      const convs = s.conversations.filter(c => c.id !== id);
      set({ conversations: convs });
      await storageSet({ [CONV_LIST_KEY]: convs });
      await storageRemove([convStorageKey(id)]);
      // If we deleted the active one, clear chat
      if (s.activeConversationId === id) {
        set({ messages: [], activeConversationId: null, conversationId: null });
        await storageRemove([ACTIVE_CONV_KEY]);
      }
    },

    clearAllHistory: async () => {
      const s = get();
      const keysToRemove = s.conversations.map(c => convStorageKey(c.id));
      keysToRemove.push(CONV_LIST_KEY, ACTIVE_CONV_KEY);
      set({ conversations: [], activeConversationId: null });
      await storageRemove(keysToRemove);
    },

    newConversation: async () => {
      const s = get();
      // Save current conversation first if it has messages
      if (s.messages.length > 0) {
        await s.saveCurrentConversation();
      }
      // Clear for a new conversation
      set({
        messages: [],
        activeStream: null,
        conversationId: null,
        activeConversationId: null,
        pendingCtx: null,
        thinkingMode: false,
        deepResearch: false,
      });
      await storageRemove([ACTIVE_CONV_KEY]);
      // Tell backend to clear
      if (s.wsReady && s.ws?.readyState === WebSocket.OPEN) {
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).catch(() => {});
      }
    },
  };
});
