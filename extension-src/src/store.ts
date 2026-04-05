import { create } from 'zustand';
import type { Message, StreamState, Context, PanelId, ThinkingLevel, ConversationMeta } from './types';
import { HTTP, API_KEY } from './api';
import ext from './ext';
import * as db from './utils/db';

// --- Conversation history constants ---
const MAX_CONVERSATIONS = 500;
const MAX_MESSAGES_PER_CONVERSATION = 1000;
const USE_IDB = db.isIndexedDBAvailable();
const CONV_LIST_KEY = 'aura_conversations';
const ACTIVE_CONV_KEY = 'aura_active_conversation';
const CONV_FOLDERS_KEY = 'aura_conv_folders';
const convStorageKey = (id: string) => `aura_chat_${id}`;
let _idbMigrated = false;

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

// Late-bound reconnect handler — set by ws.ts to avoid circular imports
let _reconnectHandler: (() => void) | null = null;
export function registerReconnectHandler(handler: () => void) {
  _reconnectHandler = handler;
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
  agentReady: boolean;
  connectionAbandoned: boolean;

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
  mdlDirectList: string[];
  mdlListsLoaded: boolean;

  // Actions
  setWs: (ws: WebSocket | null) => void;
  setWsReady: (ready: boolean) => void;
  setConversationId: (id: string | null) => void;
  setMood: (mood: string) => void;
  setModelName: (name: string) => void;
  setBackendStatus: (status: 'online' | 'offline' | 'connecting') => void;
  setAgentReady: (ready: boolean) => void;
  setConnectionAbandoned: (abandoned: boolean) => void;
  reconnect: () => void;
  setPanel: (panel: PanelId) => void;
  /** Navigate to a panel with a data handoff */
  handoffToPanel: (panel: PanelId, data: Record<string, any>) => void;
  /** Consume pending handoff data (one-shot, clears after read) */
  consumePanelHandoff: () => Record<string, any> | null;
  _pendingHandoff: Record<string, any> | null;
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
  setMdlLists: (cloud: string[], local: string[], chatgpt?: string[], direct?: string[]) => void;
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

  // Folder & Pin Actions
  folders: string[];
  createFolder: (name: string) => Promise<void>;
  deleteFolder: (name: string) => Promise<void>;
  renameFolder: (oldName: string, newName: string) => Promise<void>;
  moveToFolder: (convId: string, folder?: string) => Promise<void>;
  pinConversation: (convId: string, pinned: boolean) => Promise<void>;
}

export const useStore = create<AuraStore>((set, get) => {
  // Load saved model prefs + cached model lists
  ext?.storage?.local?.get(['featureModels', 'cachedModelLists'], (d: any) => {
    const cached = d?.cachedModelLists;
    const savedFeatureModels = d?.featureModels || {};
    // If cached lists exist, pre-populate so UI doesn't flash empty
    if (cached) {
      const allAvailable = [...(cached.cloud || []), ...(cached.local || []), ...(cached.chatgpt || []), ...(cached.direct || [])];
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
        mdlDirectList: cached.direct || [],
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

  // Load saved autoSpeak + thinking/research prefs
  ext?.storage?.local?.get(['autoSpeak', 'thinkingMode', 'thinkingLevel', 'deepResearch'], (d: any) => {
    set({
      autoSpeak: !!d?.autoSpeak,
      thinkingMode: !!d?.thinkingMode,
      thinkingLevel: d?.thinkingLevel || 'medium',
      deepResearch: !!d?.deepResearch,
    });
  });

  // Load saved custom instructions & user name
  ext?.storage?.local?.get(['customInstructions', 'userName'], (d: any) => {
    set({
      customInstructions: d?.customInstructions || '',
      userName: d?.userName || '',
    });
  });

  // Load saved conversation folders
  ext?.storage?.local?.get([CONV_FOLDERS_KEY], (d: any) => {
    set({ folders: d?.[CONV_FOLDERS_KEY] || [] });
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
    agentReady: false,
    connectionAbandoned: false,
    activePanel: 'chat',
    moreOpen: false,
    theme: 'dark' as 'dark' | 'light',
    messages: [],
    activeStream: null,
    pendingCtx: null,
    conversations: [],
    activeConversationId: null,
    historyLoaded: false,
    folders: [],
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
    mdlDirectList: [],
    mdlListsLoaded: false,

    setWs: (ws) => set({ ws }),
    setWsReady: (wsReady) => set({ wsReady }),
    setConversationId: (conversationId) => set({ conversationId }),
    setMood: (mood) => set({ mood }),
    setModelName: (modelName) => set({ modelName }),
    setBackendStatus: (backendStatus) => set({ backendStatus }),
    setAgentReady: (agentReady) => set({ agentReady }),
    setConnectionAbandoned: (connectionAbandoned) => set({ connectionAbandoned }),
    reconnect: () => {
      // Calls the registered reconnect handler from ws.ts
      // (set via registerReconnectHandler to avoid circular imports)
      if (_reconnectHandler) _reconnectHandler();
    },
    setPanel: (activePanel) => set({ activePanel }),
    _pendingHandoff: null,
    handoffToPanel: (panel, data) => set({ activePanel: panel, _pendingHandoff: data }),
    consumePanelHandoff: () => {
      const state = get();
      if (!state._pendingHandoff) return null;
      const data = state._pendingHandoff;
      set({ _pendingHandoff: null });
      return data;
    },
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
    setThinkingMode: (thinkingMode) => {
      set({ thinkingMode });
      ext?.storage?.local?.set({ thinkingMode });
    },
    setThinkingLevel: (thinkingLevel) => {
      set({ thinkingLevel });
      ext?.storage?.local?.set({ thinkingLevel });
    },
    setDeepResearch: (deepResearch) => {
      set({ deepResearch });
      ext?.storage?.local?.set({ deepResearch });
    },
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

    setMdlLists: (mdlCloudList, mdlLocalList, mdlChatgptList = [], mdlDirectList = []) => {
      set({ mdlCloudList, mdlLocalList, mdlChatgptList, mdlDirectList, mdlListsLoaded: true });
      // Cache to storage for fast restore on next open
      ext?.storage?.local?.set({
        cachedModelLists: { cloud: mdlCloudList, local: mdlLocalList, chatgpt: mdlChatgptList, direct: mdlDirectList },
      });
      // Prune stale feature model assignments
      const allAvailable = [...mdlCloudList, ...mdlLocalList, ...mdlChatgptList, ...mdlDirectList];
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
      if (s.mdlListsLoaded && (s.mdlCloudList.length || s.mdlLocalList.length || s.mdlChatgptList.length || s.mdlDirectList.length)) return;
      let cloud: string[] = [];
      let local: string[] = [];
      let chatgpt: string[] = [];
      let direct: string[] = [];

      // 1. Try Ollama for local/cloud models
      try {
        const d = await fetch('http://localhost:11434/api/tags', { signal: AbortSignal.timeout(3000) }).then(r => r.json());
        const all: string[] = (d.models || []).map((m: any) => m.name);
        cloud = all.filter(n => n.includes(':cloud'));
        local = all.filter(n => !n.includes(':cloud'));
      } catch { /* Ollama not available */ }

      // 2. Try backend for models (also gets ChatGPT + direct API lists)
      try {
        const d = await fetch(`${HTTP}/api/models/available`, { signal: AbortSignal.timeout(3000), headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).then(r => r.json());
        // Merge backend models with Ollama models (Ollama may have more)
        const backendCloud = (d.cloud || []).map((m: any) => m.name || m);
        const backendLocal = (d.local || []).map((m: any) => m.name || m);
        chatgpt = (d.chatgpt || []).map((m: any) => m.name || m);
        direct = (d.direct_api || []).map((m: any) => m.name || m);
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

      get().setMdlLists(cloud, local, chatgpt, direct);
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

    // --- Conversation History (IndexedDB with chrome.storage fallback) ---

    loadConversationList: async () => {
      try {
        // Migrate from chrome.storage to IndexedDB on first load
        if (USE_IDB && !_idbMigrated) {
          _idbMigrated = true;
          await db.initDB();
          const old = await storageGet([CONV_LIST_KEY]);
          const oldConvs: ConversationMeta[] = old[CONV_LIST_KEY] || [];
          if (oldConvs.length > 0) {
            for (const meta of oldConvs) {
              const msgData = await storageGet([convStorageKey(meta.id)]);
              const msgs = msgData[convStorageKey(meta.id)] || [];
              await db.saveConversation(meta, msgs).catch(() => {});
            }
            console.log(`[Aura] Migrated ${oldConvs.length} conversations to IndexedDB`);
          }
        }

        let convs: ConversationMeta[];
        if (USE_IDB) {
          convs = await db.getAllConversations();
        } else {
          const data = await storageGet([CONV_LIST_KEY]);
          convs = data[CONV_LIST_KEY] || [];
        }

        const activeData = await storageGet([ACTIVE_CONV_KEY]);
        const activeId: string | null = activeData[ACTIVE_CONV_KEY] || null;
        set({ conversations: convs, historyLoaded: true });

        if (activeId && convs.some(c => c.id === activeId)) {
          await get().loadConversation(activeId);
        }
      } catch (err) {
        console.error('[Aura] Failed to load conversations:', err);
        set({ conversations: [], historyLoaded: true });
      }
    },

    saveCurrentConversation: async () => {
      const s = get();
      if (s.messages.length === 0) return;

      const convId = s.activeConversationId || crypto.randomUUID();
      if (!s.activeConversationId) set({ activeConversationId: convId });

      const firstUserMsg = s.messages.find(m => m.role === 'user');
      const title = firstUserMsg
        ? firstUserMsg.text.slice(0, 60) + (firstUserMsg.text.length > 60 ? '...' : '')
        : 'New conversation';

      const messagesToSave = s.messages.slice(-MAX_MESSAGES_PER_CONVERSATION).map(m => ({
        id: m.id,
        role: m.role,
        text: m.text,
        timestamp: m.timestamp,
        thinkingContent: m.thinkingContent,
      })) as Message[];

      // Preserve existing folder/pinned state
      const existing = s.conversations.find(c => c.id === convId);
      const meta: ConversationMeta = {
        id: convId,
        title,
        timestamp: Date.now(),
        messageCount: s.messages.length,
        folder: existing?.folder,
        pinned: existing?.pinned,
      };

      let convs = [...s.conversations];
      const idx = convs.findIndex(c => c.id === convId);
      if (idx >= 0) convs[idx] = meta; else convs.unshift(meta);

      if (convs.length > MAX_CONVERSATIONS) {
        const removed = convs.splice(MAX_CONVERSATIONS);
        for (const c of removed) {
          if (USE_IDB) db.deleteConversation(c.id).catch(() => {});
          else storageRemove([convStorageKey(c.id)]).catch(() => {});
        }
      }
      convs.sort((a, b) => {
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        return b.timestamp - a.timestamp;
      });

      set({ conversations: convs });

      if (USE_IDB) {
        await db.saveConversation(meta, messagesToSave);
      } else {
        await storageSet({
          [CONV_LIST_KEY]: convs,
          [ACTIVE_CONV_KEY]: convId,
          [convStorageKey(convId)]: messagesToSave,
        });
      }
      await storageSet({ [ACTIVE_CONV_KEY]: convId });
    },

    loadConversation: async (id: string) => {
      let messages: Message[] = [];
      if (USE_IDB) {
        const loaded = await db.loadConversation(id);
        messages = loaded?.messages || [];
      } else {
        const data = await storageGet([convStorageKey(id)]);
        messages = data[convStorageKey(id)] || [];
      }
      set({ messages, activeConversationId: id, activeStream: null, conversationId: null, pendingCtx: null });
      await storageSet({ [ACTIVE_CONV_KEY]: id });
      const s = get();
      if (s.wsReady && s.ws?.readyState === WebSocket.OPEN) {
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).catch(() => {});
      }
    },

    deleteConversation: async (id: string) => {
      const s = get();
      const convs = s.conversations.filter(c => c.id !== id);
      set({ conversations: convs });
      if (USE_IDB) {
        await db.deleteConversation(id);
      } else {
        await storageSet({ [CONV_LIST_KEY]: convs });
        await storageRemove([convStorageKey(id)]);
      }
      if (s.activeConversationId === id) {
        set({ messages: [], activeConversationId: null, conversationId: null });
        await storageRemove([ACTIVE_CONV_KEY]);
      }
    },

    clearAllHistory: async () => {
      const s = get();
      if (USE_IDB) {
        for (const c of s.conversations) await db.deleteConversation(c.id).catch(() => {});
      } else {
        const keys = s.conversations.map(c => convStorageKey(c.id));
        keys.push(CONV_LIST_KEY, ACTIVE_CONV_KEY);
        await storageRemove(keys);
      }
      set({ conversations: [], activeConversationId: null });
    },

    newConversation: async () => {
      const s = get();
      if (s.messages.length > 0) await s.saveCurrentConversation();
      set({ messages: [], activeStream: null, conversationId: null, activeConversationId: null, pendingCtx: null });
      await storageRemove([ACTIVE_CONV_KEY]);
      if (s.wsReady && s.ws?.readyState === WebSocket.OPEN) {
        fetch(`${HTTP}/api/chat/clear`, { method: 'POST', headers: API_KEY ? { 'X-API-Key': API_KEY } : {} }).catch(() => {});
      }
    },

    // --- Folder & Pin Management ---

    createFolder: async (name: string) => {
      const s = get();
      if (s.folders.includes(name)) return;
      const folders = [...s.folders, name];
      set({ folders });
      await storageSet({ [CONV_FOLDERS_KEY]: folders });
    },
    deleteFolder: async (name: string) => {
      const s = get();
      const folders = s.folders.filter(f => f !== name);
      const convs = s.conversations.map(c => c.folder === name ? { ...c, folder: undefined } : c);
      set({ folders, conversations: convs });
      await storageSet({ [CONV_FOLDERS_KEY]: folders });
      if (USE_IDB) {
        for (const c of convs.filter(c => !c.folder)) await db.saveConversationMeta(c).catch(() => {});
      }
    },
    renameFolder: async (oldName: string, newName: string) => {
      const s = get();
      if (!s.folders.includes(oldName) || s.folders.includes(newName)) return;
      const folders = s.folders.map(f => f === oldName ? newName : f);
      const convs = s.conversations.map(c => c.folder === oldName ? { ...c, folder: newName } : c);
      set({ folders, conversations: convs });
      await storageSet({ [CONV_FOLDERS_KEY]: folders });
      if (USE_IDB) {
        for (const c of convs.filter(c => c.folder === newName)) await db.saveConversationMeta(c).catch(() => {});
      }
    },
    moveToFolder: async (convId: string, folder?: string) => {
      const s = get();
      const convs = s.conversations.map(c => c.id === convId ? { ...c, folder } : c);
      set({ conversations: convs });
      const conv = convs.find(c => c.id === convId);
      if (conv) {
        if (USE_IDB) await db.saveConversationMeta(conv).catch(() => {});
        else await storageSet({ [CONV_LIST_KEY]: convs });
      }
    },
    pinConversation: async (convId: string, pinned: boolean) => {
      const s = get();
      const convs = s.conversations.map(c => c.id === convId ? { ...c, pinned } : c);
      convs.sort((a, b) => {
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        return b.timestamp - a.timestamp;
      });
      set({ conversations: convs });
      const conv = convs.find(c => c.id === convId);
      if (conv) {
        if (USE_IDB) await db.saveConversationMeta(conv).catch(() => {});
        else await storageSet({ [CONV_LIST_KEY]: convs });
      }
    },
  };
});
