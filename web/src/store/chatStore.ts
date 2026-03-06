import { create } from 'zustand';
import type { Message, MoodState, StatusResponse, ConnectionStatus, Conversation } from '../types';

interface ChatState {
  // Messages
  messages: Message[];
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, content: string) => void;
  appendToMessage: (id: string, chunk: string) => void;
  setMessageStreaming: (id: string, isStreaming: boolean) => void;
  setMessageModelUsed: (id: string, model: string) => void;
  clearMessages: () => void;

  // Connection
  connectionStatus: ConnectionStatus;
  setConnectionStatus: (status: ConnectionStatus) => void;

  // Status
  mood: MoodState | null;
  setMood: (mood: MoodState | null) => void;
  status: StatusResponse | null;
  setStatus: (status: StatusResponse | null) => void;

  // Model selection
  selectedModel: string | null; // null = auto (let AURA decide)
  availableModels: string[];
  setSelectedModel: (model: string | null) => void;
  setAvailableModels: (models: string[]) => void;

  // UI State
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;

  // Conversations
  conversations: Conversation[];
  currentConversationId: string | null;
  setConversations: (conversations: Conversation[]) => void;
  setCurrentConversationId: (id: string | null) => void;

  // Error handling
  error: string | null;
  setError: (error: string | null) => void;

  // Tool status
  toolStatus: { name: string; action: string } | null;
  setToolStatus: (s: { name: string; action: string } | null) => void;

  // Conversation switching
  isSwitchingConversation: boolean;
  setIsSwitchingConversation: (val: boolean) => void;
}

const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export const useChatStore = create<ChatState>((set, get) => ({
  // Messages
  messages: [],

  addMessage: (message) => {
    const id = generateId();
    const newMessage: Message = {
      ...message,
      id,
      timestamp: Date.now(),
    };
    set((state) => ({
      messages: [...state.messages, newMessage],
    }));
    return id;
  },

  updateMessage: (id, content) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content } : msg
      ),
    }));
  },

  appendToMessage: (id, chunk) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, content: msg.content + chunk } : msg
      ),
    }));
  },

  setMessageStreaming: (id, isStreaming) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, isStreaming } : msg
      ),
    }));
  },

  setMessageModelUsed: (id, model) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, model_used: model } : msg
      ),
    }));
  },

  clearMessages: () => set({ messages: [] }),

  // Connection
  connectionStatus: 'disconnected',
  setConnectionStatus: (status) => set({ connectionStatus: status }),

  // Status
  mood: null,
  setMood: (mood) => set({ mood }),
  status: null,
  setStatus: (status) => set({ status, mood: status?.mood || get().mood }),

  // Model selection
  selectedModel: null, // null = auto
  availableModels: [],
  setSelectedModel: (model) => set({ selectedModel: model }),
  setAvailableModels: (models) => set({ availableModels: models }),

  // UI State
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  sidebarOpen: true,
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  // Conversations
  conversations: [],
  currentConversationId: null,
  setConversations: (conversations) => set({ conversations }),
  setCurrentConversationId: (id) => set({ currentConversationId: id }),

  // Error handling
  error: null,
  setError: (error) => set({ error }),

  // Tool status
  toolStatus: null,
  setToolStatus: (s) => set({ toolStatus: s }),

  // Conversation switching
  isSwitchingConversation: false,
  setIsSwitchingConversation: (val) => set({ isSwitchingConversation: val }),
}));
