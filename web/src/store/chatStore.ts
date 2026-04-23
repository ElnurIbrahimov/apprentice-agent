import { create } from 'zustand';
import type { Message, MoodState, StatusResponse, ConnectionStatus, Conversation, Citation, ToolTrace, FleetTask, ResearchProgress, ResearchProgressStep } from '../types';

interface ChatState {
  // Messages
  messages: Message[];
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  updateMessage: (id: string, content: string) => void;
  appendToMessage: (id: string, chunk: string) => void;
  setMessageStreaming: (id: string, isStreaming: boolean) => void;
  setMessageModelUsed: (id: string, model: string) => void;
  setCitationsForMessage: (id: string, citations: Citation[]) => void;
  appendToolTrace: (id: string, trace: ToolTrace) => void;
  removeMessagesFrom: (id: string) => void;
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

  // Suggestion chips
  suggestions: string[];
  setSuggestions: (suggestions: string[]) => void;
  clearSuggestions: () => void;

  // Fleet dashboard
  fleetData: { goal: string; tasks: FleetTask[]; totalElapsed: number } | null;
  setFleetData: (data: { goal: string; tasks: FleetTask[]; totalElapsed: number } | null) => void;
  clearFleetData: () => void;

  // Research progress
  researchProgress: ResearchProgress | null;
  addResearchStep: (step: ResearchProgressStep) => void;
  clearResearchProgress: () => void;

  // Citations panel
  citationsPanelOpen: boolean;
  setCitationsPanelOpen: (open: boolean) => void;
  toggleCitationsPanel: () => void;
  hoveredCitation: { id: number; messageId: string } | null;
  setHoveredCitation: (hovered: { id: number; messageId: string } | null) => void;
  activeCitationRef: { id: number; messageId: string } | null;
  setActiveCitationRef: (ref: { id: number; messageId: string } | null) => void;

  // Tool suggestions / prefill
  toolSuggestion: { toolId: string; label: string; reason: string } | null;
  setToolSuggestion: (s: { toolId: string; label: string; reason: string } | null) => void;
  toolPrefill: { toolId: string; query: string } | null;
  setToolPrefill: (p: { toolId: string; query: string } | null) => void;
  clearToolPrefill: () => void;

  // Forked-from-another-conversation tracking (ephemeral — cleared on next send or convo switch)
  forkedFrom: { conversationId: string | null; atMessageId: string; atMessageTimestamp: number } | null;
  forkFromMessage: (messageId: string) => boolean;
  clearForkedFrom: () => void;
}

const generateId = () => `msg_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;

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
    set((state) => {
      const updated = [...state.messages, newMessage];
      // Cap at 2000 messages to prevent unbounded memory growth
      const trimmed = updated.length > 2000 ? updated.slice(updated.length - 2000) : updated;
      return { messages: trimmed };
    });
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

  setCitationsForMessage: (id, citations) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, citations } : msg
      ),
    }));
  },

  appendToolTrace: (id, trace) => {
    set((state) => ({
      messages: state.messages.map((msg) =>
        msg.id === id ? { ...msg, toolTrace: [...(msg.toolTrace || []), trace] } : msg
      ),
    }));
  },

  removeMessagesFrom: (id) => {
    set((state) => {
      const idx = state.messages.findIndex((m) => m.id === id);
      if (idx < 0) return state;
      return { messages: state.messages.slice(0, idx) };
    });
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

  // Suggestion chips
  suggestions: [],
  setSuggestions: (suggestions) => set({ suggestions }),
  clearSuggestions: () => set({ suggestions: [] }),

  // Fleet dashboard
  fleetData: null,
  setFleetData: (data) => set({ fleetData: data }),
  clearFleetData: () => set({ fleetData: null }),

  // Research progress
  researchProgress: null,
  addResearchStep: (step) => set((state) => {
    const current = state.researchProgress;
    const isSynthesis = step.stage === 'synthesis';
    return {
      researchProgress: {
        active: !isSynthesis,
        stage: step.stage,
        steps: [...(current?.steps || []), step],
      },
    };
  }),
  clearResearchProgress: () => set({ researchProgress: null }),

  // Citations panel
  citationsPanelOpen: false,
  setCitationsPanelOpen: (open) => set({ citationsPanelOpen: open }),
  toggleCitationsPanel: () => set((state) => ({ citationsPanelOpen: !state.citationsPanelOpen })),
  hoveredCitation: null,
  setHoveredCitation: (hovered) => set({ hoveredCitation: hovered }),
  activeCitationRef: null,
  setActiveCitationRef: (ref) => set({ activeCitationRef: ref }),

  // Tool suggestions / prefill
  toolSuggestion: null,
  setToolSuggestion: (s) => set({ toolSuggestion: s }),
  toolPrefill: null,
  setToolPrefill: (p) => set({ toolPrefill: p }),
  clearToolPrefill: () => set({ toolPrefill: null }),

  // Forked-from tracking
  forkedFrom: null,
  forkFromMessage: (messageId) => {
    const state = get();
    const idx = state.messages.findIndex((m) => m.id === messageId);
    if (idx < 0) return false;
    const sliced = state.messages.slice(0, idx + 1);
    set({
      messages: sliced,
      currentConversationId: null,
      forkedFrom: {
        conversationId: state.currentConversationId,
        atMessageId: messageId,
        atMessageTimestamp: state.messages[idx].timestamp,
      },
      suggestions: [],
      toolSuggestion: null,
      researchProgress: null,
      fleetData: null,
    });
    return true;
  },
  clearForkedFrom: () => set({ forkedFrom: null }),
}));
