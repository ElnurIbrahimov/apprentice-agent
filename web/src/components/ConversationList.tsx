import { useState, useEffect, useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import { toast } from './Toast';
import type { Conversation } from '../types';
import {
  PlusIcon,
  TrashIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
  ChatBubbleLeftIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  BookmarkIcon,
  MagnifyingGlassIcon,
} from '@heroicons/react/24/outline';

const API_BASE = '/api/chat';

/** Group conversations by date */
function groupByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const todayStart = new Date();
  todayStart.setHours(0, 0, 0, 0);
  const todayTs = todayStart.getTime() / 1000;
  const yesterdayTs = todayTs - 86400;
  const weekAgoTs = todayTs - 7 * 86400;

  const groups: Record<string, Conversation[]> = {};

  for (const conv of conversations) {
    const ts = conv.updated_at;
    let group: string;
    if (ts >= todayTs) group = 'Today';
    else if (ts >= yesterdayTs) group = 'Yesterday';
    else if (ts >= weekAgoTs) group = 'Previous 7 Days';
    else group = 'Older';

    if (!groups[group]) groups[group] = [];
    groups[group].push(conv);
  }

  return groups;
}

export function ConversationList() {
  const {
    conversations,
    setConversations,
    currentConversationId,
    setCurrentConversationId,
    clearMessages,
    addMessage,
    setIsSwitchingConversation,
  } = useChatStore();

  const [collapsed, setCollapsed] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [msgSearchResults, setMsgSearchResults] = useState<Array<{
    conversation_id: string;
    conversation_title: string;
    role: string;
    snippet: string;
    timestamp: number;
  }>>([]);
  const searchDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [confirmingDeleteId, setConfirmingDeleteId] = useState<string | null>(null);
  const confirmDeleteTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [contextMenuId, setContextMenuId] = useState<string | null>(null);
  const [contextMenuPos, setContextMenuPos] = useState({ x: 0, y: 0 });
  const [savingToMemory, setSavingToMemory] = useState<string | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  // Fetch conversations list
  const fetchConversations = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`);
      if (res.ok) {
        const data: Conversation[] = await res.json();
        setConversations(data);
        // Set current conversation ID from active flag
        const active = data.find((c) => c.is_active);
        if (active && !currentConversationId) {
          setCurrentConversationId(active.id);
        }
      }
    } catch (e) {
      console.error('[ConversationList] Fetch error:', e);
    }
  }, [setConversations, setCurrentConversationId, currentConversationId]);

  // Load messages for a conversation from the switch endpoint
  const loadConversationMessages = useCallback(async (id: string) => {
    setIsSwitchingConversation(true);
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}/switch`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCurrentConversationId(data.id);
        clearMessages();
        if (data.messages && data.messages.length > 0) {
          for (const msg of data.messages) {
            addMessage({
              role: msg.role,
              content: msg.content,
            });
          }
        }
      }
    } catch (e) {
      console.error('[ConversationList] Load messages error:', e);
    } finally {
      setIsSwitchingConversation(false);
    }
  }, [setCurrentConversationId, clearMessages, addMessage, setIsSwitchingConversation]);

  // Fetch on mount
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  // Ref to avoid stale closure on handleNewChat in the event listener
  const handleNewChatRef = useRef<() => Promise<void>>();

  // Keyboard shortcut: Ctrl+N new chat
  useEffect(() => {
    const handler = () => handleNewChatRef.current?.();
    document.addEventListener('aura:new-chat', handler);
    return () => document.removeEventListener('aura:new-chat', handler);
  }, []);

  // Message-level search with 300ms debounce
  useEffect(() => {
    if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current);
    if (searchQuery.length >= 2) {
      searchDebounceRef.current = setTimeout(async () => {
        try {
          const res = await fetch(`${API_BASE}/conversations/search?q=${encodeURIComponent(searchQuery)}`);
          if (res.ok) {
            const data = await res.json();
            setMsgSearchResults(data.results || []);
          }
        } catch {
          setMsgSearchResults([]);
        }
      }, 300);
    } else {
      setMsgSearchResults([]);
    }
    return () => { if (searchDebounceRef.current) clearTimeout(searchDebounceRef.current); };
  }, [searchQuery]);

  // Clear delete-confirm timeout on unmount to prevent setState on unmounted component
  useEffect(() => {
    return () => { if (confirmDeleteTimeout.current) clearTimeout(confirmDeleteTimeout.current); };
  }, []);

  // Load messages for the active conversation on initial mount
  const initialLoadDone = useRef(false);
  useEffect(() => {
    if (!initialLoadDone.current && currentConversationId && conversations.length > 0) {
      const active = conversations.find((c) => c.id === currentConversationId);
      if (active && active.message_count > 0) {
        initialLoadDone.current = true;
        loadConversationMessages(currentConversationId);
      }
    }
  }, [currentConversationId, conversations, loadConversationMessages]);

  // Close context menu on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenuId(null);
      }
    };
    if (contextMenuId) {
      document.addEventListener('mousedown', handleClick);
      return () => document.removeEventListener('mousedown', handleClick);
    }
  }, [contextMenuId]);

  // Focus edit input
  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingId]);

  // New Chat — also update ref so the event listener always sees the latest version
  const handleNewChat = async () => {
    try {
      const res = await fetch(`${API_BASE}/conversations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
      if (res.ok) {
        const data = await res.json();
        setCurrentConversationId(data.id);
        clearMessages();
        await fetchConversations();
      }
    } catch (e) {
      console.error('[ConversationList] Create error:', e);
    }
  };
  handleNewChatRef.current = handleNewChat;

  // Switch conversation
  const handleSwitch = async (id: string) => {
    if (id === currentConversationId) return;
    await loadConversationMessages(id);
    await fetchConversations();
  };

  // Delete conversation with inline confirmation
  const handleDeleteClick = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (confirmingDeleteId === id) {
      // Already confirming — execute delete
      if (confirmDeleteTimeout.current) clearTimeout(confirmDeleteTimeout.current);
      setConfirmingDeleteId(null);
      handleDelete(id);
    } else {
      // First click — enter confirm state, auto-cancel after 3s
      if (confirmDeleteTimeout.current) clearTimeout(confirmDeleteTimeout.current);
      setConfirmingDeleteId(id);
      confirmDeleteTimeout.current = setTimeout(() => setConfirmingDeleteId(null), 3000);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}`, { method: 'DELETE' });
      if (res.ok) {
        const data = await res.json();
        if (id === currentConversationId) {
          // Switched to a new active conversation
          if (data.new_active_id) {
            // Re-fetch to get the new state
            await fetchConversations();
            // Switch to the new active
            await handleSwitch(data.new_active_id);
          }
        } else {
          await fetchConversations();
        }
      }
    } catch (e) {
      console.error('[ConversationList] Delete error:', e);
    }
    setContextMenuId(null);
  };

  // Rename conversation
  const handleRenameStart = (id: string, currentTitle: string) => {
    setEditingId(id);
    setEditTitle(currentTitle);
    setContextMenuId(null);
  };

  const handleRenameSubmit = async () => {
    if (!editingId || !editTitle.trim()) {
      setEditingId(null);
      return;
    }
    try {
      await fetch(`${API_BASE}/conversations/${editingId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: editTitle.trim() }),
      });
      await fetchConversations();
    } catch (e) {
      console.error('[ConversationList] Rename error:', e);
    }
    setEditingId(null);
  };

  // Save to memory
  const handleSaveToMemory = async (id: string) => {
    setSavingToMemory(id);
    setContextMenuId(null);
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}/save-to-memory`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          toast.success('Saved to memory');
          setTimeout(() => setSavingToMemory(null), 1500);
        } else {
          toast.error('Failed to save', data.error);
          setSavingToMemory(null);
        }
      }
    } catch (e) {
      console.error('[ConversationList] Save to memory error:', e);
      setSavingToMemory(null);
    }
  };

  // Context menu
  const handleContextMenu = (e: React.MouseEvent, id: string) => {
    e.preventDefault();
    e.stopPropagation();
    setContextMenuId(id);
    setContextMenuPos({ x: e.clientX, y: e.clientY });
  };

  const filtered = searchQuery.trim()
    ? conversations.filter(c =>
        c.title.toLowerCase().includes(searchQuery.toLowerCase().trim())
      )
    : conversations;
  const grouped = groupByDate(filtered);
  const groupOrder = ['Today', 'Yesterday', 'Previous 7 Days', 'Older'];

  return (
    <div className="select-none">
      {/* New Chat button */}
      <button
        onClick={handleNewChat}
        aria-label="New conversation"
        className="w-full flex items-center gap-2 px-3 py-2.5 mb-2 text-sm font-medium text-chat-text bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 rounded-lg transition-all duration-200 group"
      >
        <PlusIcon className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
        <span>New Chat</span>
      </button>

      {/* Conversation search */}
      <div className="relative mb-2">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search..."
          className="w-full px-3 py-2 pl-8 text-[16px] sm:text-xs bg-chat-input border border-chat-border/50 rounded-lg text-chat-text placeholder-chat-text-secondary/50 outline-none focus:border-purple-500/50 transition-colors"
        />
        <MagnifyingGlassIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-chat-text-secondary/50 pointer-events-none" />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery('')}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-chat-text-secondary/50 hover:text-chat-text"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between px-2 py-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
      >
        <span className="uppercase tracking-wider font-medium">Conversations ({filtered.length})</span>
        {collapsed ? <ChevronDownIcon className="w-3.5 h-3.5" /> : <ChevronUpIcon className="w-3.5 h-3.5" />}
      </button>

      {/* Conversation list */}
      {!collapsed && (
        <div className="max-h-[280px] overflow-y-auto space-y-0.5 pr-1 scrollbar-thin scrollbar-thumb-chat-border scrollbar-track-transparent">
          {groupOrder.map((group) => {
            const items = grouped[group];
            if (!items || items.length === 0) return null;
            return (
              <div key={group}>
                <div className="px-2 py-1 text-[10px] text-chat-text-secondary/60 uppercase tracking-wider font-medium">
                  {group}
                </div>
                {items.map((conv) => (
                  <div
                    key={conv.id}
                    onClick={() => handleSwitch(conv.id)}
                    onContextMenu={(e) => handleContextMenu(e, conv.id)}
                    role="button"
                    tabIndex={0}
                    className={`group flex items-center gap-2 px-2.5 py-3 rounded-lg cursor-pointer transition-all duration-150 ${
                      conv.id === currentConversationId
                        ? 'bg-purple-600/20 border border-purple-500/30'
                        : 'hover:bg-chat-assistant/30 border border-transparent'
                    }`}
                  >
                    <ChatBubbleLeftIcon className={`w-3.5 h-3.5 flex-shrink-0 ${
                      conv.id === currentConversationId ? 'text-purple-400' : 'text-chat-text-secondary/50'
                    }`} />

                    {editingId === conv.id ? (
                      <div className="flex-1 flex items-center gap-1">
                        <input
                          ref={editInputRef}
                          value={editTitle}
                          onChange={(e) => setEditTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRenameSubmit();
                            if (e.key === 'Escape') setEditingId(null);
                          }}
                          className="flex-1 bg-chat-input border border-purple-500/50 rounded px-1.5 py-1 text-[16px] sm:text-xs text-chat-text outline-none"
                        />
                        <button onClick={handleRenameSubmit} className="p-0.5 text-green-400 hover:text-green-300">
                          <CheckIcon className="w-3.5 h-3.5" />
                        </button>
                        <button onClick={() => setEditingId(null)} className="p-0.5 text-red-400 hover:text-red-300">
                          <XMarkIcon className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div className="flex-1 min-w-0">
                        <div className={`text-xs truncate ${
                          conv.id === currentConversationId ? 'text-chat-text font-medium' : 'text-chat-text-secondary'
                        }`}>
                          {conv.title}
                        </div>
                        {conv.message_count > 0 && (
                          <div className="text-[10px] text-chat-text-secondary/40 truncate">
                            {conv.message_count} msgs
                          </div>
                        )}
                      </div>
                    )}

                    {/* Saving to memory indicator */}
                    {savingToMemory === conv.id && (
                      <span className="text-[10px] text-green-400 animate-pulse">Saved!</span>
                    )}

                    {/* Actions — visible on hover (desktop) or always (touch) */}
                    {editingId !== conv.id && (
                      <div className="flex items-center gap-0.5 flex-shrink-0 opacity-50 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleRenameStart(conv.id, conv.title); }}
                          className="p-1.5 text-chat-text-secondary/50 hover:text-chat-text rounded transition-colors"
                          title="Rename"
                        >
                          <PencilIcon className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleSaveToMemory(conv.id); }}
                          className="p-1.5 text-chat-text-secondary/50 hover:text-purple-400 rounded transition-colors"
                          title="Save to Memory"
                        >
                          <BookmarkIcon className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => handleDeleteClick(e, conv.id)}
                          className={`p-1 rounded transition-colors text-xs ${
                            confirmingDeleteId === conv.id
                              ? 'text-red-400 bg-red-500/20 px-1.5 font-medium'
                              : 'text-chat-text-secondary/50 hover:text-red-400'
                          }`}
                          title={confirmingDeleteId === conv.id ? 'Click again to confirm' : 'Delete'}
                        >
                          {confirmingDeleteId === conv.id ? 'Sure?' : <TrashIcon className="w-3 h-3" />}
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
          {filtered.length === 0 && (
            <div className="text-center py-6 text-chat-text-secondary/60 text-xs">
              {searchQuery ? 'No conversations match your search' : 'No conversations yet'}
            </div>
          )}
        </div>
      )}

      {/* Message search results */}
      {msgSearchResults.length > 0 && (
        <div className="mt-2">
          <div className="px-2 py-1 text-[10px] text-chat-text-secondary/60 uppercase tracking-wider font-medium">
            Messages ({msgSearchResults.length})
          </div>
          <div className="space-y-0.5 max-h-[150px] overflow-y-auto pr-1">
            {msgSearchResults.map((result, idx) => (
              <div
                key={idx}
                onClick={() => handleSwitch(result.conversation_id)}
                className="px-2.5 py-2 rounded-lg cursor-pointer hover:bg-chat-assistant/30 transition-all"
              >
                <div className="text-[10px] text-purple-400/70 truncate mb-0.5">{result.conversation_title}</div>
                <div className="text-xs text-chat-text-secondary/70 line-clamp-2 italic">
                  {result.snippet}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Right-click context menu */}
      {contextMenuId && (
        <div
          ref={contextMenuRef}
          className="fixed z-[100] bg-chat-sidebar border border-chat-border rounded-lg shadow-xl py-1 min-w-[160px]"
          style={{ left: Math.min(contextMenuPos.x, window.innerWidth - 180), top: Math.min(contextMenuPos.y, window.innerHeight - 150) }}
        >
          <button
            onClick={() => {
              const conv = conversations.find((c) => c.id === contextMenuId);
              if (conv) handleRenameStart(conv.id, conv.title);
            }}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-chat-text hover:bg-chat-assistant/50 transition-colors"
          >
            <PencilIcon className="w-4 h-4" />
            Rename
          </button>
          <button
            onClick={() => contextMenuId && handleSaveToMemory(contextMenuId)}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-chat-text hover:bg-chat-assistant/50 transition-colors"
          >
            <BookmarkIcon className="w-4 h-4" />
            Save to Memory
          </button>
          <div className="border-t border-chat-border/50 my-1" />
          <button
            onClick={() => contextMenuId && handleDelete(contextMenuId)}
            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-red-600/20 transition-colors"
            aria-label="Delete conversation"
          >
            <TrashIcon className="w-4 h-4" />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
