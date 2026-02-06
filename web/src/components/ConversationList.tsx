import { useState, useEffect, useCallback, useRef } from 'react';
import { useChatStore } from '../store/chatStore';
import type { Conversation, Message } from '../types';
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
} from '@heroicons/react/24/outline';

const API_BASE = '/api/chat';

/** Group conversations by date */
function groupByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const now = Date.now() / 1000;
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
    setIsLoading,
  } = useChatStore();

  const [collapsed, setCollapsed] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
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

  // Fetch on mount
  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

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

  // New Chat
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

  // Switch conversation
  const handleSwitch = async (id: string) => {
    if (id === currentConversationId) return;
    try {
      const res = await fetch(`${API_BASE}/conversations/${id}/switch`, { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setCurrentConversationId(data.id);
        // Replace messages with the switched conversation's messages
        clearMessages();
        if (data.messages && data.messages.length > 0) {
          for (const msg of data.messages) {
            addMessage({
              role: msg.role,
              content: msg.content,
            });
          }
        }
        await fetchConversations();
      }
    } catch (e) {
      console.error('[ConversationList] Switch error:', e);
    }
  };

  // Delete conversation
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
          // Brief success indicator (will fade)
          setTimeout(() => setSavingToMemory(null), 1500);
        } else {
          console.error('[ConversationList] Save to memory failed:', data.error);
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

  const grouped = groupByDate(conversations);
  const groupOrder = ['Today', 'Yesterday', 'Previous 7 Days', 'Older'];

  return (
    <div className="select-none">
      {/* New Chat button */}
      <button
        onClick={handleNewChat}
        className="w-full flex items-center gap-2 px-3 py-2.5 mb-2 text-sm font-medium text-chat-text bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 rounded-lg transition-all duration-200 group"
      >
        <PlusIcon className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
        <span>New Chat</span>
      </button>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between px-2 py-1.5 text-xs text-chat-text-secondary hover:text-chat-text transition-colors"
      >
        <span className="uppercase tracking-wider font-medium">Conversations ({conversations.length})</span>
        {collapsed ? <ChevronDownIcon className="w-3.5 h-3.5" /> : <ChevronUpIcon className="w-3.5 h-3.5" />}
      </button>

      {/* Conversation list */}
      {!collapsed && (
        <div className="max-h-[200px] overflow-y-auto space-y-0.5 pr-1 scrollbar-thin scrollbar-thumb-chat-border scrollbar-track-transparent">
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
                    className={`group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-all duration-150 ${
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
                          className="flex-1 bg-chat-input border border-purple-500/50 rounded px-1.5 py-0.5 text-xs text-chat-text outline-none"
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

                    {/* Hover actions */}
                    {editingId !== conv.id && (
                      <div className="hidden group-hover:flex items-center gap-0.5 flex-shrink-0">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleRenameStart(conv.id, conv.title); }}
                          className="p-1 text-chat-text-secondary/50 hover:text-chat-text rounded transition-colors"
                          title="Rename"
                        >
                          <PencilIcon className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleSaveToMemory(conv.id); }}
                          className="p-1 text-chat-text-secondary/50 hover:text-purple-400 rounded transition-colors"
                          title="Save to Memory"
                        >
                          <BookmarkIcon className="w-3 h-3" />
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDelete(conv.id); }}
                          className="p-1 text-chat-text-secondary/50 hover:text-red-400 rounded transition-colors"
                          title="Delete"
                        >
                          <TrashIcon className="w-3 h-3" />
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      )}

      {/* Right-click context menu */}
      {contextMenuId && (
        <div
          ref={contextMenuRef}
          className="fixed z-[100] bg-chat-sidebar border border-chat-border rounded-lg shadow-xl py-1 min-w-[160px]"
          style={{ left: contextMenuPos.x, top: contextMenuPos.y }}
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
          >
            <TrashIcon className="w-4 h-4" />
            Delete
          </button>
        </div>
      )}
    </div>
  );
}
