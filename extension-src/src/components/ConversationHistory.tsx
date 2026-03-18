import React, { useState, useMemo, useRef, useEffect } from 'react';
import { Search, Trash2, X, MessageSquare, Clock } from 'lucide-react';
import { useStore } from '../store';
import type { ConversationMeta } from '../types';

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  return new Date(ts).toLocaleDateString();
}

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function ConversationHistory({ open, onClose }: Props) {
  const { conversations, activeConversationId, loadConversation, deleteConversation, clearAllHistory } = useStore();
  const [search, setSearch] = useState('');
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Focus search on open
  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 100);
      setSearch('');
      setConfirmClearAll(false);
    }
  }, [open]);

  // Close on click outside
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, onClose]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  const filtered = useMemo(() => {
    if (!search.trim()) return conversations;
    const q = search.toLowerCase();
    return conversations.filter(c => c.title.toLowerCase().includes(q));
  }, [conversations, search]);

  const handleLoad = async (id: string) => {
    await loadConversation(id);
    onClose();
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await deleteConversation(id);
  };

  const handleClearAll = async () => {
    if (!confirmClearAll) {
      setConfirmClearAll(true);
      return;
    }
    await clearAllHistory();
    setConfirmClearAll(false);
  };

  if (!open) return null;

  return (
    <div
      ref={panelRef}
      className="conv-history-panel"
    >
      {/* Header */}
      <div className="conv-history-header">
        <div className="conv-history-title">
          <Clock size={14} />
          <span>Chat History</span>
        </div>
        <button onClick={onClose} className="conv-history-close" aria-label="Close history">
          <X size={14} />
        </button>
      </div>

      {/* Search */}
      <div className="conv-history-search">
        <Search size={12} className="conv-history-search-icon" />
        <input
          ref={searchRef}
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search conversations..."
          className="conv-history-search-input"
        />
        {search && (
          <button onClick={() => setSearch('')} className="conv-history-search-clear" aria-label="Clear search">
            <X size={10} />
          </button>
        )}
      </div>

      {/* List */}
      <div className="conv-history-list">
        {filtered.length === 0 ? (
          <div className="conv-history-empty">
            {conversations.length === 0
              ? 'No conversations yet'
              : 'No matches found'}
          </div>
        ) : (
          filtered.map(conv => (
            <button
              key={conv.id}
              onClick={() => handleLoad(conv.id)}
              className={`conv-history-item ${conv.id === activeConversationId ? 'conv-history-item-active' : ''}`}
            >
              <div className="conv-history-item-content">
                <div className="conv-history-item-title">{conv.title}</div>
                <div className="conv-history-item-meta">
                  <span>{relativeTime(conv.timestamp)}</span>
                  <span className="conv-history-item-dot" />
                  <MessageSquare size={10} />
                  <span>{conv.messageCount}</span>
                </div>
              </div>
              <button
                onClick={(e) => handleDelete(e, conv.id)}
                className="conv-history-item-delete"
                aria-label="Delete conversation"
                title="Delete"
              >
                <Trash2 size={12} />
              </button>
            </button>
          ))
        )}
      </div>

      {/* Footer */}
      {conversations.length > 0 && (
        <div className="conv-history-footer">
          <button
            onClick={handleClearAll}
            className={`conv-history-clear-btn ${confirmClearAll ? 'conv-history-clear-confirm' : ''}`}
          >
            <Trash2 size={11} />
            <span>{confirmClearAll ? 'Confirm clear all?' : 'Clear all history'}</span>
          </button>
        </div>
      )}
    </div>
  );
}
