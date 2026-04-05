import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { Search, Trash2, X, MessageSquare, Clock, Download, Pin, PinOff, Folder, FolderPlus, ChevronRight, MoreHorizontal } from 'lucide-react';
import { useStore } from '../store';
import { exportJSON, formatChatExport } from '../utils/exportUtils';
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

type TabId = 'all' | 'pinned' | string;

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function ConversationHistory({ open, onClose }: Props) {
  const {
    conversations, activeConversationId, loadConversation, deleteConversation, clearAllHistory,
    folders, createFolder, deleteFolder, pinConversation, moveToFolder,
  } = useStore();
  const [search, setSearch] = useState('');
  const [confirmClearAll, setConfirmClearAll] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('all');
  const [showFolderInput, setShowFolderInput] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [menuOpen, setMenuOpen] = useState<string | null>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 100);
      setSearch('');
      setConfirmClearAll(false);
      setMenuOpen(null);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  useEffect(() => {
    if (showFolderInput) setTimeout(() => folderInputRef.current?.focus(), 50);
  }, [showFolderInput]);

  const filtered = useMemo(() => {
    let list = conversations;
    // Filter by tab
    if (activeTab === 'pinned') list = list.filter(c => c.pinned);
    else if (activeTab !== 'all') list = list.filter(c => c.folder === activeTab);
    // Filter by search
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(c => c.title.toLowerCase().includes(q));
    }
    return list;
  }, [conversations, search, activeTab]);

  const handleLoad = async (id: string) => {
    try { await loadConversation(id); } catch {}
    onClose();
  };

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    setMenuOpen(null);
    try { await deleteConversation(id); } catch {}
  };

  const handleClearAll = async () => {
    if (!confirmClearAll) { setConfirmClearAll(true); return; }
    try { await clearAllHistory(); } catch {}
    setConfirmClearAll(false);
  };

  const handleExport = useCallback(() => {
    const messages = useStore.getState().messages;
    if (!messages?.length) return;
    const data = {
      exported: new Date().toISOString(),
      messages: formatChatExport(messages.map(m => ({ role: m.role, text: m.text || '', timestamp: m.timestamp }))),
    };
    exportJSON(data, `aura-chat-${Date.now()}.json`);
  }, []);

  const handleCreateFolder = () => {
    const name = newFolderName.trim();
    if (!name) return;
    createFolder(name);
    setNewFolderName('');
    setShowFolderInput(false);
  };

  const handlePin = (e: React.MouseEvent, conv: ConversationMeta) => {
    e.stopPropagation();
    setMenuOpen(null);
    pinConversation(conv.id, !conv.pinned);
  };

  const handleMove = (e: React.MouseEvent, convId: string, folder?: string) => {
    e.stopPropagation();
    setMenuOpen(null);
    moveToFolder(convId, folder);
  };

  if (!open) return null;

  return (
    <div ref={panelRef} className="conv-history-panel">
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

      {/* Tabs */}
      <div className="conv-history-tabs">
        <button
          className={`conv-history-tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >All</button>
        <button
          className={`conv-history-tab ${activeTab === 'pinned' ? 'active' : ''}`}
          onClick={() => setActiveTab('pinned')}
        >
          <Pin size={10} style={{ marginRight: 3 }} />
          Pinned
        </button>
        {folders.map(f => (
          <button
            key={f}
            className={`conv-history-tab ${activeTab === f ? 'active' : ''}`}
            onClick={() => setActiveTab(f)}
            onContextMenu={(e) => { e.preventDefault(); deleteFolder(f); }}
            title={`Right-click to delete "${f}"`}
          >
            <Folder size={10} style={{ marginRight: 3 }} />
            {f}
          </button>
        ))}
        {showFolderInput ? (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <input
              ref={folderInputRef}
              value={newFolderName}
              onChange={e => setNewFolderName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleCreateFolder(); if (e.key === 'Escape') setShowFolderInput(false); }}
              placeholder="Name..."
              className="conv-history-search-input"
              style={{ width: 80, padding: '2px 6px', fontSize: 11 }}
            />
          </div>
        ) : (
          <button
            className="conv-history-tab"
            onClick={() => setShowFolderInput(true)}
            title="New folder"
          >
            <FolderPlus size={12} />
          </button>
        )}
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
            {conversations.length === 0 ? 'No conversations yet' : 'No matches found'}
          </div>
        ) : (
          filtered.map(conv => (
            <button
              key={conv.id}
              onClick={() => handleLoad(conv.id)}
              className={`conv-history-item ${conv.id === activeConversationId ? 'conv-history-item-active' : ''}`}
            >
              <div className="conv-history-item-content">
                <div className="conv-history-item-title">
                  {conv.pinned && <Pin size={10} style={{ marginRight: 4, opacity: 0.6, flexShrink: 0 }} />}
                  <span>{conv.title}</span>
                </div>
                <div className="conv-history-item-meta">
                  <span>{relativeTime(conv.timestamp)}</span>
                  <span className="conv-history-item-dot" />
                  <MessageSquare size={10} />
                  <span>{conv.messageCount}</span>
                  {conv.folder && (
                    <>
                      <span className="conv-history-item-dot" />
                      <Folder size={9} />
                      <span>{conv.folder}</span>
                    </>
                  )}
                </div>
              </div>
              <div className="conv-history-item-actions">
                {/* Context menu button */}
                <button
                  onClick={(e) => { e.stopPropagation(); setMenuOpen(menuOpen === conv.id ? null : conv.id); }}
                  className="conv-history-item-delete"
                  aria-label="More actions"
                  title="More"
                >
                  <MoreHorizontal size={12} />
                </button>
                {/* Dropdown menu */}
                {menuOpen === conv.id && (
                  <div className="conv-history-dropdown" onClick={e => e.stopPropagation()}>
                    <button onClick={(e) => handlePin(e, conv)}>
                      {conv.pinned ? <><PinOff size={11} /> Unpin</> : <><Pin size={11} /> Pin</>}
                    </button>
                    {folders.length > 0 && (
                      <div className="conv-history-dropdown-sub">
                        <button style={{ opacity: 0.7, fontSize: 10, pointerEvents: 'none' }}>
                          <ChevronRight size={9} /> Move to folder
                        </button>
                        {conv.folder && (
                          <button onClick={(e) => handleMove(e, conv.id, undefined)}>
                            <X size={11} /> Remove from folder
                          </button>
                        )}
                        {folders.map(f => (
                          <button key={f} onClick={(e) => handleMove(e, conv.id, f)} style={{ paddingLeft: 20 }}>
                            <Folder size={10} /> {f}
                          </button>
                        ))}
                      </div>
                    )}
                    <button onClick={(e) => handleDelete(e, conv.id)} style={{ color: 'var(--dg)' }}>
                      <Trash2 size={11} /> Delete
                    </button>
                  </div>
                )}
              </div>
            </button>
          ))
        )}
      </div>

      {/* Footer */}
      {conversations.length > 0 && (
        <div className="conv-history-footer">
          <button onClick={handleExport} className="conv-history-clear-btn" title="Export current chat as JSON">
            <Download size={11} />
            <span>Export chat</span>
          </button>
          <button
            onClick={handleClearAll}
            className={`conv-history-clear-btn ${confirmClearAll ? 'conv-history-clear-confirm' : ''}`}
          >
            <Trash2 size={11} />
            <span>{confirmClearAll ? 'Confirm clear all?' : 'Clear all'}</span>
          </button>
        </div>
      )}
    </div>
  );
}
