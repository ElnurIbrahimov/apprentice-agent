import { useState, useEffect } from 'react';
import { usePolling } from '../hooks/usePolling';
import type { AMEMStats, AMEMNote, AMEMSearchResult } from '../types';
import {
  ArrowPathIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  LinkIcon,
  TagIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

const CATEGORY_COLORS: Record<string, string> = {
  general: 'bg-gray-600',
  episodic: 'bg-blue-600',
  semantic: 'bg-green-600',
  procedural: 'bg-purple-600',
  fact: 'bg-amber-600',
};

const CATEGORY_ICONS: Record<string, string> = {
  general: '📝',
  episodic: '📅',
  semantic: '🧠',
  procedural: '⚙️',
  fact: '✓',
};

export function AMEMPanel() {
  const [stats, setStats] = useState<AMEMStats | null>(null);
  const [notes, setNotes] = useState<AMEMNote[]>([]);
  const [searchResults, setSearchResults] = useState<AMEMSearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'notes' | 'search' | 'add'>('notes');
  const [selectedNote, setSelectedNote] = useState<AMEMNote | null>(null);

  const [actionError, setActionError] = useState<string | null>(null);

  // Add form state
  const [newContent, setNewContent] = useState('');
  const [newTags, setNewTags] = useState('');
  const [newCategory, setNewCategory] = useState('general');
  const [addLoading, setAddLoading] = useState(false);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/amem/stats');
      if (res.ok) {
        const data = await res.json();
        setStats(data);
      }
    } catch (e) {
      console.error('Failed to fetch A-MEM stats:', e);
    }
  };

  const fetchNotes = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/amem/notes?limit=20');
      if (res.ok) {
        const data = await res.json();
        setNotes(data.notes || []);
      }
    } catch (e) {
      console.error('Failed to fetch A-MEM notes:', e);
    } finally {
      setLoading(false);
    }
  };

  const searchNotes = async () => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/amem/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, k: 10, follow_links: true }),
      });
      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
      }
    } catch {
      setActionError('Search failed');
    } finally {
      setLoading(false);
    }
  };

  const addMemory = async () => {
    if (!newContent.trim()) return;
    setAddLoading(true);
    try {
      const res = await fetch('/api/amem/remember', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: newContent,
          tags: newTags.split(',').map((t) => t.trim()).filter(Boolean),
          category: newCategory,
          importance: 0.6,
        }),
      });
      if (res.ok) {
        setNewContent('');
        setNewTags('');
        setActiveTab('notes');
        fetchNotes();
        fetchStats();
      }
    } catch {
      setActionError('Failed to add memory');
    } finally {
      setAddLoading(false);
    }
  };

  const consolidate = async () => {
    setLoading(true);
    try {
      await fetch('/api/amem/consolidate', { method: 'POST' });
      await fetchStats();
      await fetchNotes();
    } catch {
      setActionError('Failed to consolidate');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchNotes();
  }, []);

  usePolling(fetchStats, 10000);

  return (
    <div className="bg-chat-sidebar rounded-lg p-4">
      {actionError && (
        <div className="mb-2 text-xs text-red-400 bg-red-500/10 px-2 py-1 rounded cursor-pointer" onClick={() => setActionError(null)}>
          {actionError} (click to dismiss)
        </div>
      )}
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-chat-text font-medium flex items-center gap-2">
          <DocumentTextIcon className="w-5 h-5 text-emerald-400" />
          A-MEM
          <span className="text-xs text-chat-text-secondary">Zettelkasten Memory</span>
        </h3>
        <button
          onClick={() => { fetchStats(); fetchNotes(); }}
          className="p-1 text-chat-text-secondary hover:text-chat-text rounded"
          disabled={loading}
        >
          <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-1 mb-3">
          <div className="bg-chat-assistant rounded p-2 text-center">
            <div className="text-lg font-bold text-chat-text">{stats.total_notes}</div>
            <div className="text-xs text-chat-text-secondary">Notes</div>
          </div>
          <div className="bg-chat-assistant rounded p-2 text-center">
            <div className="text-lg font-bold text-chat-text">{stats.total_links}</div>
            <div className="text-xs text-chat-text-secondary">Links</div>
          </div>
          <div className="bg-chat-assistant rounded p-2 text-center">
            <div className="text-lg font-bold text-chat-text">{stats.total_boxes}</div>
            <div className="text-xs text-chat-text-secondary">Boxes</div>
          </div>
          <div className="bg-chat-assistant rounded p-2 text-center">
            <div className="text-lg font-bold text-emerald-400">{stats.has_embeddings}</div>
            <div className="text-xs text-chat-text-secondary">Embedded</div>
          </div>
        </div>
      )}

      {/* Categories */}
      {stats && stats.categories && Object.keys(stats.categories).length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {Object.entries(stats.categories).map(([cat, count]) => (
            <span
              key={cat}
              className={`text-xs px-2 py-0.5 rounded-full text-white ${CATEGORY_COLORS[cat] || 'bg-gray-500'}`}
            >
              {CATEGORY_ICONS[cat] || '📌'} {cat}: {count}
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-3 border-b border-chat-text-secondary/20 pb-2">
        <button
          onClick={() => setActiveTab('notes')}
          className={`px-3 py-1 text-xs rounded-t ${
            activeTab === 'notes'
              ? 'bg-emerald-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          Recent
        </button>
        <button
          onClick={() => setActiveTab('search')}
          className={`px-3 py-1 text-xs rounded-t ${
            activeTab === 'search'
              ? 'bg-emerald-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          Search
        </button>
        <button
          onClick={() => setActiveTab('add')}
          className={`px-3 py-1 text-xs rounded-t ${
            activeTab === 'add'
              ? 'bg-emerald-600 text-white'
              : 'text-chat-text-secondary hover:text-chat-text'
          }`}
        >
          + Add
        </button>
      </div>

      {/* Tab Content */}
      {activeTab === 'notes' && (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {notes.length === 0 ? (
            <div className="text-chat-text-secondary text-sm text-center py-4">
              No memories yet. Add some!
            </div>
          ) : (
            notes.map((note) => (
              <div
                key={note.id}
                className="bg-chat-assistant rounded p-2 cursor-pointer hover:bg-chat-assistant/80 transition-colors"
                onClick={() => setSelectedNote(selectedNote?.id === note.id ? null : note)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-xs text-chat-text line-clamp-2">
                      {note.content}
                    </div>
                    {selectedNote?.id === note.id && (
                      <div className="mt-2 space-y-1">
                        {note.context && (
                          <div className="text-xs text-emerald-400 italic">
                            {note.context}
                          </div>
                        )}
                        <div className="flex flex-wrap gap-1">
                          {note.keywords.slice(0, 5).map((kw, i) => (
                            <span key={i} className="text-xs bg-chat-sidebar px-1 rounded text-chat-text-secondary">
                              {kw}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span className={`text-xs px-1.5 py-0.5 rounded ${CATEGORY_COLORS[note.category] || 'bg-gray-500'} text-white`}>
                      {CATEGORY_ICONS[note.category] || '📌'}
                    </span>
                    {note.links > 0 && (
                      <span className="text-xs text-emerald-400 flex items-center gap-0.5">
                        <LinkIcon className="w-3 h-3" />
                        {note.links}
                      </span>
                    )}
                  </div>
                </div>
                {note.tags.length > 0 && (
                  <div className="flex items-center gap-1 mt-1">
                    <TagIcon className="w-3 h-3 text-chat-text-secondary" />
                    <div className="flex gap-1 overflow-hidden">
                      {note.tags.slice(0, 3).map((tag, i) => (
                        <span key={i} className="text-xs text-blue-400">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'search' && (
        <div>
          <div className="flex gap-2 mb-3">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && searchNotes()}
              placeholder="Search memories..."
              className="flex-1 bg-chat-assistant text-chat-text text-sm px-3 py-2 rounded border border-chat-text-secondary/20 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={searchNotes}
              disabled={loading || !searchQuery.trim()}
              className="px-3 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded text-sm flex items-center gap-1"
            >
              <MagnifyingGlassIcon className="w-4 h-4" />
            </button>
          </div>

          <div className="space-y-2 max-h-48 overflow-y-auto">
            {searchResults.map((result, i) => (
              <div key={i} className="bg-chat-assistant rounded p-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1">
                    <div className="text-xs text-chat-text">{result.content}</div>
                    {result.context && (
                      <div className="text-xs text-emerald-400 italic mt-1">{result.context}</div>
                    )}
                  </div>
                  <div className="flex flex-col items-end">
                    <span className="text-xs text-emerald-400 font-bold">
                      {Math.round(result.relevance * 100)}%
                    </span>
                    {result.hop > 0 && (
                      <span className="text-xs text-chat-text-secondary">
                        hop {result.hop}
                      </span>
                    )}
                  </div>
                </div>
                {result.keywords.length > 0 && (
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {result.keywords.slice(0, 4).map((kw, j) => (
                      <span key={j} className="text-xs bg-chat-sidebar px-1 rounded text-chat-text-secondary">
                        {kw}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'add' && (
        <div className="space-y-3">
          <div>
            <label className="text-xs text-chat-text-secondary block mb-1">Memory Content</label>
            <textarea
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              placeholder="Enter a memory to store..."
              className="w-full bg-chat-assistant text-chat-text text-sm px-3 py-2 rounded border border-chat-text-secondary/20 focus:outline-none focus:border-emerald-500 min-h-[80px] resize-none"
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-xs text-chat-text-secondary block mb-1">Tags (comma-separated)</label>
              <input
                type="text"
                value={newTags}
                onChange={(e) => setNewTags(e.target.value)}
                placeholder="tag1, tag2"
                className="w-full bg-chat-assistant text-chat-text text-sm px-3 py-2 rounded border border-chat-text-secondary/20 focus:outline-none focus:border-emerald-500"
              />
            </div>
            <div>
              <label className="text-xs text-chat-text-secondary block mb-1">Category</label>
              <select
                value={newCategory}
                onChange={(e) => setNewCategory(e.target.value)}
                className="w-full bg-chat-assistant text-chat-text text-sm px-3 py-2 rounded border border-chat-text-secondary/20 focus:outline-none focus:border-emerald-500"
              >
                <option value="general">General</option>
                <option value="episodic">Episodic (Events)</option>
                <option value="semantic">Semantic (Knowledge)</option>
                <option value="procedural">Procedural (How-to)</option>
                <option value="fact">Fact (Verified)</option>
              </select>
            </div>
          </div>

          <button
            onClick={addMemory}
            disabled={addLoading || !newContent.trim()}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded text-sm"
          >
            {addLoading ? (
              <ArrowPathIcon className="w-4 h-4 animate-spin" />
            ) : (
              <PlusIcon className="w-4 h-4" />
            )}
            Store Memory
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 mt-3 pt-3 border-t border-chat-text-secondary/20">
        <button
          onClick={consolidate}
          disabled={loading}
          className="flex-1 flex items-center justify-center gap-1 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white rounded text-xs"
        >
          <SparklesIcon className="w-3 h-3" />
          Consolidate
        </button>
      </div>
    </div>
  );
}
