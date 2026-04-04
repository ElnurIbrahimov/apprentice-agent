import { useState, useEffect, useRef, useCallback } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import type { TabId } from '../types';
import type { ToolId } from './ToolLauncher';

type CreateSubTab = 'code' | 'webcreator' | 'image';

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  setActiveTab: (tab: TabId) => void;
  setToolsSubTab: (tab: ToolId) => void;
  setCreateSubTab: (tab: CreateSubTab) => void;
  toggleSidebar: () => void;
  newChat: () => void;
  toggleTheme: () => void;
}

interface Command {
  id: string;
  label: string;
  shortcut?: string;
  category: string;
  action: () => void;
}

function buildCommands(props: Omit<CommandPaletteProps, 'isOpen' | 'onClose'>): Command[] {
  const { setActiveTab, setToolsSubTab, setCreateSubTab, toggleSidebar, newChat, toggleTheme } = props;

  const tool = (id: ToolId, label: string): Command => ({
    id: `tool-${id}`,
    label,
    category: 'Tools',
    action: () => { setActiveTab('tools'); setToolsSubTab(id); },
  });

  return [
    // Navigation
    { id: 'nav-chat',     label: 'Go to Chat',     shortcut: 'Ctrl+1', category: 'Navigation', action: () => setActiveTab('chat') },
    { id: 'nav-create',   label: 'Go to Create',   shortcut: 'Ctrl+2', category: 'Navigation', action: () => setActiveTab('create') },
    { id: 'nav-tools',    label: 'Go to Tools',    shortcut: 'Ctrl+3', category: 'Navigation', action: () => setActiveTab('tools') },
    { id: 'nav-insights', label: 'Go to Insights', shortcut: 'Ctrl+4', category: 'Navigation', action: () => setActiveTab('insights') },
    { id: 'nav-settings', label: 'Go to Settings', shortcut: 'Ctrl+5', category: 'Navigation', action: () => setActiveTab('settings') },

    // Actions
    { id: 'new-chat',       label: 'New Chat',       shortcut: 'Ctrl+N',       category: 'Actions', action: newChat },
    { id: 'toggle-sidebar', label: 'Toggle Sidebar', shortcut: 'Ctrl+B',       category: 'Actions', action: toggleSidebar },
    { id: 'toggle-theme',   label: 'Toggle Theme',   shortcut: 'Ctrl+Shift+T', category: 'Actions', action: toggleTheme },
    { id: 'settings',       label: 'Open Settings',  shortcut: 'Ctrl+/',       category: 'Actions', action: () => setActiveTab('settings') },

    // Create sub-tabs
    { id: 'create-code',  label: 'Open Code Interpreter', category: 'Create', action: () => { setActiveTab('create'); setCreateSubTab('code'); } },
    { id: 'create-web',   label: 'Open Web Creator',      category: 'Create', action: () => { setActiveTab('create'); setCreateSubTab('webcreator'); } },
    { id: 'create-image', label: 'Open Image Generator',  category: 'Create', action: () => { setActiveTab('create'); setCreateSubTab('image'); } },

    // Tools sub-tabs
    tool('ask',       'Open Ask'),
    tool('search',    'Open Search'),
    tool('research',  'Open Research'),
    tool('agent',     'Open Agent'),
    tool('compare',   'Open Compare'),
    tool('write',     'Open Write'),
    tool('translate', 'Open Translate'),
    tool('summary',   'Open Summary'),
    tool('grammar',   'Open Grammar'),
    tool('math',      'Open Math'),
    tool('pdf',       'Open PDF'),
    tool('ocr',       'Open OCR'),
    tool('capture',   'Open Capture'),
    tool('youtube',   'Open YouTube'),
    tool('voice',     'Open Voice'),
    tool('record',    'Open Record'),
    tool('slides',    'Open Slides'),
    tool('wisebase',  'Open Wisebase'),
    tool('models',    'Open Models'),
  ];
}

function fuzzyMatch(text: string, query: string): boolean {
  const t = text.toLowerCase();
  const q = query.toLowerCase();
  let qi = 0;
  for (let i = 0; i < t.length && qi < q.length; i++) {
    if (t[i] === q[qi]) qi++;
  }
  return qi === q.length;
}

const CATEGORY_ORDER = ['Navigation', 'Actions', 'Create', 'Tools'] as const;

export function CommandPalette({
  isOpen,
  onClose,
  setActiveTab,
  setToolsSubTab,
  setCreateSubTab,
  toggleSidebar,
  newChat,
  toggleTheme,
}: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  const commands = buildCommands({ setActiveTab, setToolsSubTab, setCreateSubTab, toggleSidebar, newChat, toggleTheme });

  const filtered = query.trim()
    ? commands.filter(cmd => fuzzyMatch(cmd.label, query) || fuzzyMatch(cmd.category, query))
    : commands;

  const grouped = CATEGORY_ORDER.reduce<Partial<Record<string, Command[]>>>((acc, cat) => {
    const items = filtered.filter(c => c.category === cat);
    if (items.length) acc[cat] = items;
    return acc;
  }, {});

  const flatFiltered = CATEGORY_ORDER.flatMap(cat => grouped[cat] ?? []);

  const execute = useCallback((cmd: Command) => {
    cmd.action();
    onClose();
  }, [onClose]);

  // Reset state when opening
  useEffect(() => {
    if (isOpen) {
      setQuery('');
      setSelectedIndex(0);
      // Small delay so the portal has rendered before focusing
      const id = setTimeout(() => inputRef.current?.focus(), 10);
      return () => clearTimeout(id);
    }
  }, [isOpen]);

  // Clamp selection when list shrinks
  useEffect(() => {
    setSelectedIndex(prev => Math.min(prev, Math.max(0, flatFiltered.length - 1)));
  }, [flatFiltered.length]);

  // Keep selected item visible
  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest' });
  }, [selectedIndex]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(i => Math.min(i + 1, flatFiltered.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(i => Math.max(i - 1, 0));
        break;
      case 'Enter': {
        e.preventDefault();
        const cmd = flatFiltered[selectedIndex];
        if (cmd) execute(cmd);
        break;
      }
      case 'Escape':
        e.preventDefault();
        onClose();
        break;
    }
  }, [flatFiltered, selectedIndex, execute, onClose]);

  if (!isOpen) return null;

  let runningIndex = 0;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[250] bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Palette modal */}
      <div
        className="fixed z-[260] top-[12%] left-1/2 -translate-x-1/2 w-full max-w-lg rounded-xl border border-chat-border overflow-hidden"
        style={{
          background: 'var(--surface-2)',
          boxShadow: '0 24px 64px rgba(0,0,0,0.55)',
        }}
        onKeyDown={handleKeyDown}
        // Prevent backdrop click from firing when clicking inside the modal
        onClick={e => e.stopPropagation()}
      >
        {/* Search row */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-chat-border">
          <MagnifyingGlassIcon className="w-5 h-5 text-chat-text-secondary flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search commands..."
            value={query}
            onChange={e => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            className="flex-1 bg-transparent text-chat-text outline-none text-sm placeholder:text-chat-text-secondary"
          />
          <kbd className="hidden sm:inline-block px-1.5 py-0.5 rounded text-[11px] font-mono text-chat-text-secondary border border-chat-border">
            ESC
          </kbd>
        </div>

        {/* Results list */}
        <div className="max-h-[400px] overflow-y-auto py-1.5">
          {flatFiltered.length === 0 ? (
            <div className="px-4 py-10 text-center text-sm text-chat-text-secondary">
              No commands found
            </div>
          ) : (
            CATEGORY_ORDER.map(category => {
              const items = grouped[category];
              if (!items) return null;

              return (
                <div key={category}>
                  <div className="px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-widest text-chat-text-secondary select-none">
                    {category}
                  </div>

                  {items.map(cmd => {
                    const idx = runningIndex++;
                    const isSelected = idx === selectedIndex;

                    return (
                      <button
                        key={cmd.id}
                        ref={isSelected ? selectedRef : null}
                        onClick={() => execute(cmd)}
                        onMouseEnter={() => setSelectedIndex(idx)}
                        className={`w-full flex items-center justify-between gap-3 px-3 py-2 text-sm transition-colors text-left ${
                          isSelected
                            ? 'bg-chat-accent text-white'
                            : 'text-chat-text hover:bg-chat-assistant'
                        }`}
                      >
                        <span className="truncate">{cmd.label}</span>
                        {cmd.shortcut && (
                          <kbd
                            className={`flex-shrink-0 px-1.5 py-0.5 rounded text-[11px] font-mono border ${
                              isSelected
                                ? 'border-white/30 text-white/80 bg-white/10'
                                : 'border-chat-border text-chat-text-secondary'
                            }`}
                          >
                            {cmd.shortcut}
                          </kbd>
                        )}
                      </button>
                    );
                  })}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2 border-t border-chat-border flex items-center gap-4 text-[11px] text-chat-text-secondary">
          <span><kbd className="font-mono">↑↓</kbd> navigate</span>
          <span><kbd className="font-mono">↵</kbd> select</span>
          <span><kbd className="font-mono">ESC</kbd> close</span>
        </div>
      </div>
    </>
  );
}
