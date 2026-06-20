import { useEffect, useState } from 'react';
import type { TabId } from '../types';

const TABS: TabId[] = ['chat', 'create', 'tools', 'insights', 'settings'];

interface ShortcutActions {
  setActiveTab: (tab: TabId) => void;
  toggleSidebar: () => void;
  toggleTheme: () => void;
  newChat: () => void;
}

export function useKeyboardShortcuts(actions: ShortcutActions) {
  const { setActiveTab, toggleSidebar, toggleTheme, newChat } = actions;
  const [showCommandPalette, setShowCommandPalette] = useState(false);
  const [showMobileSearch, setShowMobileSearch] = useState(false);
  const [showShortcutHelp, setShowShortcutHelp] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable;

      if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        newChat();
      }
      if (e.ctrlKey && e.key === '/') {
        e.preventDefault();
        setActiveTab('settings');
      }
      if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(prev => !prev);
      }
      if (e.ctrlKey && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
      }
      if (e.ctrlKey && e.key >= '1' && e.key <= '5') {
        e.preventDefault();
        const idx = parseInt(e.key) - 1;
        if (TABS[idx]) setActiveTab(TABS[idx]);
      }
      if (e.ctrlKey && e.altKey && (e.key === 't' || e.key === 'T')) {
        e.preventDefault();
        toggleTheme();
      }
      if (e.key === 'Escape') {
        if (showCommandPalette) { setShowCommandPalette(false); e.preventDefault(); }
        else if (showMobileSearch) { setShowMobileSearch(false); e.preventDefault(); }
        else if (showShortcutHelp) { setShowShortcutHelp(false); e.preventDefault(); }
      }
      if (e.key === '?' && !isInput) {
        setShowShortcutHelp((prev) => !prev);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showCommandPalette, showShortcutHelp, showMobileSearch, setActiveTab, toggleSidebar, toggleTheme, newChat]);

  return { showCommandPalette, setShowCommandPalette, showMobileSearch, setShowMobileSearch, showShortcutHelp, setShowShortcutHelp };
}
