/**
 * Keyboard shortcuts for Aura sidebar.
 * Call initShortcuts(store) once at startup.
 */
import type { PanelId } from './types';

// Panel order for Ctrl+1..9
const PANEL_SLOTS: PanelId[] = ['chat', 'search', 'write', 'artifacts', 'webcreator', 'code', 'research', 'summary', 'settings'];

type Store = {
  getState: () => {
    activePanel: PanelId;
    thinkingMode: boolean;
    deepResearch: boolean;
    moreOpen: boolean;
  };
  setState: (partial: Record<string, unknown>) => void;
};

// Resolve store actions from the raw zustand store
function getActions(store: Store) {
  const s = store.getState() as any;
  return {
    setPanel: s.setPanel as (p: PanelId) => void,
    setThinkingMode: s.setThinkingMode as (on: boolean) => void,
    setDeepResearch: s.setDeepResearch as (on: boolean) => void,
    setMoreOpen: s.setMoreOpen as (open: boolean) => void,
    clearAll: s.clearAll as () => void,
  };
}

export function initShortcuts(store: Store) {
  document.addEventListener('keydown', (e: KeyboardEvent) => {
    const mod = e.metaKey || e.ctrlKey;
    const key = e.key.toLowerCase();
    const state = store.getState();
    const actions = getActions(store);

    // --- Escape: close dropdown/modal, exit fullscreen, or blur input ---
    if (e.key === 'Escape') {
      if (state.moreOpen) {
        actions.setMoreOpen(false);
        e.preventDefault();
        return;
      }
      // Exit fullscreen panels
      const fs = document.querySelector('[style*="position: fixed"][style*="inset: 0"]');
      if (fs) {
        const exitBtn = fs.querySelector('button[title*="xit"], button[title*="inimize"]') as HTMLElement | null;
        exitBtn?.click();
        e.preventDefault();
        return;
      }
      // Blur focused input/textarea
      const active = document.activeElement as HTMLElement | null;
      if (active && (active.tagName === 'TEXTAREA' || active.tagName === 'INPUT')) {
        active.blur();
        e.preventDefault();
        return;
      }
      return;
    }

    // --- Ctrl/Cmd+K: Focus chat input ---
    if (mod && !e.shiftKey && key === 'k') {
      e.preventDefault();
      const ta = document.querySelector('textarea') as HTMLTextAreaElement | null;
      if (ta) {
        ta.focus();
        // Move cursor to end
        ta.setSelectionRange(ta.value.length, ta.value.length);
      }
      return;
    }

    // --- Ctrl/Cmd+L: Clear conversation ---
    if (mod && !e.shiftKey && key === 'l') {
      e.preventDefault();
      actions.clearAll();
      return;
    }

    // --- Ctrl/Cmd+N: New conversation ---
    if (mod && !e.shiftKey && key === 'n') {
      e.preventDefault();
      actions.clearAll();
      actions.setPanel('chat');
      // Focus input after clearing
      requestAnimationFrame(() => {
        const ta = document.querySelector('textarea') as HTMLTextAreaElement | null;
        ta?.focus();
      });
      return;
    }

    // --- Ctrl+Shift+T: Toggle thinking mode ---
    if (mod && e.shiftKey && key === 't') {
      e.preventDefault();
      actions.setThinkingMode(!state.thinkingMode);
      return;
    }

    // --- Ctrl+Shift+R: Toggle deep research mode ---
    if (mod && e.shiftKey && key === 'r') {
      e.preventDefault();
      actions.setDeepResearch(!state.deepResearch);
      return;
    }

    // --- Ctrl+1..9: Switch panels ---
    if (mod && !e.shiftKey && e.key >= '1' && e.key <= '9') {
      e.preventDefault();
      const idx = parseInt(e.key, 10) - 1;
      const panel = PANEL_SLOTS[idx];
      if (panel) {
        actions.setPanel(panel);
      }
      return;
    }

  });
}
