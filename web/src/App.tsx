import { useState, useEffect, useCallback } from 'react';
import AuthGate from './components/AuthGate';
import { ChatContainer } from './components/ChatContainer';
import { Sidebar } from './components/Sidebar';
import { ToastContainer, useToastStore } from './components/Toast';
import { CommandPalette } from './components/CommandPalette';
import { useChatStore } from './store/chatStore';
import { useSettingsStore, applyFontSize, applyTheme, applyColorPreset } from './store/settingsStore';
import { ErrorBoundary } from './components/ErrorBoundary';
import { BottomTabBar } from './components/BottomTabBar';
import { setFaviconState } from './utils/statusFavicon';
import TabRouter, { CreateSubTab, InsightsSubTab, ToolsSubTab } from './components/TabRouter';
import {
  Bars3Icon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon,
  Cog8ToothIcon,
  CommandLineIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

import type { TabId } from './types';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'chat', label: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'create', label: 'Create', icon: CommandLineIcon },
  { id: 'tools', label: 'Tools', icon: WrenchScrewdriverIcon },
  { id: 'insights', label: 'Insights', icon: ChartBarIcon },
  { id: 'settings', label: 'Settings', icon: Cog8ToothIcon },
];

function MainApp() {
  const sidebarOpen = useChatStore(s => s.sidebarOpen);
  const setSidebarOpen = useChatStore(s => s.setSidebarOpen);
  const toggleSidebar = useChatStore(s => s.toggleSidebar);
  const conversations = useChatStore(s => s.conversations);
  const connectionStatus = useChatStore(s => s.connectionStatus);
  const isLoading = useChatStore(s => s.isLoading);
  const settings = useSettingsStore(s => s.settings);
  const toasts = useToastStore(s => s.toasts);
  const removeToast = useToastStore(s => s.removeToast);

  const [isMobile, setIsMobile] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('chat');
  const [createSubTab, setCreateSubTab] = useState<CreateSubTab>('code');
  const [insightsSubTab, setInsightsSubTab] = useState<InsightsSubTab>('monitor');
  const [toolsSubTab, setToolsSubTab] = useState<ToolsSubTab>('launcher');

  const toggleTheme = useCallback(() => {
    const current = useSettingsStore.getState().settings.theme;
    useSettingsStore.getState().updateSettings({ theme: current === 'dark' ? 'light' : 'dark' });
  }, []);

  const newChat = useCallback(() => {
    document.dispatchEvent(new CustomEvent('aura:new-chat'));
  }, []);

  const { showCommandPalette, setShowCommandPalette, showMobileSearch, setShowMobileSearch, showShortcutHelp, setShowShortcutHelp } = useKeyboardShortcuts({
    setActiveTab,
    toggleSidebar,
    toggleTheme,
    newChat,
  });

  const [mobileSearchQuery, setMobileSearchQuery] = useState('');
  const filteredConversations = mobileSearchQuery
    ? conversations.filter(c =>
        (c.title || '').toLowerCase().includes(mobileSearchQuery.toLowerCase())
      )
    : conversations;

  useEffect(() => {
    applyFontSize(settings.fontSize);
  }, [settings.fontSize]);

  useEffect(() => {
    applyTheme(settings.theme);
    if (settings.theme === 'system') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = () => applyTheme('system');
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }
  }, [settings.theme]);

  useEffect(() => {
    applyColorPreset(settings.colorPreset ?? 'aura');
  }, [settings.colorPreset]);

  useEffect(() => {
    const syncFavicon = () => {
      if (isLoading) {
        setFaviconState('thinking');
        return;
      }
      setFaviconState(document.hidden ? 'attention' : 'idle');
    };
    syncFavicon();
    const onVis = () => {
      if (!document.hidden) setFaviconState(isLoading ? 'thinking' : 'idle');
    };
    document.addEventListener('visibilitychange', onVis);
    return () => document.removeEventListener('visibilitychange', onVis);
  }, [isLoading]);

  useEffect(() => {
    const handler = (e: Event) => {
      const tab = (e as CustomEvent).detail;
      if (tab) setActiveTab(tab);
    };
    document.addEventListener('aura:switch-tab', handler);
    return () => document.removeEventListener('aura:switch-tab', handler);
  }, []);

  useEffect(() => {
    const handler = (e: Event) => {
      const { toolId } = (e as CustomEvent).detail;
      setActiveTab('tools');
      setToolsSubTab(toolId as ToolsSubTab);
    };
    document.addEventListener('aura:tool-open', handler);
    return () => document.removeEventListener('aura:tool-open', handler);
  }, []);

  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 1024;
      setIsMobile(mobile);
      if (mobile) {
        setSidebarOpen(false);
      }
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, [setSidebarOpen]);

  return (
    <>
      <div className="mesh-bg">
        <div className="mesh-blob blob-1" />
        <div className="mesh-blob blob-2" />
        <div className="mesh-blob blob-3" />
        <div className="mesh-blob blob-4" />
      </div>
      <div className="grain" />

      <div className="app-shell">
        <ToastContainer toasts={toasts} onDismiss={removeToast} />

        {connectionStatus !== 'connected' && (
          <div
            role="status"
            aria-live="polite"
            className="fixed top-3 left-1/2 -translate-x-1/2 z-[60] animate-spring-scale"
            style={{ paddingTop: 'env(safe-area-inset-top, 0px)' }}
          >
            <div
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium shadow-lg ${
                connectionStatus === 'connecting'
                  ? 'text-yellow-200'
                  : 'text-red-200'
              }`}
              style={{
                background: connectionStatus === 'connecting'
                  ? 'rgba(234, 179, 8, 0.9)'
                  : 'rgba(220, 38, 38, 0.9)',
                backdropFilter: 'blur(12px)',
                WebkitBackdropFilter: 'blur(12px)',
                color: connectionStatus === 'connecting' ? '#000' : '#fff',
              }}
            >
              <span className={`w-2 h-2 rounded-full ${connectionStatus === 'connecting' ? 'bg-yellow-900 animate-pulse' : 'bg-red-300'}`} />
              {connectionStatus === 'connecting' ? 'Reconnecting...' : 'Connection lost'}
            </div>
          </div>
        )}

        <aside
          className={`
            fixed lg:relative inset-y-0 left-0 z-40
            w-64 lg:w-64 shrink-0
            transform transition-transform duration-200 ease-in-out
            ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
            ${isMobile && sidebarOpen ? 'shadow-2xl' : ''}
          `}
        >
          <ErrorBoundary label="Sidebar">
            <Sidebar onClose={() => setSidebarOpen(false)} />
          </ErrorBoundary>
        </aside>

        {isMobile && sidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
            onClick={() => setSidebarOpen(false)}
            onTouchStart={(e) => {
              const startX = e.touches[0].clientX;
              const el = e.currentTarget;
              const handleMove = (ev: TouchEvent) => {
                if (ev.touches[0].clientX - startX < -50) {
                  setSidebarOpen(false);
                  el.removeEventListener('touchmove', handleMove);
                }
              };
              el.addEventListener('touchmove', handleMove, { passive: true });
              el.addEventListener('touchend', () => el.removeEventListener('touchmove', handleMove), { once: true });
            }}
          />
        )}

        <main className="flex-1 flex flex-col min-w-0">
          <header className="flex items-center justify-between px-3 lg:px-4 py-1.5 lg:py-2 border-b border-chat-border" style={{ background: 'var(--bg-panel)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)' }}>
            <div className="flex items-center gap-2">
              <button
                onClick={toggleSidebar}
                className="p-2 -ml-1 text-chat-text-secondary hover:text-chat-text active:scale-90 rounded-xl lg:hidden touch-target transition-all duration-150"
              >
                <Bars3Icon className="w-5 h-5" />
              </button>
              <span className="text-chat-text font-semibold text-sm tracking-wide lg:hidden" style={{ color: 'var(--chat-accent)' }}>AURA</span>
            </div>

            <nav className="hidden lg:flex items-center gap-1">
              {TABS.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 rounded-lg text-sm transition-colors touch-target ${
                      activeTab === tab.id
                        ? 'bg-chat-accent text-white'
                        : 'text-chat-text-secondary hover:text-chat-text active:text-white hover:bg-chat-assistant'
                    }`}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span className="hidden sm:inline truncate">{tab.label}</span>
                  </button>
                );
              })}
            </nav>

            <div className="flex items-center gap-1 lg:hidden">
              <span className="text-xs text-chat-text-secondary font-medium mr-1">
                {TABS.find(t => t.id === activeTab)?.label}
              </span>
              <button
                onClick={() => setShowMobileSearch(true)}
                className="p-2 text-chat-text-secondary hover:text-chat-text active:scale-90 rounded-xl touch-target transition-all duration-150"
                aria-label="Search conversations"
              >
                <MagnifyingGlassIcon className="w-5 h-5" />
              </button>
            </div>
          </header>

          <div className="flex-1 overflow-hidden relative">
            <div style={{ display: activeTab === 'chat' ? 'flex' : 'none' }} className="h-full flex-col">
              <ErrorBoundary label="Chat">
                <ChatContainer />
              </ErrorBoundary>
            </div>
            {activeTab !== 'chat' && (
              <div className="h-full overflow-hidden">
                <ErrorBoundary label={TABS.find((t) => t.id === activeTab)?.label ?? 'Tab'}>
                  <TabRouter
                    activeTab={activeTab}
                    createSubTab={createSubTab}
                    setCreateSubTab={setCreateSubTab}
                    insightsSubTab={insightsSubTab}
                    setInsightsSubTab={setInsightsSubTab}
                    toolsSubTab={toolsSubTab}
                    setToolsSubTab={setToolsSubTab}
                  />
                </ErrorBoundary>
              </div>
            )}
          </div>
        </main>
      </div>

      {isMobile && <BottomTabBar activeTab={activeTab} onTabChange={setActiveTab} />}

      {showMobileSearch && (
        <>
          <div className="fixed inset-0 z-[180] bg-black/60 backdrop-blur-sm lg:hidden animate-fade-in" onClick={() => { setShowMobileSearch(false); setMobileSearchQuery(''); }} />
          <div className="fixed inset-x-0 top-0 z-[190] lg:hidden animate-slide-down" style={{ background: 'var(--surface-1)', borderBottom: '1px solid var(--border-default)', borderRadius: '0 0 16px 16px' }}>
            <div className="flex items-center gap-3 px-4 py-3" style={{ paddingTop: 'max(12px, env(safe-area-inset-top, 0px))' }}>
              <MagnifyingGlassIcon className="w-5 h-5 text-chat-text-secondary flex-shrink-0" />
              <input
                autoFocus
                type="text"
                placeholder="Search conversations..."
                value={mobileSearchQuery}
                onChange={(e) => setMobileSearchQuery(e.target.value)}
                className="flex-1 bg-transparent text-chat-text outline-none text-base"
                style={{ fontSize: 16 }}
              />
              <button
                onClick={() => { setShowMobileSearch(false); setMobileSearchQuery(''); }}
                className="p-1.5 rounded-lg text-chat-text-secondary active:bg-white/10 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="max-h-[60vh] overflow-y-auto px-2 pb-3 overscroll-contain">
              {filteredConversations.length === 0 && !mobileSearchQuery && (
                <div className="text-center text-sm py-8" style={{ color: 'var(--text-tertiary)' }}>No recent conversations</div>
              )}
              {filteredConversations.map((conv) => (
                <button
                  key={conv.id}
                  onClick={() => {
                    useChatStore.getState().setCurrentConversationId(conv.id);
                    setShowMobileSearch(false);
                    setMobileSearchQuery('');
                    setActiveTab('chat');
                  }}
                  className="w-full text-left px-3 py-3 rounded-xl active:bg-white/8 transition-colors"
                  style={{ color: 'var(--text-primary)' }}
                >
                  <div className="text-sm truncate font-medium">{conv.title || 'Untitled'}</div>
                  <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>
                    {conv.updated_at ? new Date(conv.updated_at * 1000).toLocaleDateString() : ''}
                  </div>
                </button>
              ))}
              {mobileSearchQuery && filteredConversations.length === 0 && (
                <div className="text-center text-sm py-8" style={{ color: 'var(--text-secondary)' }}>No conversations found</div>
              )}
            </div>
          </div>
        </>
      )}

      <CommandPalette
        isOpen={showCommandPalette}
        onClose={() => setShowCommandPalette(false)}
        setActiveTab={setActiveTab}
        setToolsSubTab={(tab) => setToolsSubTab(tab as ToolsSubTab)}
        setCreateSubTab={setCreateSubTab}
        toggleSidebar={toggleSidebar}
        newChat={newChat}
        toggleTheme={toggleTheme}
      />

      {showShortcutHelp && (
        <>
          <div className="fixed inset-0 z-[200] bg-black/50 backdrop-blur-sm" onClick={() => setShowShortcutHelp(false)} />
          <div className="fixed z-[210] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[360px] max-w-[90vw] rounded-xl border border-chat-border p-5"
            style={{ background: 'var(--surface-2)', boxShadow: '0 16px 48px rgba(0,0,0,0.4)' }}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-chat-text">Keyboard Shortcuts</h3>
              <button onClick={() => setShowShortcutHelp(false)} className="text-chat-text-secondary hover:text-chat-text">
                <span className="sr-only">Close</span>&times;
              </button>
            </div>
            <div className="space-y-2 text-xs text-chat-text-secondary">
              {[
                ['Ctrl+1–5', 'Switch tab'],
                ['Ctrl+B', 'Toggle sidebar'],
                ['Ctrl+N', 'New chat'],
                ['Ctrl+K', 'Command palette'],
                ['Ctrl+/', 'Settings'],
                ['Ctrl+Alt+T', 'Toggle theme'],
                ['Escape', 'Close panel/modal'],
                ['?', 'This help'],
              ].map(([key, desc]) => (
                <div key={key} className="flex justify-between">
                  <kbd className="px-1.5 py-0.5 rounded bg-surface-3 text-chat-text font-mono text-[11px]">{key}</kbd>
                  <span>{desc}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </>
  );
}

function App() {
  return (
    <AuthGate>
      <MainApp />
    </AuthGate>
  );
}

export default App;
