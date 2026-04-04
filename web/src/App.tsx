import { useState, useEffect, useCallback, lazy, Suspense } from 'react';
import { ChatContainer } from './components/ChatContainer';
import { Sidebar } from './components/Sidebar';
import { ToastContainer, useToastStore } from './components/Toast';
import { CommandPalette } from './components/CommandPalette';
import { useChatStore } from './store/chatStore';
import { useSettingsStore, applyFontSize, applyTheme } from './store/settingsStore';
import { ErrorBoundary } from './components/ErrorBoundary';
import { BottomTabBar } from './components/BottomTabBar';
import { ToolLauncher, ToolSubNav } from './components/ToolLauncher';
import type { ToolId } from './components/ToolLauncher';
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

// Lazy-loaded tabs (only Chat + Sidebar are eagerly loaded)
const ThoughtStream = lazy(() => import('./components/ThoughtStream').then(m => ({ default: m.ThoughtStream })));
const AuraPanel = lazy(() => import('./components/AuraPanel').then(m => ({ default: m.AuraPanel })));
const NeuroDreamPanel = lazy(() => import('./components/NeuroDreamPanel').then(m => ({ default: m.NeuroDreamPanel })));
const ToolsPanel = lazy(() => import('./components/ToolsPanel').then(m => ({ default: m.ToolsPanel })));
const AMEMPanel = lazy(() => import('./components/AMEMPanel').then(m => ({ default: m.AMEMPanel })));
const ReasoningTreePanel = lazy(() => import('./components/ReasoningTreePanel').then(m => ({ default: m.ReasoningTreePanel })));
const ActivityTimeline = lazy(() => import('./components/ActivityTimeline').then(m => ({ default: m.ActivityTimeline })));
const HandsDashboard = lazy(() => import('./components/HandsDashboard'));
const SettingsPage = lazy(() => import('./components/SettingsPage').then(m => ({ default: m.SettingsPage })));
const CodeInterpreter = lazy(() => import('./components/CodeInterpreter').then(m => ({ default: m.CodeInterpreter })));
const WebCreator = lazy(() => import('./components/WebCreator').then(m => ({ default: m.WebCreator })));
const ImageGenPanel = lazy(() => import('./components/ImageGenPanel').then(m => ({ default: m.ImageGenPanel })));
const SearchPanel = lazy(() => import('./components/SearchPanel').then(m => ({ default: m.SearchPanel })));
const ResearchPanel = lazy(() => import('./components/ResearchPanel').then(m => ({ default: m.ResearchPanel })));
const WritePanel = lazy(() => import('./components/WritePanel').then(m => ({ default: m.WritePanel })));
const TranslatePanel = lazy(() => import('./components/TranslatePanel').then(m => ({ default: m.TranslatePanel })));
const SummaryPanel = lazy(() => import('./components/SummaryPanel').then(m => ({ default: m.SummaryPanel })));
const GrammarPanel = lazy(() => import('./components/GrammarPanel').then(m => ({ default: m.GrammarPanel })));
const MathPanel = lazy(() => import('./components/MathPanel').then(m => ({ default: m.MathPanel })));
const AskPanel = lazy(() => import('./components/AskPanel').then(m => ({ default: m.AskPanel })));
const AgentPanel = lazy(() => import('./components/AgentPanel').then(m => ({ default: m.AgentPanel })));
const ComparePanel = lazy(() => import('./components/ComparePanel').then(m => ({ default: m.ComparePanel })));
const ModelsPanel = lazy(() => import('./components/ModelsPanel').then(m => ({ default: m.ModelsPanel })));
const PdfPanel = lazy(() => import('./components/PdfPanel').then(m => ({ default: m.PdfPanel })));
const VoicePanel = lazy(() => import('./components/VoicePanel').then(m => ({ default: m.VoicePanel })));
const OcrPanel = lazy(() => import('./components/OcrPanel').then(m => ({ default: m.OcrPanel })));
const YoutubePanel = lazy(() => import('./components/YoutubePanel').then(m => ({ default: m.YoutubePanel })));
const CapturePanel = lazy(() => import('./components/CapturePanel').then(m => ({ default: m.CapturePanel })));
const WisebasePanel = lazy(() => import('./components/WisebasePanel').then(m => ({ default: m.WisebasePanel })));
const SlidesPanel = lazy(() => import('./components/SlidesPanel').then(m => ({ default: m.SlidesPanel })));
const RecordPanel = lazy(() => import('./components/RecordPanel').then(m => ({ default: m.RecordPanel })));

import type { TabId } from './types';

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'chat', label: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'create', label: 'Create', icon: CommandLineIcon },
  { id: 'tools', label: 'Tools', icon: WrenchScrewdriverIcon },
  { id: 'insights', label: 'Insights', icon: ChartBarIcon },
  { id: 'settings', label: 'Settings', icon: Cog8ToothIcon },
];

type CreateSubTab = 'code' | 'webcreator' | 'image';
type InsightsSubTab = 'monitor' | 'activity' | 'hands' | 'advanced';
type ToolsSubTab = 'launcher' | 'system' | ToolId;

function TabSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center">
      <div className="animate-pulse flex flex-col gap-3 w-64">
        <div className="h-4 bg-surface-2 rounded w-3/4" />
        <div className="h-4 bg-surface-2 rounded w-1/2" />
        <div className="h-4 bg-surface-2 rounded w-5/6" />
      </div>
    </div>
  );
}

function SubTabBar({ tabs, active, onChange }: { tabs: { id: string; label: string }[]; active: string; onChange: (id: string) => void }) {
  return (
    <div className="flex items-center gap-1 px-4 py-2 border-b border-chat-border overflow-x-auto scrollbar-hide">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`flex-shrink-0 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
            active === tab.id
              ? 'bg-chat-accent text-white'
              : 'text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant'
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function App() {
  const { sidebarOpen, setSidebarOpen, toggleSidebar, conversations } = useChatStore();
  const { settings } = useSettingsStore();
  const { toasts, removeToast } = useToastStore();
  const [isMobile, setIsMobile] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('chat');
  const [createSubTab, setCreateSubTab] = useState<CreateSubTab>('code');
  const [insightsSubTab, setInsightsSubTab] = useState<InsightsSubTab>('monitor');
  const [toolsSubTab, setToolsSubTab] = useState<ToolsSubTab>('launcher');

  // Command palette state
  const [showCommandPalette, setShowCommandPalette] = useState(false);

  const toggleTheme = useCallback(() => {
    const current = useSettingsStore.getState().settings.theme;
    useSettingsStore.getState().updateSettings({ theme: current === 'dark' ? 'light' : 'dark' });
  }, []);

  const newChat = useCallback(() => {
    document.dispatchEvent(new CustomEvent('aura:new-chat'));
  }, []);

  // Mobile search state
  const [showMobileSearch, setShowMobileSearch] = useState(false);
  const [mobileSearchQuery, setMobileSearchQuery] = useState('');
  const filteredConversations = mobileSearchQuery
    ? conversations.filter(c =>
        (c.title || '').toLowerCase().includes(mobileSearchQuery.toLowerCase())
      )
    : conversations.slice(0, 20);

  // Apply font size setting
  useEffect(() => {
    applyFontSize(settings.fontSize);
  }, [settings.fontSize]);

  // Apply theme setting + listen for OS theme changes when set to "system"
  useEffect(() => {
    applyTheme(settings.theme);
    if (settings.theme === 'system') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const handler = () => applyTheme('system');
      mq.addEventListener('change', handler);
      return () => mq.removeEventListener('change', handler);
    }
  }, [settings.theme]);

  // Keyboard shortcuts
  const [showShortcutHelp, setShowShortcutHelp] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable;

      if (e.ctrlKey && e.key === 'n') {
        e.preventDefault();
        document.dispatchEvent(new CustomEvent('aura:new-chat'));
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
        if (TABS[idx]) setActiveTab(TABS[idx].id);
      }
      if (e.ctrlKey && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        toggleTheme();
      }
      if (e.key === 'Escape') {
        if (showCommandPalette) { setShowCommandPalette(false); e.preventDefault(); }
        else if (showMobileSearch) { setShowMobileSearch(false); setMobileSearchQuery(''); e.preventDefault(); }
        else if (showShortcutHelp) { setShowShortcutHelp(false); e.preventDefault(); }
      }
      if (e.key === '?' && !isInput) {
        setShowShortcutHelp((prev) => !prev);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showCommandPalette, showShortcutHelp, showMobileSearch, toggleSidebar, toggleTheme]);

  // Listen for tab switch events (from Sidebar gear button, etc.)
  useEffect(() => {
    const handler = (e: Event) => {
      const tab = (e as CustomEvent).detail;
      if (tab) setActiveTab(tab);
    };
    document.addEventListener('aura:switch-tab', handler);
    return () => document.removeEventListener('aura:switch-tab', handler);
  }, []);

  // Detect mobile viewport
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

  const renderTabContent = () => {
    switch (activeTab) {
      case 'create':
        return (
          <div className="h-full flex flex-col">
            <SubTabBar
              tabs={[
                { id: 'code', label: 'Code' },
                { id: 'webcreator', label: 'Web Creator' },
                { id: 'image', label: 'Image' },
              ]}
              active={createSubTab}
              onChange={(id) => setCreateSubTab(id as CreateSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {createSubTab === 'code' && <CodeInterpreter />}
              {createSubTab === 'webcreator' && <WebCreator />}
              {createSubTab === 'image' && <ImageGenPanel />}
            </div>
          </div>
        );

      case 'tools': {
        const showLauncher = toolsSubTab === 'launcher';
        return (
          <div className="h-full flex flex-col">
            {/* Compact sub-nav: only visible when a specific tool (or system) is active */}
            {!showLauncher && (
              <ToolSubNav
                activeId={toolsSubTab === 'system' ? 'tools' : toolsSubTab as ToolId}
                onSelect={(id) => setToolsSubTab(id === 'tools' ? 'system' : id as ToolsSubTab)}
                onBack={() => setToolsSubTab('launcher')}
              />
            )}

            <div className="flex-1 overflow-hidden">
              {/* Launcher grid */}
              {showLauncher && (
                <ToolLauncher
                  onSelect={(id) =>
                    setToolsSubTab(id === 'tools' ? 'system' : id as ToolsSubTab)
                  }
                />
              )}

              {/* System Tools panel */}
              {toolsSubTab === 'system' && (
                <div className="h-full overflow-y-auto p-4 tab-panel-scroll">
                  <h2 className="text-xl font-semibold text-chat-text mb-4">Tools &amp; Systems</h2>
                  <ToolsPanel />
                </div>
              )}

              {/* Individual tool panels */}
              {toolsSubTab === 'ask'       && <AskPanel />}
              {toolsSubTab === 'search'    && <SearchPanel />}
              {toolsSubTab === 'research'  && <ResearchPanel />}
              {toolsSubTab === 'agent'     && <AgentPanel />}
              {toolsSubTab === 'compare'   && <ComparePanel />}
              {toolsSubTab === 'write'     && <WritePanel />}
              {toolsSubTab === 'translate' && <TranslatePanel />}
              {toolsSubTab === 'summary'   && <SummaryPanel />}
              {toolsSubTab === 'grammar'   && <GrammarPanel />}
              {toolsSubTab === 'math'      && <MathPanel />}
              {toolsSubTab === 'pdf'       && <PdfPanel />}
              {toolsSubTab === 'ocr'       && <OcrPanel />}
              {toolsSubTab === 'capture'   && <CapturePanel />}
              {toolsSubTab === 'youtube'   && <YoutubePanel />}
              {toolsSubTab === 'voice'     && <VoicePanel />}
              {toolsSubTab === 'record'    && <RecordPanel />}
              {toolsSubTab === 'slides'    && <SlidesPanel />}
              {toolsSubTab === 'wisebase'  && <WisebasePanel />}
              {toolsSubTab === 'models'    && <ModelsPanel />}
            </div>
          </div>
        );
      }

      case 'insights':
        return (
          <div className="h-full flex flex-col">
            <SubTabBar
              tabs={[
                { id: 'monitor', label: 'Monitor' },
                { id: 'activity', label: 'Activity' },
                { id: 'hands', label: 'Hands' },
                { id: 'advanced', label: 'Advanced' },
              ]}
              active={insightsSubTab}
              onChange={(id) => setInsightsSubTab(id as InsightsSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {insightsSubTab === 'monitor' && (
                <div className="h-full overflow-y-auto p-4 space-y-4 tab-panel-scroll">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <ThoughtStream />
                    <AuraPanel />
                  </div>
                </div>
              )}
              {insightsSubTab === 'activity' && <ActivityTimeline />}
              {insightsSubTab === 'hands' && <HandsDashboard />}
              {insightsSubTab === 'advanced' && (
                <div className="h-full overflow-y-auto p-4 space-y-4 tab-panel-scroll">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <ReasoningTreePanel />
                    <NeuroDreamPanel />
                    <AMEMPanel />
                  </div>
                </div>
              )}
            </div>
          </div>
        );

      case 'settings':
        return (
          <div className="h-full overflow-hidden">
            <SettingsPage />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <>
      {/* Animated mesh gradient background */}
      <div className="mesh-bg">
        <div className="mesh-blob blob-1" />
        <div className="mesh-blob blob-2" />
        <div className="mesh-blob blob-3" />
        <div className="mesh-blob blob-4" />
      </div>
      <div className="grain" />

    <div className="app-shell">
      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
      {/* Sidebar - Desktop: fixed, Mobile: overlay */}
      <aside
        className={`
          fixed lg:relative inset-y-0 left-0 z-40
          w-64 lg:w-64 shrink-0
          transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          ${isMobile && sidebarOpen ? 'shadow-2xl' : ''}
        `}
      >
        <ErrorBoundary>
          <Sidebar onClose={() => setSidebarOpen(false)} />
        </ErrorBoundary>
      </aside>

      {/* Mobile overlay backdrop */}
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

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with tabs */}
        <header className="flex items-center justify-between px-4 py-2 border-b border-chat-border" style={{ background: 'var(--bg-panel)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}>
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 -ml-2 text-chat-text-secondary hover:text-chat-text active:text-white rounded-lg lg:hidden touch-target"
            >
              <Bars3Icon className="w-6 h-6" />
            </button>
            <span className="ml-2 text-chat-text font-semibold lg:hidden">AURA</span>
          </div>

          {/* Tab buttons — hidden on mobile (bottom bar used instead) */}
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

          {/* Mobile search button */}
          <button
            onClick={() => setShowMobileSearch(true)}
            className="p-2 text-chat-text-secondary hover:text-chat-text rounded-lg lg:hidden touch-target"
            aria-label="Search conversations"
          >
            <MagnifyingGlassIcon className="w-5 h-5" />
          </button>
        </header>

        {/* Tab content — ChatContainer is always mounted to keep WebSocket alive */}
        <div className="flex-1 overflow-hidden relative">
          <div style={{ display: activeTab === 'chat' ? 'flex' : 'none' }} className="h-full flex-col">
            <ErrorBoundary>
              <ChatContainer />
            </ErrorBoundary>
          </div>
          {activeTab !== 'chat' && (
            <div className="h-full overflow-hidden">
              <ErrorBoundary>
                <Suspense fallback={<TabSkeleton />}>
                  {renderTabContent()}
                </Suspense>
              </ErrorBoundary>
            </div>
          )}
        </div>
      </main>
    </div>

    {/* Bottom tab bar for mobile */}
    {isMobile && <BottomTabBar activeTab={activeTab} onTabChange={setActiveTab} />}

    {/* Mobile search overlay */}
    {showMobileSearch && (
      <>
        <div className="fixed inset-0 z-[180] bg-black/60 backdrop-blur-sm lg:hidden" onClick={() => { setShowMobileSearch(false); setMobileSearchQuery(''); }} />
        <div className="fixed inset-x-0 top-0 z-[190] lg:hidden" style={{ background: 'var(--surface-1)', borderBottom: '1px solid var(--border-default)' }}>
          <div className="flex items-center gap-2 px-4 py-3">
            <MagnifyingGlassIcon className="w-5 h-5 text-chat-text-secondary flex-shrink-0" />
            <input
              autoFocus
              type="text"
              placeholder="Search conversations..."
              value={mobileSearchQuery}
              onChange={(e) => setMobileSearchQuery(e.target.value)}
              className="flex-1 bg-transparent text-chat-text outline-none text-sm"
            />
            <button onClick={() => { setShowMobileSearch(false); setMobileSearchQuery(''); }} className="text-chat-text-secondary p-1">
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>
          <div className="max-h-[60vh] overflow-y-auto px-2 pb-3">
            {filteredConversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => {
                  useChatStore.getState().setCurrentConversationId(conv.id);
                  setShowMobileSearch(false);
                  setMobileSearchQuery('');
                }}
                className="w-full text-left px-3 py-2.5 rounded-lg transition-colors"
                style={{ color: 'var(--text-primary)' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--bg-panel-hover)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <div className="text-sm truncate">{conv.title || 'Untitled'}</div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>{new Date(conv.updated_at * 1000).toLocaleDateString()}</div>
              </button>
            ))}
            {mobileSearchQuery && filteredConversations.length === 0 && (
              <div className="text-center text-sm py-8" style={{ color: 'var(--text-secondary)' }}>No conversations found</div>
            )}
          </div>
        </div>
      </>
    )}

    {/* Command palette */}
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

    {/* Keyboard shortcut help modal */}
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
              ['Ctrl+Shift+T', 'Toggle theme'],
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

export default App;
