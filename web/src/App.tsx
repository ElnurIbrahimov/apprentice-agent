import { useState, useEffect, useCallback, lazy, Suspense } from 'react';
import { LoginPage } from './components/LoginPage';
import { ChatContainer } from './components/ChatContainer';
import { Sidebar } from './components/Sidebar';
import { ToastContainer, useToastStore } from './components/Toast';
import { CommandPalette } from './components/CommandPalette';
import { useChatStore } from './store/chatStore';
import { useSettingsStore, applyFontSize, applyTheme, applyColorPreset } from './store/settingsStore';
import { ErrorBoundary } from './components/ErrorBoundary';
import { OnboardingFlow } from './components/OnboardingFlow';
import { BottomTabBar } from './components/BottomTabBar';
import { haptic } from './utils/haptics';
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

// ─── Eager imports: lightweight panels that users switch between frequently ───
// These are small (<15KB each) and load instantly instead of showing a skeleton.
import { ConsciousnessPanel } from './components/ConsciousnessPanel';
import { InsightsFeed } from './components/InsightsFeed';
import { WorldModelPanel } from './components/WorldModelPanel';
import { MorningBriefingCard } from './components/MorningBriefingCard';
import { DreamInsightsPanel } from './components/DreamInsightsPanel';
import { EvolutionTracker } from './components/EvolutionTracker';

// ─── Lazy-loaded: heavier panels that benefit from code-splitting ───
const ThoughtStream = lazy(() => import('./components/ThoughtStream').then(m => ({ default: m.ThoughtStream })));
const AuraPanel = lazy(() => import('./components/AuraPanel').then(m => ({ default: m.AuraPanel })));
const NeuroDreamPanel = lazy(() => import('./components/NeuroDreamPanel').then(m => ({ default: m.NeuroDreamPanel })));
const ToolsPanel = lazy(() => import('./components/ToolsPanel').then(m => ({ default: m.ToolsPanel })));
const AMEMPanel = lazy(() => import('./components/AMEMPanel').then(m => ({ default: m.AMEMPanel })));
const ReasoningTreePanel = lazy(() => import('./components/ReasoningTreePanel').then(m => ({ default: m.ReasoningTreePanel })));
const ActivityTimeline = lazy(() => import('./components/ActivityTimeline').then(m => ({ default: m.ActivityTimeline })));
const MemoryTimeline = lazy(() => import('./components/MemoryTimeline').then(m => ({ default: m.MemoryTimeline })));
const KnowledgeGraphExplorer = lazy(() => import('./components/KnowledgeGraphExplorer').then(m => ({ default: m.KnowledgeGraphExplorer })));
const HandsDashboard = lazy(() => import('./components/HandsDashboard'));
const TaskQueuePanel = lazy(() => import('./components/TaskQueuePanel').then(m => ({ default: m.TaskQueuePanel })));
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
const AppCreator = lazy(() => import('./components/AppCreator').then(m => ({ default: m.AppCreator })));
const GameCreator = lazy(() => import('./components/GameCreator').then(m => ({ default: m.GameCreator })));
const DashboardCreator = lazy(() => import('./components/DashboardCreator').then(m => ({ default: m.DashboardCreator })));

import type { TabId } from './types';

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'chat', label: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'create', label: 'Create', icon: CommandLineIcon },
  { id: 'tools', label: 'Tools', icon: WrenchScrewdriverIcon },
  { id: 'insights', label: 'Insights', icon: ChartBarIcon },
  { id: 'settings', label: 'Settings', icon: Cog8ToothIcon },
];

type CreateSubTab = 'code' | 'webcreator' | 'app' | 'game' | 'dashboard' | 'slides' | 'image';
type InsightsSubTab = 'monitor' | 'feed' | 'world' | 'briefing' | 'dreams' | 'evolution' | 'activity' | 'memory' | 'knowledge' | 'hands' | 'queue' | 'advanced';
type ToolsSubTab = 'launcher' | 'system' | ToolId;

function TabSkeleton() {
  return (
    <div className="flex-1 flex items-center justify-center opacity-30">
      <div className="w-5 h-5 rounded-full border-2 border-chat-accent/40 border-t-chat-accent animate-spin" />
    </div>
  );
}

function SubTabBar({ tabs, active, onChange }: { tabs: { id: string; label: string }[]; active: string; onChange: (id: string) => void }) {
  return (
    <div className="flex items-center gap-1.5 px-3 lg:px-4 py-2 border-b border-chat-border overflow-x-auto scrollbar-hide snap-x snap-mandatory">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => { haptic(8); onChange(tab.id); }}
          className={`flex-shrink-0 snap-start px-3.5 py-2 rounded-xl text-xs sm:text-sm font-medium transition-all duration-200 ${
            active === tab.id
              ? 'bg-chat-accent text-white shadow-sm'
              : 'text-chat-text-secondary hover:text-chat-text active:scale-95 hover:bg-chat-assistant'
          }`}
          style={{ minHeight: 36 }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function MainApp() {
  const { sidebarOpen, setSidebarOpen, toggleSidebar, conversations, connectionStatus } = useChatStore();
  const { settings } = useSettingsStore();
  const { toasts, removeToast } = useToastStore();
  const [isMobile, setIsMobile] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('chat');
  const [createSubTab, setCreateSubTab] = useState<CreateSubTab>('code');
  const [insightsSubTab, setInsightsSubTab] = useState<InsightsSubTab>('monitor');
  const [toolsSubTab, setToolsSubTab] = useState<ToolsSubTab>('launcher');

  // Command palette state
  const [showCommandPalette, setShowCommandPalette] = useState(false);

  // Prefetch all lazy chunks after initial render so tab switches are instant
  useEffect(() => {
    const timer = setTimeout(() => {
      // Trigger imports in background — browser caches the chunks
      import('./components/ThoughtStream');
      import('./components/AuraPanel');
      import('./components/ActivityTimeline');
      import('./components/MemoryTimeline');
      import('./components/KnowledgeGraphExplorer');
      import('./components/HandsDashboard');
      import('./components/TaskQueuePanel');
      import('./components/SettingsPage');
      import('./components/ToolsPanel');
      import('./components/ToolLauncher');
      import('./components/ReasoningTreePanel');
      import('./components/NeuroDreamPanel');
      import('./components/AMEMPanel');
    }, 2000); // Wait 2s after mount so initial render isn't blocked
    return () => clearTimeout(timer);
  }, []);

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

  // Apply color preset
  useEffect(() => {
    applyColorPreset(settings.colorPreset ?? 'aura');
  }, [settings.colorPreset]);

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

  // Listen for tool-open events (from tool suggestion chips)
  useEffect(() => {
    const handler = (e: Event) => {
      const { toolId } = (e as CustomEvent).detail;
      setActiveTab('tools');
      setToolsSubTab(toolId as ToolsSubTab);
    };
    document.addEventListener('aura:tool-open', handler);
    return () => document.removeEventListener('aura:tool-open', handler);
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

  const renderTabContent = useCallback(() => {
    switch (activeTab) {
      case 'create':
        return (
          <div className="h-full flex flex-col">
            <SubTabBar
              tabs={[
                { id: 'code', label: 'Code' },
                { id: 'webcreator', label: 'Website' },
                { id: 'app', label: 'App' },
                { id: 'game', label: 'Game' },
                { id: 'dashboard', label: 'Dashboard' },
                { id: 'slides', label: 'Slides' },
                { id: 'image', label: 'Image' },
              ]}
              active={createSubTab}
              onChange={(id) => setCreateSubTab(id as CreateSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {createSubTab === 'code' && <CodeInterpreter />}
              {createSubTab === 'webcreator' && <WebCreator />}
              {createSubTab === 'app' && <AppCreator />}
              {createSubTab === 'game' && <GameCreator />}
              {createSubTab === 'dashboard' && <DashboardCreator />}
              {createSubTab === 'slides' && <SlidesPanel />}
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

            <div className="flex-1 overflow-hidden tab-panel-scroll">
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
                { id: 'monitor', label: 'Mind' },
                { id: 'feed', label: 'Insights' },
                { id: 'briefing', label: 'Briefing' },
                { id: 'dreams', label: 'Dreams' },
                { id: 'evolution', label: 'Evolution' },
                { id: 'world', label: 'World' },
                { id: 'activity', label: 'Activity' },
                { id: 'memory', label: 'Memory' },
                { id: 'knowledge', label: 'Graph' },
                { id: 'hands', label: 'Hands' },
                { id: 'queue', label: 'Queue' },
                { id: 'advanced', label: 'Advanced' },
              ]}
              active={insightsSubTab}
              onChange={(id) => setInsightsSubTab(id as InsightsSubTab)}
            />
            <div className="flex-1 overflow-hidden">
              {insightsSubTab === 'monitor' && <ConsciousnessPanel />}
              {insightsSubTab === 'feed' && <InsightsFeed />}
              {insightsSubTab === 'briefing' && (
                <div className="h-full overflow-y-auto p-3 sm:p-4 space-y-4 tab-panel-scroll">
                  <MorningBriefingCard />
                </div>
              )}
              {insightsSubTab === 'dreams' && <DreamInsightsPanel />}
              {insightsSubTab === 'evolution' && <EvolutionTracker />}
              {insightsSubTab === 'world' && <WorldModelPanel />}
              {insightsSubTab === 'activity' && <ActivityTimeline />}
              {insightsSubTab === 'memory' && <MemoryTimeline />}
              {insightsSubTab === 'knowledge' && <KnowledgeGraphExplorer />}
              {insightsSubTab === 'hands' && <HandsDashboard />}
              {insightsSubTab === 'queue' && <TaskQueuePanel />}
              {insightsSubTab === 'advanced' && (
                <div className="h-full overflow-y-auto p-4 space-y-4 tab-panel-scroll">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <ThoughtStream />
                    <AuraPanel />
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
  }, [activeTab, createSubTab, insightsSubTab, toolsSubTab]);

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

      {/* Connection status — floating pill instead of full-width banner */}
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

          {/* Mobile header right: active tab label + search */}
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

type AuthState =
  | { status: 'checking' }
  | { status: 'anonymous'; configured: boolean }
  | { status: 'authenticated'; username: string };

function App() {
  const { settings, updateSettings } = useSettingsStore();
  const [auth, setAuth] = useState<AuthState>({ status: 'checking' });

  // Probe current session on mount. If the server doesn't have cookie-auth
  // configured (`configured === false`), treat as authenticated so behavior
  // matches the pre-login world — useful for local dev / older deployments.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch('/api/auth/web/me', { credentials: 'same-origin' });
        if (!res.ok) throw new Error(`status=${res.status}`);
        const data = await res.json();
        if (cancelled) return;
        if (data.authenticated) {
          setAuth({ status: 'authenticated', username: data.username || '' });
        } else if (data.configured === false) {
          // Server isn't set up for cookie auth — let the app through.
          setAuth({ status: 'authenticated', username: '' });
        } else {
          setAuth({ status: 'anonymous', configured: true });
        }
      } catch {
        // /me endpoint unreachable (old server build). Fall back to anonymous
        // *only if* the app would also 401 — but since we can't easily know,
        // let the app through and surface errors organically.
        if (!cancelled) setAuth({ status: 'authenticated', username: '' });
      }
    })();
    return () => { cancelled = true; };
  }, []);

  if (auth.status === 'checking') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-400">
        <div className="animate-pulse">Loading…</div>
      </div>
    );
  }

  if (auth.status === 'anonymous') {
    return <LoginPage onLoggedIn={(u) => setAuth({ status: 'authenticated', username: u })} />;
  }

  if (!settings.onboardingDone) {
    return (
      <OnboardingFlow
        onComplete={() => updateSettings({ onboardingDone: true })}
      />
    );
  }

  return <MainApp />;
}

export default App;
