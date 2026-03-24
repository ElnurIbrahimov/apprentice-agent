import { useState, useEffect } from 'react';
import { ChatContainer } from './components/ChatContainer';
import { Sidebar } from './components/Sidebar';
import { ThoughtStream } from './components/ThoughtStream';
import { AuraPanel } from './components/AuraPanel';
import { NeuroDreamPanel } from './components/NeuroDreamPanel';
import { ToolsPanel } from './components/ToolsPanel';
import { AMEMPanel } from './components/AMEMPanel';
import { ReasoningTreePanel } from './components/ReasoningTreePanel';
import { ToastContainer, useToastStore } from './components/Toast';
import { ActivityTimeline } from './components/ActivityTimeline';
import { useChatStore } from './store/chatStore';
import { useSettingsStore, applyFontSize, applyTheme } from './store/settingsStore';
import { ErrorBoundary } from './components/ErrorBoundary';
import {
  Bars3Icon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon,
  CogIcon,
  ClockIcon,
  Cog8ToothIcon,
} from '@heroicons/react/24/outline';
import { SettingsPage } from './components/SettingsPage';
import type { TabId } from './types';

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'chat', label: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'monitoring', label: 'Monitor', icon: ChartBarIcon },
  { id: 'tools', label: 'Tools', icon: WrenchScrewdriverIcon },
  { id: 'advanced', label: 'Advanced', icon: CogIcon },
  { id: 'activity', label: 'Activity', icon: ClockIcon },
  { id: 'settings', label: 'Settings', icon: Cog8ToothIcon },
];

function App() {
  const { sidebarOpen, setSidebarOpen, toggleSidebar } = useChatStore();
  const { settings } = useSettingsStore();
  const { toasts, removeToast } = useToastStore();
  const [isMobile, setIsMobile] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('chat');

  // Apply font size setting
  useEffect(() => {
    applyFontSize(settings.fontSize);
  }, [settings.fontSize]);

  // Apply theme setting
  useEffect(() => {
    applyTheme(settings.theme);
  }, [settings.theme]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
        document.dispatchEvent(new CustomEvent('aura:focus-input'));
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

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
      case 'monitoring':
        return (
          <div className="h-full overflow-y-auto p-4 space-y-4">
            <h2 className="text-xl font-semibold text-chat-text mb-4">Monitoring</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ThoughtStream />
              <AuraPanel />
            </div>
          </div>
        );

      case 'tools':
        return (
          <div className="h-full overflow-y-auto p-4">
            <h2 className="text-xl font-semibold text-chat-text mb-4">Tools & Systems</h2>
            <ToolsPanel />
          </div>
        );

      case 'advanced':
        return (
          <div className="h-full overflow-y-auto p-4 space-y-4">
            <h2 className="text-xl font-semibold text-chat-text mb-4">Advanced Features</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ReasoningTreePanel />
              <NeuroDreamPanel />
              <AMEMPanel />
            </div>
          </div>
        );

      case 'activity':
        return (
          <div className="h-full overflow-hidden">
            <ActivityTimeline />
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

      {/* Mobile overlay backdrop — tap or swipe left to close */}
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
        <header className="flex items-center justify-between px-4 py-2 border-b border-chat-border" style={{ background: 'rgba(3,3,3,0.7)', backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)' }}>
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 -ml-2 text-chat-text-secondary hover:text-chat-text active:text-white rounded-lg lg:hidden touch-target"
            >
              <Bars3Icon className="w-6 h-6" />
            </button>
            <span className="ml-2 text-chat-text font-semibold lg:hidden">AURA</span>
          </div>

          {/* Tab buttons */}
          <nav className="flex items-center gap-1">
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

          {/* Spacer for layout balance */}
          <div className="w-10 lg:hidden" />
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
                {renderTabContent()}
              </ErrorBoundary>
            </div>
          )}
        </div>
      </main>
    </div>
    </>
  );
}

export default App;
