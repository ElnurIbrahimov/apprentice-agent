import { useState, useEffect } from 'react';
import { ChatContainer } from './components/ChatContainer';
import { Sidebar } from './components/Sidebar';
import { ThoughtStream } from './components/ThoughtStream';
import { AuraPanel } from './components/AuraPanel';
import { GuardianPanel } from './components/GuardianPanel';
import { NeuroDreamPanel } from './components/NeuroDreamPanel';
import { ToolsPanel } from './components/ToolsPanel';
import { AMEMPanel } from './components/AMEMPanel';
import { ProtoAGIPanel } from './components/ProtoAGIPanel';
import { ReasoningTreePanel } from './components/ReasoningTreePanel';
import { IntrospectionPanel } from './components/IntrospectionPanel';
import { ToastContainer, useToastStore } from './components/Toast';
import { useChatStore } from './store/chatStore';
import { useSettingsStore, applyFontSize } from './store/settingsStore';
import {
  Bars3Icon,
  ChatBubbleLeftRightIcon,
  ChartBarIcon,
  WrenchScrewdriverIcon,
  CogIcon,
} from '@heroicons/react/24/outline';
import type { TabId } from './types';

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'chat', label: 'Chat', icon: ChatBubbleLeftRightIcon },
  { id: 'monitoring', label: 'Monitor', icon: ChartBarIcon },
  { id: 'tools', label: 'Tools', icon: WrenchScrewdriverIcon },
  { id: 'advanced', label: 'Advanced', icon: CogIcon },
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
              <ProtoAGIPanel />
              <GuardianPanel />
              <IntrospectionPanel />
              <ReasoningTreePanel />
              <NeuroDreamPanel />
              <AMEMPanel />
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-chat-bg">
      {/* Toast notifications */}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
      {/* Sidebar - Desktop: fixed, Mobile: overlay */}
      <aside
        className={`
          fixed lg:relative inset-y-0 left-0 z-40
          w-64 lg:w-64 shrink-0
          transform transition-transform duration-200 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0 lg:hidden'}
          ${isMobile && sidebarOpen ? 'shadow-2xl' : ''}
        `}
      >
        <Sidebar onClose={() => setSidebarOpen(false)} />
      </aside>

      {/* Mobile overlay backdrop */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header with tabs */}
        <header className="flex items-center justify-between px-4 py-2 border-b border-chat-border bg-chat-bg">
          <div className="flex items-center">
            <button
              onClick={toggleSidebar}
              className="p-2 -ml-2 text-chat-text-secondary hover:text-chat-text rounded-lg lg:hidden"
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
                  className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'bg-chat-accent text-white'
                      : 'text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
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
            <ChatContainer />
          </div>
          {activeTab !== 'chat' && (
            <div className="h-full overflow-hidden">
              {renderTabContent()}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
