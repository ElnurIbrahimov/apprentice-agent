import { useEffect, useState } from 'react';
import { useChatStore } from '../store/chatStore';
import { EmotionPanel } from './EmotionPanel';
import { SettingsModal } from './SettingsModal';
import { AuraBreathingAvatar, AuraStatusLine, AuraConsideringIndicator } from './AuraBreathingAvatar';
import { ProactiveDaemonPanel } from './ProactiveDaemonPanel';
import { SystemStatsPanel } from './SystemStatsPanel';
import { InnerThoughtsPanel } from './InnerThoughtsPanel';
import { ThinkingAboutTeaser } from './ThinkingAboutTeaser';
import { MemoryRecallIndicator } from './MemoryRecallIndicator';
import { ContextHeatmap } from './ContextHeatmap';
import { ConversationStarterPanel } from './ConversationStarterPanel';
import {
  XMarkIcon,
  TrashIcon,
  Cog6ToothIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline';

interface ConsiderationState {
  is_considering: boolean;
  decided_against: boolean;
  topic: string | null;
}

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  const {
    status,
    setStatus,
    connectionStatus,
    clearMessages,
    selectedModel,
    setSelectedModel,
    availableModels,
    setAvailableModels,
    isLoading,
  } = useChatStore();

  // Ambient status messages that make AURA feel alive
  const [ambientStatus, setAmbientStatus] = useState<string | null>(null);

  // AURA consideration state - "thinking about saying something"
  const [consideration, setConsideration] = useState<ConsiderationState>({
    is_considering: false,
    decided_against: false,
    topic: null,
  });

  // Poll consideration state
  useEffect(() => {
    if (connectionStatus !== 'connected') return;

    const fetchConsideration = async () => {
      try {
        const response = await fetch('/api/aura/consideration');
        if (response.ok) {
          const data = await response.json();
          setConsideration(data);
        }
      } catch (e) {
        // Silently ignore - not critical
      }
    };

    // Poll every 2 seconds to catch considerations
    const interval = setInterval(fetchConsideration, 2000);
    fetchConsideration(); // Initial fetch

    return () => clearInterval(interval);
  }, [connectionStatus]);

  // Simulate idle "noticing" behavior
  useEffect(() => {
    if (connectionStatus !== 'connected') return;

    const idleMessages = [
      'Monitoring context...',
      'Processing memories...',
      'Observing patterns...',
      'Integrating knowledge...',
      'Reflecting quietly...',
      null, // Sometimes show nothing
      null,
      null,
    ];

    const showRandomStatus = () => {
      const msg = idleMessages[Math.floor(Math.random() * idleMessages.length)];
      setAmbientStatus(msg);
    };

    // Initial delay before first status
    const initialDelay = setTimeout(() => {
      showRandomStatus();
    }, 5000);

    // Show status periodically with random intervals
    const interval = setInterval(() => {
      showRandomStatus();
    }, 8000 + Math.random() * 12000); // 8-20 seconds

    return () => {
      clearTimeout(initialDelay);
      clearInterval(interval);
    };
  }, [connectionStatus]);

  // Poll status every 5 seconds
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('/api/status');
        if (response.ok) {
          const data = await response.json();
          setStatus(data);
        }
      } catch (e) {
        console.error('Failed to fetch status:', e);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [setStatus]);

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        console.log('[Sidebar] Fetching models...');
        const response = await fetch('/api/models');
        console.log('[Sidebar] Models response status:', response.status);
        if (response.ok) {
          const data = await response.json();
          console.log('[Sidebar] Models data:', data);
          const allModels = [...(data.local_models || []), ...(data.cloud_models || [])];
          console.log('[Sidebar] All models:', allModels);
          setAvailableModels(allModels);
        } else {
          console.error('[Sidebar] Models API returned:', response.status);
        }
      } catch (e) {
        console.error('[Sidebar] Failed to fetch models:', e);
      }
    };

    fetchModels();
  }, []); // Run once on mount

  const handleClearHistory = async () => {
    if (window.confirm('Clear all messages?')) {
      try {
        await fetch('/api/chat/clear', { method: 'POST' });
        clearMessages();
      } catch (e) {
        console.error('Failed to clear history:', e);
      }
    }
  };

  return (
    <>
      <div className="h-full glass flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-chat-border/50 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AuraBreathingAvatar
              isActive={connectionStatus === 'connected'}
              isThinking={isLoading}
              size="md"
            />
            <div>
              <span className="text-chat-text font-bold text-lg">AURA</span>
              <div className="text-xs text-chat-text-secondary">v3.0 ALIVE</div>
              <AuraStatusLine
                status={isLoading ? 'Thinking...' : ambientStatus}
                isVisible={connectionStatus === 'connected'}
              />
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-lg transition-all duration-200 lg:hidden"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Status section */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5">
          {/* Connection status */}
          <div className="flex items-center gap-3 text-sm">
            {connectionStatus === 'connected' ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="connected-dot absolute inline-flex h-full w-full rounded-full bg-green-500 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                </span>
                <span className="text-green-400 font-medium">Connected</span>
              </>
            ) : connectionStatus === 'connecting' ? (
              <>
                <span className="relative flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-yellow-500"></span>
                </span>
                <span className="text-yellow-400 font-medium">Connecting...</span>
              </>
            ) : (
              <>
                <span className="h-3 w-3 rounded-full bg-red-500"></span>
                <span className="text-red-400 font-medium">Disconnected</span>
              </>
            )}
          </div>

          {/* AURA Consideration - "Thinking about saying something" */}
          <AuraConsideringIndicator
            isConsidering={consideration.is_considering}
            decidedAgainst={consideration.decided_against}
            topic={consideration.topic}
          />

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* ALMA Emotion Panel */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              ALMA Emotional State
            </h3>
            <EmotionPanel />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Gateway Daemon - Proactive System */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Proactive System
            </h3>
            <ProactiveDaemonPanel />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Spontaneous Conversation Starters */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Conversation
            </h3>
            <ConversationStarterPanel />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Model Selector */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Model Selection {availableModels.length > 0 && <span className="text-purple-400">({availableModels.length})</span>}
            </h3>
            <div className="relative">
              <button
                onClick={() => setShowModelDropdown(!showModelDropdown)}
                className="w-full flex items-center justify-between p-3 rounded-lg bg-chat-assistant/30 hover:bg-chat-assistant/50 transition-colors duration-200 border border-chat-border/50"
              >
                <span className="text-chat-text text-sm font-medium truncate">
                  {selectedModel || '🤖 Auto (AURA decides)'}
                </span>
                <ChevronDownIcon className={`w-4 h-4 text-chat-text-secondary transition-transform duration-200 ${showModelDropdown ? 'rotate-180' : ''}`} />
              </button>

              {showModelDropdown && (
                <div className="absolute z-50 w-full mt-1 bg-chat-sidebar border border-chat-border rounded-lg shadow-xl max-h-60 overflow-y-auto">
                  <button
                    onClick={() => {
                      setSelectedModel(null);
                      setShowModelDropdown(false);
                    }}
                    className={`w-full text-left px-3 py-2 text-sm hover:bg-chat-assistant/50 transition-colors ${
                      !selectedModel ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                    }`}
                  >
                    🤖 Auto (AURA decides)
                  </button>
                  <div className="border-t border-chat-border/50 my-1" />
                  {availableModels.map((model) => (
                    <button
                      key={model}
                      onClick={() => {
                        setSelectedModel(model);
                        setShowModelDropdown(false);
                      }}
                      className={`w-full text-left px-3 py-2 text-sm hover:bg-chat-assistant/50 transition-colors truncate ${
                        selectedModel === model ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                      }`}
                    >
                      {model.includes('-cloud') ? '☁️ ' : '💻 '}{model}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Memory Recall Indicator */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Memory System
            </h3>
            <MemoryRecallIndicator />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Context Awareness Heatmap */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Context Awareness
            </h3>
            <ContextHeatmap />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Thinking About Teaser */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Thought Process
            </h3>
            <ThinkingAboutTeaser />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Inner Thoughts */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              Inner Monologue
            </h3>
            <InnerThoughtsPanel />
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* System Stats */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              System
            </h3>
            <SystemStatsPanel status={status} />
          </div>
        </div>

        {/* Footer actions */}
        <div className="p-4 border-t border-chat-border/50 space-y-2">
          <button
            onClick={handleClearHistory}
            className="w-full flex items-center gap-3 px-4 py-3 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-xl transition-all duration-200 group"
          >
            <TrashIcon className="w-5 h-5 transition-transform duration-200 group-hover:scale-110" />
            <span>Clear History</span>
          </button>
          <button
            onClick={() => setShowSettings(true)}
            className="w-full flex items-center gap-3 px-4 py-3 text-sm text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant/50 rounded-xl transition-all duration-200 group"
          >
            <Cog6ToothIcon className="w-5 h-5 transition-transform duration-200 group-hover:rotate-90" />
            <span>Settings</span>
          </button>
        </div>
      </div>

      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </>
  );
}
