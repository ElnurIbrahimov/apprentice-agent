import { useEffect, useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { usePolling } from '../hooks/usePolling';
import { EmotionPanel } from './EmotionPanel';
import { SettingsModal } from './SettingsModal';
import { AuraBreathingAvatar, AuraStatusLine } from './AuraBreathingAvatar';
import { SystemStatsPanel } from './SystemStatsPanel';
import { ConversationList } from './ConversationList';
import { ProactiveDaemonPanel } from './ProactiveDaemonPanel';
import { InnerThoughtsPanel } from './InnerThoughtsPanel';
import { ThinkingAboutTeaser } from './ThinkingAboutTeaser';
import { MemoryRecallIndicator } from './MemoryRecallIndicator';
import { ContextHeatmap } from './ContextHeatmap';
import { IdleBehaviorPanel } from './IdleBehaviorPanel';
import { MotivationDrivesPanel } from './MotivationDrivesPanel';
import {
  XMarkIcon,
  Cog6ToothIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline';

interface SidebarProps {
  onClose?: () => void;
}

export function Sidebar({ onClose }: SidebarProps) {
  const [showSettings, setShowSettings] = useState(false);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [modelFilter, setModelFilter] = useState('');

  const {
    status,
    setStatus,
    connectionStatus,
    selectedModel,
    setSelectedModel,
    availableModels,
    setAvailableModels,
    isLoading,
  } = useChatStore();

  // Keyboard shortcut: Ctrl+/ toggle settings
  useEffect(() => {
    const handler = () => setShowSettings(prev => !prev);
    document.addEventListener('aura:toggle-settings', handler);
    return () => document.removeEventListener('aura:toggle-settings', handler);
  }, []);

  // Poll status only (10s - lightweight, just /api/status)
  const fetchStatus = useCallback(async () => {
    try {
      const response = await fetch('/api/status');
      if (response.ok) {
        const data = await response.json();
        setStatus(data);
      }
    } catch {
      // Ignore - status is secondary to WebSocket connection
    }
  }, [setStatus]);
  usePolling(fetchStatus, 30000);

  // Fetch available models on mount (retry once on failure)
  useEffect(() => {
    let cancelled = false;
    const fetchModels = async (attempt = 0) => {
      try {
        const response = await fetch('/api/models');
        if (response.ok) {
          const data = await response.json();
          const allModels = [...(data.chatgpt_models || []), ...(data.cloud_models || []), ...(data.local_models || [])];
          if (!cancelled) setAvailableModels(allModels);
        } else if (attempt < 1) {
          setTimeout(() => { if (!cancelled) fetchModels(attempt + 1); }, 3000);
        }
      } catch (e) {
        console.error('[Sidebar] Failed to fetch models:', e);
        if (attempt < 1) {
          setTimeout(() => { if (!cancelled) fetchModels(attempt + 1); }, 3000);
        }
      }
    };
    fetchModels();
    return () => { cancelled = true; };
  }, [setAvailableModels]);

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
              <div className="text-xs text-chat-text-secondary">v4.3.0 ALIVE</div>
              <AuraStatusLine
                status={isLoading ? 'Thinking...' : null}
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

        {/* Conversations */}
        <div className="px-4 pt-3 pb-1">
          <ConversationList />
        </div>

        {/* Gradient divider */}
        <div className="mx-4 divider-gradient" />

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

          {/* ALMA Emotion Panel */}
          <div>
            <h3 className="text-chat-text-secondary text-xs uppercase tracking-wider mb-3 font-medium">
              ALMA Emotional State
            </h3>
            <EmotionPanel />
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
                onClick={() => { setShowModelDropdown(!showModelDropdown); if (showModelDropdown) setModelFilter(''); }}
                className="w-full flex items-center justify-between p-3 rounded-lg bg-chat-assistant/30 hover:bg-chat-assistant/50 transition-colors duration-200 border border-chat-border/50"
              >
                <span className="text-chat-text text-sm font-medium truncate">
                  {selectedModel || '🤖 Auto (AURA decides)'}
                </span>
                <ChevronDownIcon className={`w-4 h-4 text-chat-text-secondary transition-transform duration-200 ${showModelDropdown ? 'rotate-180' : ''}`} />
              </button>

              {showModelDropdown && (
                <div className="absolute z-50 w-full mt-1 bg-chat-sidebar border border-chat-border rounded-lg shadow-xl overflow-hidden"
                  style={{ maxHeight: '280px', display: 'flex', flexDirection: 'column' }}
                >
                  {/* Search filter */}
                  <div className="p-2 border-b border-chat-border/50 flex-shrink-0">
                    <input
                      type="text"
                      placeholder="Search models..."
                      autoFocus
                      value={modelFilter}
                      className="w-full px-2 py-1.5 text-xs bg-chat-bg border border-chat-border/50 rounded text-chat-text placeholder:text-chat-text-secondary/50 outline-none focus:border-purple-500/50"
                      onChange={(e) => setModelFilter(e.target.value.toLowerCase())}
                    />
                  </div>
                  <div className="overflow-y-auto flex-1" style={{ maxHeight: '240px' }}>
                    {(() => {
                      const filtered = modelFilter
                        ? availableModels.filter(m => m.toLowerCase().includes(modelFilter))
                        : availableModels;
                      const chatgptModels = filtered.filter(m => m.startsWith('chatgpt:'));
                      const cloudModels = filtered.filter(m => !m.startsWith('chatgpt:') && (m.includes(':cloud') || m.includes('-cloud')));
                      const localModels = filtered.filter(m => !m.startsWith('chatgpt:') && !m.includes(':cloud') && !m.includes('-cloud'));
                      return (
                        <>
                          {'auto'.includes(modelFilter) && (
                            <button
                              onClick={() => {
                                setSelectedModel(null);
                                setShowModelDropdown(false);
                                setModelFilter('');
                              }}
                              className={`w-full text-left px-3 py-1.5 text-xs hover:bg-chat-assistant/50 transition-colors ${
                                !selectedModel ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                              }`}
                            >
                              🤖 Auto (AURA decides)
                            </button>
                          )}

                          {/* ChatGPT models */}
                          {chatgptModels.length > 0 && (
                            <>
                              <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-green-400/70 font-semibold bg-green-500/5 border-t border-chat-border/30">
                                ChatGPT (subscription)
                              </div>
                              {chatgptModels.map((model) => (
                                <button
                                  key={model}
                                  onClick={() => { setSelectedModel(model); setShowModelDropdown(false); setModelFilter(''); }}
                                  className={`w-full text-left px-3 py-1.5 text-xs hover:bg-chat-assistant/50 transition-colors truncate ${
                                    selectedModel === model ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                                  }`}
                                >
                                  🟢 {model.replace('chatgpt:', '')}
                                </button>
                              ))}
                            </>
                          )}

                          {/* Cloud models */}
                          {cloudModels.length > 0 && (
                            <>
                              <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-blue-400/70 font-semibold bg-blue-500/5 border-t border-chat-border/30">
                                Cloud (Ollama)
                              </div>
                              {cloudModels.map((model) => (
                                <button
                                  key={model}
                                  onClick={() => { setSelectedModel(model); setShowModelDropdown(false); setModelFilter(''); }}
                                  className={`w-full text-left px-3 py-1.5 text-xs hover:bg-chat-assistant/50 transition-colors truncate ${
                                    selectedModel === model ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                                  }`}
                                >
                                  ☁️ {model.replace(/:cloud$/, '')}
                                </button>
                              ))}
                            </>
                          )}

                          {/* Local models */}
                          {localModels.length > 0 && (
                            <>
                              <div className="px-3 py-1 text-[10px] uppercase tracking-wider text-orange-400/70 font-semibold bg-orange-500/5 border-t border-chat-border/30">
                                Local
                              </div>
                              {localModels.map((model) => (
                                <button
                                  key={model}
                                  onClick={() => { setSelectedModel(model); setShowModelDropdown(false); setModelFilter(''); }}
                                  className={`w-full text-left px-3 py-1.5 text-xs hover:bg-chat-assistant/50 transition-colors truncate ${
                                    selectedModel === model ? 'bg-purple-600/20 text-purple-400' : 'text-chat-text'
                                  }`}
                                >
                                  💻 {model}
                                </button>
                              ))}
                            </>
                          )}
                        </>
                      );
                    })()}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Gradient divider */}
          <div className="divider-gradient" />

          {/* Sidebar panels - enabling one by one to find culprit */}
          <ProactiveDaemonPanel />
          <InnerThoughtsPanel />
          <MemoryRecallIndicator />
          <ContextHeatmap />
          <ThinkingAboutTeaser />
          <IdleBehaviorPanel />
          <MotivationDrivesPanel />

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
        <div className="p-4 border-t border-chat-border/50">
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
