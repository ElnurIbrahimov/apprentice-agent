import { useRef, useEffect, useState, useCallback } from 'react';
import { useChatStore } from '../store/chatStore';
import { useSettingsStore } from '../store/settingsStore';
import { MessageBubble } from './MessageBubble';
import { MessageInput } from './MessageInput';
import { useWebSocket } from '../hooks/useWebSocket';
import { useProactiveMessages } from '../hooks/useProactiveMessages';
import { haptic } from '../utils/haptics';
import { sounds } from '../utils/sounds';
import { useMoodTheme } from '../hooks/useMoodTheme';
import type { FileAttachment, ModelResult } from '../types';
import { FleetDashboard } from './FleetDashboard';
import { ProactiveCard } from './ProactiveCard';
import { ResearchProgress } from './ResearchProgress';
import { CitationsPanel } from './CitationsPanel';
import { ArtifactsPanel } from './ArtifactsPanel';
import { ConversationList } from './ConversationList';
import { exportAsMarkdown, exportAsJSON, downloadExport } from '../utils/exportConversation';
import { copyText } from '../utils/clipboard';
import { toast } from './Toast';
import type { ArtifactType } from '../utils/artifactRenderer';
import {
  GlobeAltIcon,
  CommandLineIcon,
  ScaleIcon,
  CpuChipIcon,
  ChevronDownIcon,
  XMarkIcon,
  BookOpenIcon,
  EllipsisHorizontalIcon,
  ArrowDownTrayIcon,
  DocumentTextIcon,
  CodeBracketIcon,
} from '@heroicons/react/24/outline';

// Swipe drawer constants
const EDGE_ZONE = 30; // px from left edge to trigger
const DRAWER_WIDTH = 280;
const OPEN_THRESHOLD = 0.35; // fraction of drawer width to snap open

// Quick action button configurations with icons
const QUICK_ACTIONS = [
  {
    text: 'Research the latest breakthroughs in AI',
    subtitle: 'Deep web research with citations',
    icon: GlobeAltIcon,
  },
  {
    text: 'Generate a Python data visualization',
    subtitle: 'Code interpreter with live output',
    icon: CommandLineIcon,
  },
  {
    text: 'Compare models on a creative task',
    subtitle: 'Side-by-side model comparison',
    icon: ScaleIcon,
  },
  {
    text: 'Build me a landing page',
    subtitle: 'Web creator with live preview',
    icon: CpuChipIcon,
  },
];

// Thinking shimmer skeleton
function ThinkingShimmer({ toolStatus }: { toolStatus: { name: string; action: string } }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const label = toolStatus.action === 'thinking'
    ? `Thinking... ${elapsed}s`
    : `Using ${toolStatus.name}... ${elapsed}s`;

  return (
    <div className="py-5 px-4 md:px-8">
      <div className="max-w-3xl mx-auto flex gap-4">
        {/* Avatar placeholder */}
        <div className="flex-shrink-0 mt-1">
          <div className="w-9 h-9 rounded-lg shimmer-bar" />
        </div>
        {/* Shimmer content */}
        <div className="flex-1 min-w-0 space-y-3">
          <span className="text-xs text-chat-text-secondary">{label}</span>
          <div className="shimmer-bar h-3" style={{ width: '80%' }} />
          <div className="shimmer-bar h-3" style={{ width: '60%' }} />
          <div className="shimmer-bar h-3" style={{ width: '40%' }} />
        </div>
      </div>
    </div>
  );
}

// Default follow-up suggestions
const FOLLOW_UP_SETS = [
  ['Tell me more', 'Can you explain that differently?', 'What else should I know?'],
  ['Go deeper on that', 'Give me an example', 'What are the tradeoffs?'],
  ['Summarize that', 'What would you recommend?', 'Any alternatives?'],
];

export function ChatContainer() {
  useMoodTheme();
  const { messages, isLoading, error, setError, connectionStatus, toolStatus, isSwitchingConversation, suggestions, setSuggestions, clearSuggestions, fleetData, clearFleetData, clearResearchProgress, citationsPanelOpen, toggleCitationsPanel } = useChatStore();
  const { sendMessage, stopGeneration, connect: reconnect } = useWebSocket();
  const { settings } = useSettingsStore();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolledUp, setIsUserScrolledUp] = useState(false);
  const initialMessageCountRef = useRef(messages.length);

  // Artifact panel state
  const [artifactCode, setArtifactCode] = useState<string | null>(null);
  const [artifactType, setArtifactType] = useState<ArtifactType>('html');

  const handleOpenArtifact = useCallback((code: string, type: ArtifactType) => {
    setArtifactCode(code);
    setArtifactType(type);
  }, []);
  const prevIsLoadingRef = useRef(isLoading);

  // Thinking history tracking
  const [thinkingHistory, setThinkingHistory] = useState<{ elapsed: number; timestamp: number } | null>(null);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);
  const thinkingStartRef = useRef<number | null>(null);
  const prevToolStatusRef = useRef(toolStatus);

  // --- Proactive notification cards ---
  const [dismissedProactiveIds, setDismissedProactiveIds] = useState<Set<string>>(new Set());

  const proactiveMessages = messages
    .filter(m => m.proactive && !dismissedProactiveIds.has(m.id))
    .slice(-3); // Show max 3 cards

  const handleDismissProactive = useCallback((id: string) => {
    setDismissedProactiveIds(prev => new Set(prev).add(id));
  }, []);

  // --- Export menu ---
  const [exportMenuOpen, setExportMenuOpen] = useState(false);

  const handleExportMarkdown = useCallback(() => {
    const md = exportAsMarkdown(messages);
    downloadExport(md, `aura-chat-${Date.now()}.md`, 'text/markdown');
    toast.success('Exported as Markdown');
    setExportMenuOpen(false);
  }, [messages]);

  const handleExportJSON = useCallback(() => {
    const json = exportAsJSON(messages);
    downloadExport(json, `aura-chat-${Date.now()}.json`, 'application/json');
    toast.success('Exported as JSON');
    setExportMenuOpen(false);
  }, [messages]);

  const [copyFeedback, setCopyFeedback] = useState(false);
  const handleCopyConversation = useCallback(async () => {
    const md = exportAsMarkdown(messages);
    if (await copyText(md)) {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 1500);
    }
    setExportMenuOpen(false);
  }, [messages]);

  // --- Swipe drawer state ---
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerDragging, setDrawerDragging] = useState(false);
  const [drawerTranslateX, setDrawerTranslateX] = useState(-DRAWER_WIDTH);
  const [edgeHintActive, setEdgeHintActive] = useState(false);
  const drawerTouchStartXRef = useRef(0);
  const drawerTouchStartYRef = useRef(0);
  const drawerActiveRef = useRef(false); // whether this swipe is a drawer gesture
  const drawerTranslateXRef = useRef(-DRAWER_WIDTH);

  const handleDrawerTouchStart = useCallback((e: React.TouchEvent) => {
    if (window.innerWidth >= 768) return;
    const touchX = e.touches[0].clientX;
    if (touchX <= EDGE_ZONE && !drawerOpen) {
      drawerTouchStartXRef.current = touchX;
      drawerTouchStartYRef.current = e.touches[0].clientY;
      drawerActiveRef.current = true;
      setDrawerDragging(true);
      setEdgeHintActive(true);
    }
  }, [drawerOpen]);

  const handleDrawerTouchMove = useCallback((e: React.TouchEvent) => {
    if (!drawerActiveRef.current) return;
    const deltaX = e.touches[0].clientX - drawerTouchStartXRef.current;
    const deltaY = Math.abs(e.touches[0].clientY - drawerTouchStartYRef.current);
    // If vertical scroll is dominant, cancel drawer gesture
    if (deltaY > 30 && deltaX < 20) {
      drawerActiveRef.current = false;
      setDrawerDragging(false);
      setEdgeHintActive(false);
      setDrawerTranslateX(-DRAWER_WIDTH);
      return;
    }
    const clamped = Math.min(Math.max(deltaX - DRAWER_WIDTH, -DRAWER_WIDTH), 0);
    drawerTranslateXRef.current = clamped;
    setDrawerTranslateX(clamped);
  }, []);

  const handleDrawerTouchEnd = useCallback(() => {
    if (!drawerActiveRef.current) return;
    drawerActiveRef.current = false;
    setDrawerDragging(false);
    setEdgeHintActive(false);
    // Read from ref to avoid stale closure (state may not have re-rendered yet)
    const currentX = drawerTranslateXRef.current;
    const progress = (currentX + DRAWER_WIDTH) / DRAWER_WIDTH;
    if (progress > OPEN_THRESHOLD) {
      setDrawerOpen(true);
      setDrawerTranslateX(0);
      drawerTranslateXRef.current = 0;
    } else {
      setDrawerOpen(false);
      setDrawerTranslateX(-DRAWER_WIDTH);
      drawerTranslateXRef.current = -DRAWER_WIDTH;
    }
  }, []);

  const closeDrawer = useCallback(() => {
    setDrawerOpen(false);
    setDrawerTranslateX(-DRAWER_WIDTH);
  }, []);

  // Track thinking start/end transitions
  useEffect(() => {
    const prev = prevToolStatusRef.current;
    if (!prev && toolStatus) {
      // Thinking just started
      thinkingStartRef.current = Date.now();
      setThinkingHistory(null);
      setThinkingExpanded(false);
    } else if (prev && !toolStatus && thinkingStartRef.current) {
      // Thinking just ended
      const elapsed = Math.floor((Date.now() - thinkingStartRef.current) / 1000);
      setThinkingHistory({ elapsed, timestamp: Date.now() });
      setThinkingExpanded(false);
      thinkingStartRef.current = null;
    }
    prevToolStatusRef.current = toolStatus;
  }, [toolStatus]);

  useProactiveMessages(connectionStatus === 'connected');

  // Handle scroll events to detect if user scrolled up
  const handleScroll = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return;
    const distFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
    setIsUserScrolledUp(distFromBottom > 80);
  }, []);

  // Smart auto-scroll: only scroll if user is already at bottom and autoScroll is enabled
  useEffect(() => {
    if (settings.autoScroll && !isUserScrolledUp) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isUserScrolledUp, settings.autoScroll]);

  // Auto-scroll to bottom when user sends a new message (always, regardless of autoScroll setting)
  useEffect(() => {
    if (isLoading) {
      setIsUserScrolledUp(false);
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isLoading]);

  // Show suggestion chips + play receive sound after response completes
  useEffect(() => {
    if (prevIsLoadingRef.current && !isLoading && messages.length > 0) {
      const set = FOLLOW_UP_SETS[Math.floor(Math.random() * FOLLOW_UP_SETS.length)];
      setSuggestions(set);
      if (settings.soundEnabled) sounds.receive();
    }
    prevIsLoadingRef.current = isLoading;
  }, [isLoading, messages.length, setSuggestions, settings.soundEnabled]);

  // Auto-dismiss error after 10 seconds + haptic/sound on error
  useEffect(() => {
    if (!error) return;
    haptic(100);
    if (settings.soundEnabled) sounds.error();
    const timer = setTimeout(() => setError(null), 30000);
    return () => clearTimeout(timer);
  }, [error, setError, settings.soundEnabled]);

  const handleSend = useCallback(async (message: string, attachments?: FileAttachment[], actionMode?: string | null) => {
    clearSuggestions();
    clearResearchProgress();
    setThinkingHistory(null);
    setThinkingExpanded(false);
    initialMessageCountRef.current = useChatStore.getState().messages.length;
    if (actionMode === 'compare') {
      // Route through REST /api/compare instead of WebSocket
      const store = useChatStore.getState();
      store.addMessage({ role: 'user', content: message });
      const assistantId = store.addMessage({ role: 'assistant', content: 'Comparing models...', isStreaming: true });
      store.setIsLoading(true);

      try {
        const res = await fetch('/api/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message }),
        });
        if (!res.ok) {
          throw new Error(`Compare request failed: ${res.status}`);
        }
        const data = await res.json();
        const results: ModelResult[] = data.results ?? [];
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: '', compareResults: results, isStreaming: false }
              : m
          ),
          isLoading: false,
        }));
      } catch {
        useChatStore.setState((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantId
              ? { ...m, content: 'Compare failed. Please try again.', isStreaming: false }
              : m
          ),
          isLoading: false,
        }));
      }
      store.setToolStatus(null);
      return;
    }
    // Pass actionMode for auto-model selection (null = use user's selected model)
    sendMessage(message, attachments, undefined, actionMode);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearSuggestions, clearResearchProgress, sendMessage]);

  const isDisabled = isLoading || connectionStatus !== 'connected';
  const hasCitationsInConversation = messages.some(m => m.citations && m.citations.length > 0);

  return (
    <div className="flex h-full" style={{ background: 'transparent' }}>
    {/* Main chat column */}
    <div className="flex flex-col flex-1 min-w-0">
      {/* Proactive notification cards */}
      {proactiveMessages.length > 0 && (
        <div className="proactive-container">
          {proactiveMessages.map(msg => (
            <ProactiveCard
              key={msg.id}
              action={msg.proactive!.action}
              content={msg.content}
              trigger={msg.proactive!.trigger}
              onDismiss={() => handleDismissProactive(msg.id)}
            />
          ))}
        </div>
      )}

      {/* Chat toolbar — export menu */}
      {messages.length > 0 && (
        <div className="flex justify-end px-4 py-1.5 relative z-20">
          <div className="relative">
            <button
              onClick={() => setExportMenuOpen(!exportMenuOpen)}
              className="p-1.5 rounded-lg text-chat-text-secondary hover:text-chat-text hover:bg-white/[0.06] transition-colors"
              aria-label="Chat options"
            >
              <EllipsisHorizontalIcon className="w-5 h-5" />
            </button>
            {exportMenuOpen && (
              <>
                <div className="fixed inset-0 z-[40]" onClick={() => setExportMenuOpen(false)} />
                <div className="absolute right-0 top-full mt-1 z-[50] ctx-menu min-w-[180px]">
                  <button className="ctx-menu-item w-full" onClick={handleExportMarkdown}>
                    <DocumentTextIcon className="w-4 h-4" />
                    Export as Markdown
                  </button>
                  <button className="ctx-menu-item w-full" onClick={handleExportJSON}>
                    <CodeBracketIcon className="w-4 h-4" />
                    Export as JSON
                  </button>
                  <button className="ctx-menu-item w-full" onClick={handleCopyConversation}>
                    <ArrowDownTrayIcon className="w-4 h-4" />
                    {copyFeedback ? 'Copied!' : 'Copy to Clipboard'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Connection status banner */}
      {connectionStatus !== 'connected' && (
        <div className={`px-4 py-2 text-center text-sm transition-all duration-300 ${
          connectionStatus === 'connecting'
            ? 'bg-yellow-600/90 text-white backdrop-blur-sm'
            : 'bg-red-600/90 text-white backdrop-blur-sm'
        }`}>
          {connectionStatus === 'connecting' ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-white rounded-full animate-pulse" />
              Connecting to AURA...
            </span>
          ) : (
            <span className="flex items-center justify-center gap-2">
              <span className="w-2 h-2 bg-white rounded-full" />
              Disconnected.
              <button onClick={reconnect} className="underline hover:no-underline font-medium ml-1">
                Reconnect now
              </button>
            </span>
          )}
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-4 py-2 bg-red-900/80 text-red-200 text-sm backdrop-blur-sm animate-slide-up-fade flex items-center gap-2">
          <span className="flex-1 text-center">{error}</span>
          <button onClick={() => setError(null)} aria-label="Dismiss error" className="ml-auto flex-shrink-0">
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Swipe drawer edge hint */}
      <div className={`swipe-edge-hint${edgeHintActive ? ' hint-active' : ''}`} />

      {/* Swipe drawer overlay */}
      <div
        className={`swipe-drawer-overlay${drawerOpen ? ' drawer-visible' : ''}`}
        onClick={closeDrawer}
        aria-hidden="true"
      />

      {/* Swipe drawer */}
      <div
        className={`swipe-drawer${drawerOpen ? ' drawer-open' : ''}`}
        style={drawerDragging ? { transform: `translateX(${drawerTranslateX}px)`, transition: 'none' } : undefined}
      >
        <div className="swipe-drawer-header flex items-center justify-between">
          <span>Conversations</span>
          <button
            onClick={closeDrawer}
            className="p-1 rounded-md text-gray-400 hover:text-gray-200 hover:bg-white/[0.06] transition-colors"
            aria-label="Close drawer"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        </div>
        <div className="swipe-drawer-body">
          <ConversationList />
        </div>
      </div>

      {/* Fleet dashboard */}
      {fleetData && (
        <FleetDashboard
          goal={fleetData.goal}
          tasks={fleetData.tasks}
          totalElapsed={fleetData.totalElapsed}
          onClose={clearFleetData}
        />
      )}

      {/* Messages area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        onTouchStart={handleDrawerTouchStart}
        onTouchMove={handleDrawerTouchMove}
        onTouchEnd={handleDrawerTouchEnd}
        style={{ viewTransitionName: 'chat-messages' }}
        className="flex-1 overflow-y-auto relative messages-scroll">
        {isSwitchingConversation && messages.length === 0 ? (
          <div className="animate-pulse space-y-4 p-4">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-16 bg-chat-border/20 rounded-lg" />
            ))}
          </div>
        ) : messages.length === 0 ? (
          // Empty state with enhanced styling
          <div className="flex flex-col items-center justify-center h-full text-chat-text-secondary relative px-6">
            {/* NextGen welcome heading */}
            <h1 className="text-5xl font-light tracking-tight mb-4 text-center animate-fade-in"
              style={{
                background: 'linear-gradient(180deg, #fff 0%, rgba(255,255,255,0.4) 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                letterSpacing: '-0.04em',
              }}
            >
              What should we explore?
            </h1>

            <p className="text-center max-w-md text-chat-text-secondary mb-4 leading-relaxed animate-fade-in animation-delay-100">
              Research, create, code, and compare — powered by 40+ AI models.
            </p>
            <p className="text-center text-xs text-chat-text-tertiary mb-10 animate-fade-in animation-delay-200">
              Upload files, use voice, or pick a starting point
            </p>

            {/* NextGen suggestion cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl">
              {QUICK_ACTIONS.map((action, index) => {
                const Icon = action.icon;
                return (
                  <button
                    key={action.text}
                    onClick={() => handleSend(action.text)}
                    disabled={isDisabled}
                    aria-label={action.text}
                    className={`
                      group flex flex-col gap-3 p-5 text-left
                      disabled:opacity-50 disabled:cursor-not-allowed
                      animate-slide-up-fade transition-all duration-300
                      bg-surface-1 border border-chat-border rounded-xl backdrop-blur-sm
                      hover:bg-surface-2 hover:border-chat-text-secondary/20
                      ${index === 0 ? 'animation-delay-100' : ''}
                      ${index === 1 ? 'animation-delay-200' : ''}
                      ${index === 2 ? 'animation-delay-300' : ''}
                      ${index === 3 ? 'animation-delay-400' : ''}
                    `}
                  >
                    <div className="w-9 h-9 rounded-lg bg-surface-2 flex items-center justify-center">
                      <Icon className="w-4 h-4 text-chat-text" />
                    </div>
                    <span className="text-sm font-medium text-chat-text group-hover:text-white transition-colors">
                      {action.text}
                    </span>
                    <span className="text-xs text-chat-text-secondary mt-0.5">
                      {action.subtitle}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          // Message list
          <div className="pb-4">
            {messages.map((message, index) => {
              const isNew = index >= initialMessageCountRef.current;
              const animIndex = isNew ? index - initialMessageCountRef.current : 0;
              return (
                <MessageBubble
                  key={message.id}
                  message={message}
                  animateIn={isNew}
                  animationIndex={animIndex}
                  onRegenerate={handleSend}
                  onStop={message.isStreaming ? stopGeneration : undefined}
                  onOpenArtifact={handleOpenArtifact}
                />
              );
            })}
            <ResearchProgress />
            <div ref={messagesEndRef} />
          </div>
        )}
        {isUserScrolledUp && messages.length > 0 && (
          <button
            onClick={() => {
              setIsUserScrolledUp(false);
              messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
            }}
            className="absolute bottom-24 right-6 z-10 p-2 bg-purple-600 hover:bg-purple-500 text-white rounded-full shadow-lg transition-all"
            aria-label="Scroll to bottom"
          >
            <ChevronDownIcon className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Tool status / thinking shimmer */}
      {toolStatus && settings.showThinking && (
        <ThinkingShimmer toolStatus={toolStatus} />
      )}

      {/* Collapsed thinking history pill */}
      {!toolStatus && thinkingHistory && settings.showThinking && (
        <div className="px-4 py-2">
          <div className="max-w-3xl mx-auto">
            <button
              onClick={() => setThinkingExpanded(prev => !prev)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '4px 12px',
                fontSize: '12px',
                color: 'var(--text-secondary)',
                background: 'var(--border-subtle)',
                border: '1px solid var(--border-default)',
                borderRadius: '999px',
                cursor: 'pointer',
                backdropFilter: 'blur(8px)',
                transition: 'all 0.2s ease',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = 'var(--surface-2)';
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-strong)';
                (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'var(--border-subtle)';
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--border-default)';
                (e.currentTarget as HTMLElement).style.color = 'var(--text-secondary)';
              }}
            >
              <ChevronDownIcon
                className="w-3 h-3"
                style={{
                  transform: thinkingExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s ease',
                }}
              />
              Thought for {thinkingHistory.elapsed}s
            </button>

            {thinkingExpanded && (
              <div
                style={{
                  marginTop: '8px',
                  padding: '12px 16px',
                  background: 'var(--surface-0)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '10px',
                  animation: 'fadeIn 0.2s ease',
                }}
              >
                <div className="flex gap-4">
                  <div className="flex-shrink-0 mt-1">
                    <div
                      className="w-9 h-9 rounded-lg"
                      style={{ background: 'var(--surface-2)' }}
                    />
                  </div>
                  <div className="flex-1 min-w-0 space-y-3">
                    <span className="text-xs text-chat-text-secondary">
                      Thought for {thinkingHistory.elapsed}s
                    </span>
                    <div style={{ height: 12, width: '80%', background: 'var(--border-subtle)', borderRadius: 4 }} />
                    <div style={{ height: 12, width: '60%', background: 'var(--border-subtle)', borderRadius: 4 }} />
                    <div style={{ height: 12, width: '40%', background: 'var(--border-subtle)', borderRadius: 4 }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Suggestion chips */}
      {suggestions.length > 0 && !isLoading && (
        <div className="px-4 pb-2 relative">
          <div className="overflow-x-auto scrollbar-hide">
            <div className="max-w-3xl mx-auto flex gap-2 pr-8">
              {suggestions.map((text, i) => (
                <button
                  key={text}
                  onClick={() => {
                    clearSuggestions();
                    handleSend(text);
                  }}
                  className={`suggestion-chip chip-delay-${i} whitespace-nowrap flex-shrink-0`}
                >
                  {text}
                </button>
              ))}
            </div>
          </div>
          <div
            className="absolute right-4 top-0 bottom-2 w-12 pointer-events-none"
            style={{ background: 'linear-gradient(to right, transparent, var(--bg-base))' }}
          />
        </div>
      )}

      {/* Input area */}
      <MessageInput
        onSend={handleSend}
        onStop={stopGeneration}
        disabled={isDisabled}
        isLoading={isLoading}
        onTypingStart={clearSuggestions}
        placeholder={
          connectionStatus !== 'connected'
            ? 'Connecting...'
            : isLoading
            ? 'AURA is thinking...'
            : 'Message AURA...'
        }
      />

      {/* Citations panel toggle — floating button (always visible when citations exist) */}
      {hasCitationsInConversation && (
        <button
          onClick={toggleCitationsPanel}
          className="fixed bottom-24 right-6 z-20 flex items-center gap-1.5 px-3 py-2 rounded-full text-xs font-medium transition-all duration-200"
          style={{
            background: citationsPanelOpen
              ? 'rgba(139, 92, 246, 0.3)'
              : 'var(--bg-panel)',
            border: `1px solid ${citationsPanelOpen ? 'rgba(139, 92, 246, 0.5)' : 'var(--border-default)'}`,
            color: citationsPanelOpen ? '#e9d5ff' : 'var(--text-secondary)',
            backdropFilter: 'blur(12px)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
          }}
          aria-label={citationsPanelOpen ? 'Close sources panel' : 'Open sources panel'}
        >
          {citationsPanelOpen ? (
            <><XMarkIcon className="w-4 h-4" />Close</>
          ) : (
            <><BookOpenIcon className="w-4 h-4" />Sources</>
          )}
        </button>
      )}
    </div>

    {/* Citations panel (right sidebar) */}
    <CitationsPanel />

    {/* Artifacts panel — side panel on desktop, full-screen on mobile */}
    {artifactCode && (
      <>
        <div className="fixed inset-0 z-[99] bg-black/50 backdrop-blur-sm md:hidden" onClick={() => setArtifactCode(null)} />
        <div className="fixed inset-0 z-[100] artifact-panel-enter md:static md:inset-auto md:z-auto md:w-[45%] md:min-w-[360px] md:max-w-[600px] md:flex-shrink-0">
          <ArtifactsPanel
            code={artifactCode}
            type={artifactType}
            onClose={() => setArtifactCode(null)}
          />
        </div>
      </>
    )}
    </div>
  );
}
