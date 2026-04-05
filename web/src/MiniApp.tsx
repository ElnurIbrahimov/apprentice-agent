// Telegram Mini App — embedded UI inside Telegram
// Uses Telegram's WebApp API: window.Telegram.WebApp

import { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { EMOTION_COLORS, NEURO_INFO, PERSONALITY_INFO } from './utils/emotionConstants';

// ─── Telegram WebApp type declarations ───
declare global {
  interface Window {
    Telegram?: {
      WebApp: {
        ready: () => void;
        expand: () => void;
        close: () => void;
        setHeaderColor: (color: string) => void;
        setBackgroundColor: (color: string) => void;
        sendData: (data: string) => void;
        initDataUnsafe?: {
          user?: {
            id: number;
            first_name: string;
            last_name?: string;
            username?: string;
            photo_url?: string;
          };
        };
        themeParams?: {
          bg_color?: string;
          text_color?: string;
          hint_color?: string;
          link_color?: string;
          button_color?: string;
          button_text_color?: string;
          secondary_bg_color?: string;
        };
        MainButton: {
          text: string;
          setText: (text: string) => void;
          show: () => void;
          hide: () => void;
          onClick: (cb: () => void) => void;
          offClick: (cb: () => void) => void;
          setParams: (params: { text?: string; color?: string; text_color?: string; is_active?: boolean; is_visible?: boolean }) => void;
        };
        BackButton: {
          show: () => void;
          hide: () => void;
          onClick: (cb: () => void) => void;
          offClick: (cb: () => void) => void;
        };
        HapticFeedback: {
          impactOccurred: (style: 'light' | 'medium' | 'heavy' | 'rigid' | 'soft') => void;
          notificationOccurred: (type: 'error' | 'success' | 'warning') => void;
          selectionChanged: () => void;
        };
        colorScheme: 'dark' | 'light';
        isExpanded: boolean;
        viewportHeight: number;
        viewportStableHeight: number;
        platform: string;
      };
    };
  }
}

// ─── Constants ───
const API_BASE = `${window.location.protocol}//${window.location.host}/api`;
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/chat/stream`;

type TabId = 'chat' | 'dashboard' | 'tools' | 'emotion' | 'settings';

// ─── Interfaces ───
interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

interface AuraStatus {
  enabled: boolean;
  mood: string;
  energy: number;
  warmth: number;
  engagement: number;
  soul_name: string;
  patterns_learned: number;
  turns: number;
}

interface ALMAState {
  available: boolean;
  dominant_emotion: string;
  intensity: number;
  pad: { pleasure: number; arousal: number; dominance: number };
  mood: { label?: string; intensity?: number };
  active_emotions: Array<{ name: string; intensity: number; current_intensity: number; trigger: string }>;
  neuromodulators: { dopamine: number; serotonin: number; norepinephrine: number; oxytocin: number };
  personality: { openness: number; conscientiousness: number; extraversion: number; agreeableness: number; neuroticism: number };
}

interface StatusInfo {
  online: boolean;
  model: string;
  memory_count: number;
  query_count: number;
  last_model_used: string | null;
}

// ─── Shared constants ───
// EMOTION_COLORS, NEURO_INFO, PERSONALITY_INFO imported from utils/emotionConstants

const MOOD_EMOJIS: Record<string, string> = {
  excited: '\u{1F31F}', happy: '\u{1F60A}', content: '\u{1F642}', neutral: '\u{1F610}',
  thoughtful: '\u{1F914}', tired: '\u{1F634}', concerned: '\u{1F61F}', frustrated: '\u{1F624}',
};

// ═══════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════
export default function MiniApp() {
  const tg = window.Telegram?.WebApp;
  const [activeTab, setActiveTab] = useState<TabId>('chat');
  const [validated, setValidated] = useState(false);
  // Ref to allow tool tab to send messages into the chat WebSocket
  const sendToChatRef = useRef<((text: string) => void) | null>(null);

  // ─── Telegram WebApp init ───
  useEffect(() => {
    if (!tg) return;
    tg.ready();
    tg.expand();
    tg.setHeaderColor('#030303');
    tg.setBackgroundColor('#030303');

    // Validate initData with backend
    const initData = (tg as any).initData;
    if (initData) {
      fetch(`${API_BASE}/telegram/validate-init`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ init_data: initData }),
      })
        .then(r => r.json())
        .then(d => { if (d.valid) setValidated(true); })
        .catch(() => {});
    }
  }, [tg]);

  // ─── Back button toggles to chat ───
  useEffect(() => {
    if (!tg) return;
    const handleBack = () => {
      setActiveTab('chat');
    };
    if (activeTab !== 'chat') {
      tg.BackButton.show();
      tg.BackButton.onClick(handleBack);
    } else {
      tg.BackButton.hide();
    }
    return () => {
      tg.BackButton.offClick(handleBack);
    };
  }, [activeTab, tg]);

  const user = tg?.initDataUnsafe?.user;

  return (
    <div className="miniapp-shell">
      {/* Header */}
      <header className="miniapp-header">
        <div className="miniapp-header-left">
          <BreathingDot />
          <span className="miniapp-title">AURA</span>
        </div>
        {user && (
          <span className="miniapp-user">
            {user.first_name}
            {validated && <span className="miniapp-verified" title="Verified">{'\u2713'}</span>}
          </span>
        )}
      </header>

      {/* Tab content */}
      <main className="miniapp-content">
        {activeTab === 'chat' && <ChatTab tg={tg} sendToChatRef={sendToChatRef} />}
        {activeTab === 'dashboard' && <DashboardTab />}
        {activeTab === 'tools' && <ToolsTab tg={tg} onRunCommand={(cmd) => {
          // Switch to chat tab and send the command via WebSocket
          if (sendToChatRef.current) {
            sendToChatRef.current(cmd);
          }
          setActiveTab('chat');
          tg?.HapticFeedback?.impactOccurred('medium');
        }} />}
        {activeTab === 'emotion' && <EmotionTab />}
        {activeTab === 'settings' && <SettingsTab tg={tg} />}
      </main>

      {/* Bottom tab bar */}
      <nav className="miniapp-tabbar">
        {([
          { id: 'chat' as TabId, label: 'Chat', icon: ChatIcon },
          { id: 'dashboard' as TabId, label: 'Dashboard', icon: DashIcon },
          { id: 'tools' as TabId, label: 'Tools', icon: ToolsIcon },
          { id: 'emotion' as TabId, label: 'Emotion', icon: EmotionIcon },
          { id: 'settings' as TabId, label: 'Settings', icon: SettingsIcon },
        ]).map((tab) => (
          <button
            key={tab.id}
            className={`miniapp-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => {
              setActiveTab(tab.id);
              tg?.HapticFeedback?.selectionChanged();
            }}
          >
            <tab.icon active={activeTab === tab.id} />
            <span>{tab.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}


// ═══════════════════════════════════════════
// BREATHING DOT (simplified from AuraBreathingAvatar)
// ═══════════════════════════════════════════
function BreathingDot() {
  return (
    <div className="breathing-dot">
      <div className="breathing-dot-glow" />
      <div className="breathing-dot-core" />
    </div>
  );
}


// ═══════════════════════════════════════════
// CHAT TAB
// ═══════════════════════════════════════════
function ChatTab({ tg, sendToChatRef }: { tg?: NonNullable<Window['Telegram']>['WebApp']; sendToChatRef: React.MutableRefObject<((text: string) => void) | null> }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const currentMsgIdRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        setConnected(false);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWSMessage(data);
        } catch {
          // ignore parse errors
        }
      };
    };

    const handleWSMessage = (data: Record<string, unknown>) => {
      switch (data.type) {
        case 'chunk':
          if (data.content) {
            if (!currentMsgIdRef.current) {
              const id = `msg_${Date.now()}`;
              currentMsgIdRef.current = id;
              setMessages(prev => [...prev, {
                id,
                role: 'assistant',
                content: data.content as string,
                timestamp: Date.now(),
                isStreaming: true,
              }]);
            } else {
              const msgId = currentMsgIdRef.current;
              setMessages(prev => prev.map(m =>
                m.id === msgId ? { ...m, content: m.content + (data.content as string) } : m
              ));
            }
          }
          break;

        case 'done':
          if (currentMsgIdRef.current) {
            const msgId = currentMsgIdRef.current;
            setMessages(prev => prev.map(m =>
              m.id === msgId ? { ...m, isStreaming: false } : m
            ));
          }
          currentMsgIdRef.current = null;
          setIsLoading(false);
          break;

        case 'error':
          currentMsgIdRef.current = null;
          setIsLoading(false);
          break;

        case 'pong':
          break;
      }
    };

    connect();

    // Heartbeat
    const heartbeat = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);

    return () => {
      clearInterval(heartbeat);
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, []);

  // Telegram MainButton for sending
  useEffect(() => {
    if (!tg) return;
    const handleSend = () => {
      if (input.trim()) {
        sendMessage(input.trim());
      }
    };
    if (input.trim()) {
      tg.MainButton.setText('Send');
      tg.MainButton.setParams({ color: '#7c3aed', text_color: '#ffffff', is_active: true, is_visible: true });
      tg.MainButton.show();
      tg.MainButton.onClick(handleSend);
    } else {
      tg.MainButton.hide();
    }
    return () => {
      tg.MainButton.offClick(handleSend);
    };
  }, [input, tg]);

  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    const userMsg: ChatMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    wsRef.current.send(JSON.stringify({
      type: 'chat',
      message: text,
    }));

    tg?.HapticFeedback?.impactOccurred('light');
  }, [tg]);

  // Expose sendMessage to parent via ref (for tools tab)
  useEffect(() => {
    sendToChatRef.current = sendMessage;
    return () => { sendToChatRef.current = null; };
  }, [sendMessage, sendToChatRef]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="chat-tab">
      {/* Connection indicator */}
      <div className={`chat-connection ${connected ? 'online' : 'offline'}`}>
        <div className={`conn-dot ${connected ? 'online' : 'offline'}`} />
        <span>{connected ? 'Connected' : 'Reconnecting...'}</span>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-icon">
              <BreathingDot />
            </div>
            <p className="chat-empty-title">Talk to AURA</p>
            <p className="chat-empty-sub">Ask anything, explore ideas, or just chat.</p>
            <div className="chat-quick-actions">
              {['What can you do?', 'Search for AI news', 'Tell me about yourself'].map((text) => (
                <button key={text} className="quick-action" onClick={() => sendMessage(text)}>
                  {text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-msg ${msg.role}`}>
            {msg.role === 'assistant' && (
              <div className="chat-msg-avatar">
                <div className="avatar-dot" />
              </div>
            )}
            <div className={`chat-msg-bubble ${msg.role}`}>
              {msg.role === 'assistant' ? (
                <div className="chat-msg-text markdown-body">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {msg.content}
                  </ReactMarkdown>
                  {msg.isStreaming && <span className="streaming-cursor" />}
                </div>
              ) : (
                <div className="chat-msg-text">{msg.content}</div>
              )}
            </div>
          </div>
        ))}

        {isLoading && !currentMsgIdRef.current && (
          <div className="chat-msg assistant">
            <div className="chat-msg-avatar">
              <div className="avatar-dot" />
            </div>
            <div className="chat-msg-bubble assistant">
              <div className="thinking-dots">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder="Message AURA..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className={`chat-send ${input.trim() ? 'active' : ''}`}
          onClick={() => sendMessage(input)}
          disabled={!input.trim() || isLoading}
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════
// DASHBOARD TAB
// ═══════════════════════════════════════════
function DashboardTab() {
  const [auraStatus, setAuraStatus] = useState<AuraStatus | null>(null);
  const [statusInfo, setStatusInfo] = useState<StatusInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      try {
        const [auraRes, statusRes] = await Promise.all([
          fetch(`${API_BASE}/aura`),
          fetch(`${API_BASE}/status`),
        ]);
        if (auraRes.ok) setAuraStatus(await auraRes.json());
        if (statusRes.ok) setStatusInfo(await statusRes.json());
      } catch {
        // silent
      }
      setLoading(false);
    };
    fetchAll();
    const interval = setInterval(fetchAll, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !auraStatus) {
    return (
      <div className="dash-loading">
        <div className="shimmer-block w80" />
        <div className="shimmer-block w60" />
        <div className="shimmer-block w40" />
      </div>
    );
  }

  const emoji = MOOD_EMOJIS[auraStatus?.mood || ''] || '\u{1F916}';
  const energyPct = Math.round((auraStatus?.energy || 0) * 100);
  const warmthPct = Math.round((auraStatus?.warmth || 0) * 100);
  const engagementPct = Math.round((auraStatus?.engagement || 0) * 100);

  return (
    <div className="dash-tab">
      {/* Mood card */}
      <div className="dash-card mood-card">
        <div className="mood-header">
          <span className="mood-emoji">{emoji}</span>
          <div className="mood-info">
            <span className="mood-label">{auraStatus?.mood || 'Unknown'}</span>
            <span className="mood-soul">{auraStatus?.soul_name || 'AURA'}</span>
          </div>
          <div className={`status-badge ${statusInfo?.online ? 'online' : 'offline'}`}>
            {statusInfo?.online ? 'Online' : 'Offline'}
          </div>
        </div>
      </div>

      {/* Stats grid */}
      <div className="dash-grid">
        <div className="dash-stat">
          <div className="stat-bar">
            <div className="stat-fill green" style={{ width: `${energyPct}%` }} />
          </div>
          <div className="stat-row">
            <span className="stat-label">Energy</span>
            <span className="stat-value">{energyPct}%</span>
          </div>
        </div>
        <div className="dash-stat">
          <div className="stat-bar">
            <div className="stat-fill orange" style={{ width: `${warmthPct}%` }} />
          </div>
          <div className="stat-row">
            <span className="stat-label">Warmth</span>
            <span className="stat-value">{warmthPct}%</span>
          </div>
        </div>
        <div className="dash-stat">
          <div className="stat-bar">
            <div className="stat-fill blue" style={{ width: `${engagementPct}%` }} />
          </div>
          <div className="stat-row">
            <span className="stat-label">Engagement</span>
            <span className="stat-value">{engagementPct}%</span>
          </div>
        </div>
      </div>

      {/* Info cards */}
      <div className="dash-info-grid">
        <div className="info-card">
          <span className="info-icon">{'\u{1F9E0}'}</span>
          <span className="info-val">{statusInfo?.memory_count ?? '—'}</span>
          <span className="info-label">Memories</span>
        </div>
        <div className="info-card">
          <span className="info-icon">{'\u{1F4AC}'}</span>
          <span className="info-val">{statusInfo?.query_count ?? '—'}</span>
          <span className="info-label">Queries</span>
        </div>
        <div className="info-card">
          <span className="info-icon">{'\u{1F504}'}</span>
          <span className="info-val">{auraStatus?.patterns_learned ?? '—'}</span>
          <span className="info-label">Patterns</span>
        </div>
        <div className="info-card">
          <span className="info-icon">{'\u{1F4C8}'}</span>
          <span className="info-val">{auraStatus?.turns ?? '—'}</span>
          <span className="info-label">Turns</span>
        </div>
      </div>

      {/* Model info */}
      <div className="dash-card model-card">
        <span className="model-label">Active Model</span>
        <span className="model-name">{statusInfo?.model || statusInfo?.last_model_used || 'auto'}</span>
      </div>
    </div>
  );
}


// ═══════════════════════════════════════════
// TOOLS TAB
// ═══════════════════════════════════════════
function ToolsTab({ tg, onRunCommand }: { tg?: NonNullable<Window['Telegram']>['WebApp']; onRunCommand: (cmd: string) => void }) {
  const quickTools = [
    { id: 'research', label: 'Research', icon: '\u{1F50D}', color: '#8b5cf6', command: '/research ' },
    { id: 'code', label: 'Code', icon: '\u{1F4BB}', color: '#3b82f6', command: '/code ' },
    { id: 'image', label: 'Image', icon: '\u{1F3A8}', color: '#f59e0b', command: '/imagine ' },
    { id: 'search', label: 'Search', icon: '\u{1F310}', color: '#10b981', command: '/search ' },
    { id: 'compare', label: 'Compare', icon: '\u{2696}\u{FE0F}', color: '#ec4899', command: '/compare ' },
    { id: 'summarize', label: 'Summarize', icon: '\u{1F4DD}', color: '#14b8a6', command: '/summarize ' },
    { id: 'math', label: 'Math', icon: '\u{1F522}', color: '#f97316', command: '/math ' },
    { id: 'youtube', label: 'YouTube', icon: '\u{1F3AC}', color: '#ef4444', command: '/youtube ' },
  ];

  const [toolsInfo, setToolsInfo] = useState<{ name: string; category: string; description: string }[]>([]);

  useEffect(() => {
    const fetchTools = async () => {
      try {
        const res = await fetch(`${API_BASE}/tools`);
        if (res.ok) {
          const data = await res.json();
          setToolsInfo(data.tools || []);
        }
      } catch {
        // silent
      }
    };
    fetchTools();
  }, []);

  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [toolInput, setToolInput] = useState('');

  const handleToolTap = (tool: typeof quickTools[0]) => {
    tg?.HapticFeedback?.impactOccurred('medium');
    setSelectedTool(tool.id);
    setToolInput('');
  };

  const handleToolRun = () => {
    if (!selectedTool || !toolInput.trim()) return;
    const tool = quickTools.find(t => t.id === selectedTool);
    if (tool) {
      onRunCommand(`${tool.command}${toolInput.trim()}`);
      setSelectedTool(null);
      setToolInput('');
    }
  };

  return (
    <div className="tools-tab">
      {/* Tool input panel (shown when a tool is selected) */}
      {selectedTool && (() => {
        const tool = quickTools.find(t => t.id === selectedTool);
        if (!tool) return null;
        return (
          <div className="tool-input-panel">
            <div className="tool-input-header">
              <span className="tool-input-icon">{tool.icon}</span>
              <span className="tool-input-title">{tool.label}</span>
              <button className="tool-input-close" onClick={() => setSelectedTool(null)}>&times;</button>
            </div>
            <textarea
              className="tool-input-field"
              placeholder={`Enter ${tool.label.toLowerCase()} query...`}
              value={toolInput}
              onChange={(e) => setToolInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleToolRun(); } }}
              rows={2}
              autoFocus
            />
            <button
              className={`tool-run-btn ${toolInput.trim() ? 'active' : ''}`}
              onClick={handleToolRun}
              disabled={!toolInput.trim()}
            >
              Run {tool.label}
            </button>
          </div>
        );
      })()}

      {/* Quick access grid */}
      <div className="tools-section-label">Quick Actions</div>
      <div className="tools-grid">
        {quickTools.map((tool) => (
          <button
            key={tool.id}
            className={`tool-card ${selectedTool === tool.id ? 'selected' : ''}`}
            onClick={() => handleToolTap(tool)}
            style={{ '--tool-color': tool.color } as React.CSSProperties}
          >
            <span className="tool-icon">{tool.icon}</span>
            <span className="tool-label">{tool.label}</span>
          </button>
        ))}
      </div>

      {/* Available tools count */}
      {toolsInfo.length > 0 && (
        <>
          <div className="tools-section-label">
            All Tools ({toolsInfo.length})
          </div>
          <div className="tools-list">
            {toolsInfo.slice(0, 20).map((tool) => (
              <div key={tool.name} className="tool-list-item">
                <span className="tool-list-name">{tool.name}</span>
                <span className="tool-list-cat">{tool.category}</span>
              </div>
            ))}
            {toolsInfo.length > 20 && (
              <div className="tools-more">+{toolsInfo.length - 20} more tools</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}


// ═══════════════════════════════════════════
// EMOTION TAB
// ═══════════════════════════════════════════
function EmotionTab() {
  const [alma, setAlma] = useState<ALMAState | null>(null);
  const [section, setSection] = useState<'emotions' | 'neuro' | 'personality'>('emotions');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAlma = async () => {
      try {
        const res = await fetch(`${API_BASE}/alma/state`);
        if (res.ok) {
          setAlma(await res.json());
        }
      } catch {
        // silent
      }
      setLoading(false);
    };
    fetchAlma();
    const interval = setInterval(fetchAlma, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !alma) {
    return (
      <div className="dash-loading">
        <div className="shimmer-block w80" />
        <div className="shimmer-block w60" />
      </div>
    );
  }

  if (!alma) {
    return (
      <div className="emotion-empty">
        <p>Emotion system unavailable</p>
      </div>
    );
  }

  const emotionColor = EMOTION_COLORS[alma.dominant_emotion] || EMOTION_COLORS.neutral;

  return (
    <div className="emotion-tab">
      {/* Dominant emotion header */}
      <div className="emotion-header">
        <div className="emotion-dot" style={{ backgroundColor: emotionColor, boxShadow: `0 0 12px ${emotionColor}` }} />
        <div className="emotion-header-text">
          <span className="emotion-name">{alma.dominant_emotion}</span>
          <span className="emotion-intensity">{(alma.intensity * 100).toFixed(0)}% intensity</span>
        </div>
      </div>

      {/* Section tabs */}
      <div className="emotion-tabs">
        {(['emotions', 'neuro', 'personality'] as const).map((s) => (
          <button
            key={s}
            className={`emotion-tab-btn ${section === s ? 'active' : ''}`}
            onClick={() => setSection(s)}
          >
            {s === 'emotions' ? 'Emotions' : s === 'neuro' ? 'Neuro' : 'OCEAN'}
          </button>
        ))}
      </div>

      {/* Emotions section */}
      {section === 'emotions' && (
        <div className="emotion-section">
          {/* PAD Space */}
          <div className="section-label">PAD Space</div>
          <PADMiniBar label="Pleasure" value={alma.pad.pleasure} color="#10b981" />
          <PADMiniBar label="Arousal" value={alma.pad.arousal} color="#f59e0b" />
          <PADMiniBar label="Dominance" value={alma.pad.dominance} color="#3b82f6" />

          {/* Active emotions */}
          {alma.active_emotions.length > 0 && (
            <>
              <div className="section-label" style={{ marginTop: '12px' }}>Active Emotions</div>
              {alma.active_emotions.slice(0, 5).map((e, i) => (
                <div key={i} className="active-emotion-row">
                  <div className="ae-dot" style={{ backgroundColor: EMOTION_COLORS[e.name] || '#6b7280' }} />
                  <span className="ae-name">{e.name}</span>
                  <div className="ae-bar-track">
                    <div
                      className="ae-bar-fill"
                      style={{ width: `${e.current_intensity * 100}%`, backgroundColor: EMOTION_COLORS[e.name] || '#6b7280' }}
                    />
                  </div>
                  <span className="ae-pct">{(e.current_intensity * 100).toFixed(0)}%</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Neuromodulators section */}
      {section === 'neuro' && (
        <div className="emotion-section">
          <div className="section-label">Neuromodulator Levels</div>
          {Object.entries(alma.neuromodulators).map(([key, value]) => {
            const info = NEURO_INFO[key];
            return (
              <div key={key} className="neuro-row">
                <div className="neuro-header">
                  <span className="neuro-name">{info.label}</span>
                  <span className="neuro-pct">{(value * 100).toFixed(0)}%</span>
                </div>
                <div className="neuro-bar-track">
                  <div
                    className="neuro-bar-fill"
                    style={{ width: `${value * 100}%`, backgroundColor: info.color, boxShadow: `0 0 6px ${info.color}40` }}
                  />
                </div>
                <span className="neuro-effect">{info.effect}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Personality section */}
      {section === 'personality' && (
        <div className="emotion-section">
          <div className="section-label">OCEAN Personality</div>
          {Object.entries(alma.personality).map(([key, value]) => {
            const info = PERSONALITY_INFO[key];
            return (
              <div key={key} className="personality-row">
                <div className="personality-header">
                  <span className="personality-name">{info.label}</span>
                  <span className="personality-pct">{(value * 100).toFixed(0)}%</span>
                </div>
                <div className="personality-bar-track">
                  <div className="personality-bar-fill" style={{ width: `${value * 100}%` }} />
                </div>
                <div className="personality-labels">
                  <span>{info.low}</span>
                  <span>{info.high}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}


// ─── PAD Mini Bar ───
function PADMiniBar({ label, value, color }: { label: string; value: number; color: string }) {
  const normalized = ((value + 1) / 2) * 100;
  const isPositive = value >= 0;
  return (
    <div className="pad-row">
      <span className="pad-label">{label}</span>
      <div className="pad-track">
        <div className="pad-center" />
        <div
          className="pad-fill"
          style={{
            left: isPositive ? '50%' : `${normalized}%`,
            width: `${Math.abs(value) * 50}%`,
            backgroundColor: color,
          }}
        />
      </div>
      <span className="pad-value">
        {value >= 0 ? '+' : ''}{(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}


// ═══════════════════════════════════════════
// SETTINGS TAB
// ═══════════════════════════════════════════
function SettingsTab({ tg }: { tg?: NonNullable<Window['Telegram']>['WebApp'] }) {
  const [language, setLanguage] = useState('en');
  const [model, setModel] = useState('auto');
  const [models, setModels] = useState<string[]>([]);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // Fetch available models
    fetch(`${API_BASE}/models`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d?.models) setModels(d.models.map((m: any) => m.name || m));
      })
      .catch(() => {});
  }, []);

  const save = () => {
    tg?.HapticFeedback?.notificationOccurred('success');
    // Send settings back to bot
    tg?.sendData(JSON.stringify({
      action: 'settings',
      settings: { language, model },
    }));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const langs = [
    { code: 'en', label: '\u{1F1EC}\u{1F1E7} English' },
    { code: 'ru', label: '\u{1F1F7}\u{1F1FA} \u0420\u0443\u0441\u0441\u043a\u0438\u0439' },
    { code: 'az', label: '\u{1F1E6}\u{1F1FF} Az\u0259rbaycan' },
  ];

  return (
    <div className="settings-tab">
      <div className="settings-section">
        <div className="section-label">Language</div>
        <div className="settings-option-group">
          {langs.map(l => (
            <button
              key={l.code}
              className={`settings-option ${language === l.code ? 'active' : ''}`}
              onClick={() => { setLanguage(l.code); tg?.HapticFeedback?.selectionChanged(); }}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>

      <div className="settings-section">
        <div className="section-label">AI Model</div>
        <select
          className="settings-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="auto">Auto (recommended)</option>
          {models.slice(0, 20).map(m => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div className="settings-section">
        <div className="section-label">About</div>
        <div className="settings-about">
          <div className="about-row"><span className="about-key">Version</span><span className="about-val">4.3.0</span></div>
          <div className="about-row"><span className="about-key">Platform</span><span className="about-val">{tg?.platform || 'unknown'}</span></div>
          <div className="about-row"><span className="about-key">Theme</span><span className="about-val">{tg?.colorScheme || 'dark'}</span></div>
        </div>
      </div>

      <button className={`settings-save ${saved ? 'saved' : ''}`} onClick={save}>
        {saved ? '\u2705 Saved!' : 'Save Settings'}
      </button>
    </div>
  );
}


// ═══════════════════════════════════════════
// SVG ICONS (inline to avoid dependencies)
// ═══════════════════════════════════════════
function ChatIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? '#a78bfa' : '#a1a1aa'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
    </svg>
  );
}

function DashIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? '#a78bfa' : '#a1a1aa'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </svg>
  );
}

function ToolsIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? '#a78bfa' : '#a1a1aa'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z" />
    </svg>
  );
}

function EmotionIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? '#a78bfa' : '#a1a1aa'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
      <line x1="9" y1="9" x2="9.01" y2="9" />
      <line x1="15" y1="9" x2="15.01" y2="9" />
    </svg>
  );
}

function SettingsIcon({ active }: { active: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={active ? '#a78bfa' : '#a1a1aa'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}
