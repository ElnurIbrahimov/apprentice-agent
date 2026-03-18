import React, { useRef, useEffect, useCallback, useState } from 'react';
import { useStore } from '../store';
import MessageBubble from '../components/MessageBubble';
import HomeScreen from '../components/HomeScreen';
import ContextBar from '../components/ContextBar';
import InputBar from '../components/InputBar';
import { PAGE_KEYWORDS } from '../ws';
import { getPageContentCached, getCurrentTab } from '../ext';
import type { StreamState } from '../types';
import { speak } from '../tts';
import { ChevronDown, Brain, Pen } from 'lucide-react';

export default function ChatPanel() {
  const { messages, addMessage, activeStream, setActiveStream, setPendingCtx } = useStore();
  const bottomRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const streamingMsgId = useRef<string | null>(null);
  const [userScrolledUp, setUserScrolledUp] = useState(false);
  const [newMsgCount, setNewMsgCount] = useState(0);
  const [scrollProgress, setScrollProgress] = useState(0);
  const isAutoScrolling = useRef(false);

  // Check if user is near the bottom
  const isNearBottom = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return true;
    return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  // Scroll to bottom smoothly
  const scrollToBottom = useCallback((force = false) => {
    if (!force && userScrolledUp) return;
    isAutoScrolling.current = true;
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { isAutoScrolling.current = false; }, 500);
  }, [userScrolledUp]);

  // Auto-scroll on new messages / stream updates
  useEffect(() => {
    if (!userScrolledUp) {
      scrollToBottom();
    } else {
      // Count new messages while user is scrolled up
      setNewMsgCount(prev => prev + 1);
    }
  }, [messages, activeStream, userScrolledUp, scrollToBottom]);

  // Scroll event handler — track user scroll position + progress
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const handleScroll = () => {
      if (isAutoScrolling.current) return;
      const nearBottom = isNearBottom();
      setUserScrolledUp(!nearBottom);
      if (nearBottom) setNewMsgCount(0);

      // Update scroll progress
      const scrollable = el.scrollHeight - el.clientHeight;
      setScrollProgress(scrollable > 0 ? el.scrollTop / scrollable : 0);
    };
    el.addEventListener('scroll', handleScroll, { passive: true });
    return () => el.removeEventListener('scroll', handleScroll);
  }, [isNearBottom]);

  const handleJumpToBottom = useCallback(() => {
    setUserScrolledUp(false);
    setNewMsgCount(0);
    isAutoScrolling.current = true;
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setTimeout(() => { isAutoScrolling.current = false; }, 500);
  }, []);

  const sysmsg = useCallback((text: string) => {
    useStore.getState().addMessage({ id: crypto.randomUUID(), role: 'ai', text, timestamp: Date.now() });
  }, []);

  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim()) return;
    const st = useStore.getState();
    if (!st.wsReady) { sysmsg('AURA is offline — start the backend server.'); return; }
    if (st.activeStream) return;

    // Lock immediately
    setActiveStream(true);

    let full = text;

    // Auto page context
    if (!st.pendingCtx) {
      if (PAGE_KEYWORDS.test(text)) {
        const pageResp = await getPageContentCached();
        if (pageResp?.ok && pageResp.text) {
          setPendingCtx({ text: pageResp.text.slice(0, 20000), title: pageResp.title, url: pageResp.url, action: 'ask' });
        }
      } else {
        const tabResp = await getCurrentTab();
        if (tabResp?.ok && tabResp.url &&
            !tabResp.url.startsWith('chrome://') && !tabResp.url.startsWith('about:')) {
          full = `[Current page: ${tabResp.title || tabResp.url} — ${tabResp.url}]\n\n${text}`;
        }
      }
    }

    // Re-check WS after awaits
    const { wsReady: ready, ws: socket } = useStore.getState();
    if (!ready || socket?.readyState !== WebSocket.OPEN) {
      setActiveStream(null);
      sysmsg('AURA disconnected — reconnecting…');
      return;
    }

    const { pendingCtx: ctx, thinkingMode, thinkingLevel, deepResearch, conversationId, getModel, customInstructions, userName } = useStore.getState();
    if (ctx) {
      full = `[Context: ${ctx.title || ctx.url || 'selection'}]\n${ctx.text}\n\n---\n${text}`;
      setPendingCtx(null);
    }

    if (thinkingMode) full = '[Think step by step before answering]\n' + full;
    if (deepResearch) full = '[Do deep research on this, search the web if needed]\n' + full;

    // Reset scroll state for new conversation turn
    setUserScrolledUp(false);
    setNewMsgCount(0);

    // Add user + AI placeholder messages
    addMessage({ id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() });
    const aiId = crypto.randomUUID();
    streamingMsgId.current = aiId;
    addMessage({ id: aiId, role: 'ai', text: '', timestamp: Date.now() });

    const stream: StreamState = {
      type: 'chat',
      rawText: '',
      thinkingText: '',
      isThinkingPhase: thinkingMode,
      thinkingStartTime: thinkingMode ? Date.now() : undefined,
      onDone: (rawText, thinkingContent) => {
        useStore.setState(s => ({
          messages: s.messages.map(m =>
            m.id === aiId
              ? { ...m, text: rawText, thinkingContent: thinkingContent || undefined }
              : m
          ),
        }));
        streamingMsgId.current = null;
        if (useStore.getState().autoSpeak) {
          speak(rawText);
        }
      },
    };

    setActiveStream(stream);
    const payload: any = {
      type: 'chat',
      message: full,
      conversation_id: conversationId,
      model: getModel('chat'),
    };
    if (customInstructions) payload.custom_instructions = customInstructions;
    if (userName) payload.user_name = userName;
    if (thinkingMode) {
      payload.thinking = true;
      payload.thinking_level = thinkingLevel;
    }
    socket!.send(JSON.stringify(payload));
  }, [sysmsg, addMessage, setActiveStream, setPendingCtx]);

  // Handle aura-send events dispatched by AskPanel / ToolsPanel
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;
  useEffect(() => {
    const handler = (e: Event) => {
      const text = (e as CustomEvent<{ text: string }>).detail?.text;
      if (text) sendMessageRef.current(text);
    };
    window.addEventListener('aura-send', handler);
    return () => window.removeEventListener('aura-send', handler);
  }, []);

  // Keep streaming message updated in real time
  const streamText = (activeStream && activeStream !== true) ? activeStream.rawText : null;
  const streamThinkingText = (activeStream && activeStream !== true) ? activeStream.thinkingText : null;
  const isThinkingPhase = (activeStream && activeStream !== true) ? activeStream.isThinkingPhase : false;
  const thinkingStartTime = (activeStream && activeStream !== true) ? activeStream.thinkingStartTime : undefined;

  // Elapsed time for thinking phase
  const [thinkElapsed, setThinkElapsed] = useState(0);
  useEffect(() => {
    if (!isThinkingPhase || !thinkingStartTime) {
      setThinkElapsed(0);
      return;
    }
    const interval = setInterval(() => {
      setThinkElapsed(Math.floor((Date.now() - thinkingStartTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [isThinkingPhase, thinkingStartTime]);

  const isEmpty = messages.length === 0 && !activeStream;

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ position: 'relative' }}>
      <ContextBar />

      {/* Messages */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-3 py-3"
        style={{ scrollBehavior: 'smooth', position: 'relative' }}
      >
        {isEmpty ? (
          <HomeScreen onChip={sendMessage} />
        ) : (
          <>
            {messages.map((msg) => {
              // For streaming AI message, show live text
              if (msg.id === streamingMsgId.current && streamText !== null) {
                const liveMsg = {
                  ...msg,
                  text: streamText,
                  thinkingContent: streamThinkingText || undefined,
                };
                if (!streamText && !streamThinkingText) {
                  return (
                    <div key={msg.id} className="flex gap-2 mb-3">
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                        style={{ background: 'radial-gradient(circle at 30% 30%, var(--pl), var(--p))', color: 'white' }}
                      >
                        A
                      </div>
                      {/* Phase indicator for thinking mode */}
                      {isThinkingPhase ? (
                        <div className="think-phase-indicator mt-1.5">
                          <Brain size={12} className="think-phase-icon" />
                          <span>Thinking...{thinkElapsed > 0 && ` ${thinkElapsed}s`}</span>
                        </div>
                      ) : (
                        <div className="aura-thinking mt-1.5">
                          <span /><span /><span />
                        </div>
                      )}
                    </div>
                  );
                }
                // Show phase indicator above the message during streaming
                return (
                  <React.Fragment key={msg.id}>
                    {isThinkingPhase && streamThinkingText && !streamText && (
                      <div className="flex gap-2 mb-1 items-center" style={{ paddingLeft: 38 }}>
                        <div className="think-phase-indicator">
                          <Brain size={12} className="think-phase-icon" />
                          <span>Thinking...{thinkElapsed > 0 && ` ${thinkElapsed}s`}</span>
                        </div>
                      </div>
                    )}
                    {!isThinkingPhase && streamThinkingText && !!activeStream && (
                      <div className="flex gap-2 mb-1 items-center" style={{ paddingLeft: 38 }}>
                        <div className="respond-phase-indicator">
                          <Pen size={11} />
                          <span>Responding...</span>
                        </div>
                      </div>
                    )}
                    <MessageBubble key={msg.id} message={liveMsg} isLatest isStreaming />
                  </React.Fragment>
                );
              }
              return <MessageBubble key={msg.id} message={msg} />;
            })}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Scroll progress indicator — thin bar on right edge */}
      {!isEmpty && scrollProgress > 0 && scrollProgress < 0.98 && (
        <div
          className="scroll-progress-track"
          style={{
            position: 'absolute',
            top: 0,
            right: 0,
            width: 3,
            height: '100%',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <div
            className="scroll-progress-thumb"
            style={{
              position: 'absolute',
              top: 0,
              right: 0,
              width: '100%',
              height: `${scrollProgress * 100}%`,
              background: 'linear-gradient(to bottom, transparent, var(--pl))',
              opacity: 0.5,
              borderRadius: '0 0 2px 2px',
              transition: 'height 0.1s ease-out, opacity 0.3s ease',
            }}
          />
        </div>
      )}

      {/* New message indicator — shown when user has scrolled up */}
      {userScrolledUp && newMsgCount > 0 && (
        <button
          onClick={handleJumpToBottom}
          className="new-msg-indicator"
          style={{
            position: 'absolute',
            bottom: 72,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 20,
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '6px 14px',
            borderRadius: 20,
            background: 'var(--glass)',
            backdropFilter: 'blur(12px)',
            WebkitBackdropFilter: 'blur(12px)',
            border: '1px solid rgba(124, 58, 237, 0.3)',
            boxShadow: '0 4px 20px rgba(0,0,0,0.4), 0 0 12px rgba(124, 58, 237, 0.15)',
            color: 'var(--pl)',
            fontSize: 11,
            fontWeight: 500,
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          <ChevronDown size={14} />
          <span>{newMsgCount} new message{newMsgCount > 1 ? 's' : ''}</span>
        </button>
      )}

      {/* Input */}
      <InputBar onSend={sendMessage} featureKey="chat" />
    </div>
  );
}
