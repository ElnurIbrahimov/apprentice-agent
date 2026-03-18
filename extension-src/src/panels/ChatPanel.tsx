import React, { useRef, useEffect, useCallback } from 'react';
import { useStore } from '../store';
import MessageBubble from '../components/MessageBubble';
import HomeScreen from '../components/HomeScreen';
import ContextBar from '../components/ContextBar';
import InputBar from '../components/InputBar';
import { PAGE_KEYWORDS } from '../ws';
import { getPageContentCached, getCurrentTab } from '../ext';
import type { StreamState } from '../types';
import { speak } from '../tts';

export default function ChatPanel() {
  const { messages, addMessage, activeStream, setActiveStream, setPendingCtx } = useStore();
  const bottomRef = useRef<HTMLDivElement>(null);
  const streamingMsgId = useRef<string | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, activeStream]);

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

    const { pendingCtx: ctx, thinkingMode, deepResearch, conversationId, getModel } = useStore.getState();
    if (ctx) {
      full = `[Context: ${ctx.title || ctx.url || 'selection'}]\n${ctx.text}\n\n---\n${text}`;
      setPendingCtx(null);
    }

    if (thinkingMode) full = '[Think step by step before answering]\n' + full;
    if (deepResearch) full = '[Do deep research on this, search the web if needed]\n' + full;

    // Add user + AI placeholder messages
    addMessage({ id: crypto.randomUUID(), role: 'user', text, timestamp: Date.now() });
    const aiId = crypto.randomUUID();
    streamingMsgId.current = aiId;
    addMessage({ id: aiId, role: 'ai', text: '', timestamp: Date.now() });

    const stream: StreamState = {
      type: 'chat',
      rawText: '',
      onDone: (rawText) => {
        useStore.setState(s => ({
          messages: s.messages.map(m => m.id === aiId ? { ...m, text: rawText } : m),
        }));
        streamingMsgId.current = null;
        if (useStore.getState().autoSpeak) {
          speak(rawText);
        }
      },
    };

    setActiveStream(stream);
    socket!.send(JSON.stringify({
      type: 'chat',
      message: full,
      conversation_id: conversationId,
      model: getModel('chat'),
    }));
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

  const isEmpty = messages.length === 0 && !activeStream;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <ContextBar />

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3" style={{ scrollBehavior: 'smooth' }}>
        {isEmpty ? (
          <HomeScreen onChip={sendMessage} />
        ) : (
          <>
            {messages.map((msg) => {
              // For streaming AI message, show live text
              if (msg.id === streamingMsgId.current && streamText !== null) {
                const liveMsg = { ...msg, text: streamText };
                if (!streamText) {
                  return (
                    <div key={msg.id} className="flex gap-2 mb-3">
                      <div
                        className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                        style={{ background: 'radial-gradient(circle at 30% 30%, var(--pl), var(--p))', color: 'white' }}
                      >
                        A
                      </div>
                      <div className="dots mt-1.5">
                        <span /><span /><span />
                      </div>
                    </div>
                  );
                }
                return <MessageBubble key={msg.id} message={liveMsg} isLatest />;
              }
              return <MessageBubble key={msg.id} message={msg} />;
            })}
            <div ref={bottomRef} />
          </>
        )}
      </div>

      {/* Input */}
      <InputBar onSend={sendMessage} featureKey="chat" />
    </div>
  );
}
