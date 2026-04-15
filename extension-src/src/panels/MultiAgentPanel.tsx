/**
 * MultiAgentPanel — chat with the Aura specialist council.
 *
 * Routes each message through /api/multi-agent/chat. A side list shows
 * available specialists (loaded on mount from /api/multi-agent/agents).
 * A "Route preview" button hits /api/multi-agent/route without executing —
 * shows which specialists would fire and why before committing.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Users, Send, Eye, Loader2, Trash2 } from 'lucide-react';
import { multiAgent } from '../api/client';
import type { SpecialistAgent, MultiAgentChatResponse, RoutePreviewResponse } from '../api/types';

interface Turn {
  role: 'user' | 'council';
  text: string;
  agents?: string[];
  mode?: string;
  confidence?: number;
}

export default function MultiAgentPanel() {
  const [agents, setAgents] = useState<SpecialistAgent[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [routePreview, setRoutePreview] = useState<RoutePreviewResponse | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    multiAgent.agents().then((r) => setAgents(r.agents ?? [])).catch(() => {});
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  // Specialist context-menu hook — App.tsx dispatches aura-specialist-prefill
  // when the user right-clicks selected text and picks an "Ask specialist…"
  // entry. We pre-fill the input with the selected text + a @specialist
  // mention so the backend routes to that specialist. The real routing happens
  // server-side via /api/multi-agent/chat — prepending the name to the message
  // is the simplest signal until the backend supports an explicit override.
  useEffect(() => {
    const onPrefill = (ev: Event) => {
      const detail = (ev as CustomEvent).detail as { specialist: string; text: string };
      if (!detail?.text) return;
      setInput(`@${detail.specialist} ${detail.text}`);
    };
    window.addEventListener('aura-specialist-prefill', onPrefill);
    return () => window.removeEventListener('aura-specialist-prefill', onPrefill);
  }, []);

  const send = useCallback(async () => {
    const msg = input.trim();
    if (!msg || sending) return;
    setInput('');
    setRoutePreview(null);
    setErr(null);
    setTurns((t) => [...t, { role: 'user', text: msg }]);
    setSending(true);
    try {
      const resp: MultiAgentChatResponse = await multiAgent.chat(msg);
      setTurns((t) => [
        ...t,
        {
          role: 'council',
          text: resp.response,
          agents: resp.agents_used,
          mode: resp.routing_mode,
          confidence: resp.confidence,
        },
      ]);
    } catch (e: any) {
      setErr(e?.message || 'Council request failed');
      setTurns((t) => t.slice(0, -1));
    }
    setSending(false);
  }, [input, sending]);

  const previewRoute = useCallback(async () => {
    const q = input.trim();
    if (!q || previewing) return;
    setPreviewing(true);
    try {
      const rp = await multiAgent.route(q);
      setRoutePreview(rp);
    } catch (e: any) {
      setErr(e?.message || 'Route preview failed');
    }
    setPreviewing(false);
  }, [input, previewing]);

  const clear = useCallback(async () => {
    try {
      await multiAgent.clear();
      setTurns([]);
      setRoutePreview(null);
    } catch { /* silent */ }
  }, []);

  return (
    <div className="panel-scroll-root" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--b1)', display: 'flex', alignItems: 'center', gap: 8 }}>
        <Users size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>
          Specialist Council
          <span style={{ color: 'var(--mu)', fontWeight: 400, marginLeft: 6 }}>
            {agents.length} agents
          </span>
        </span>
        <button
          onClick={clear}
          aria-label="Clear"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <Trash2 size={12} />
        </button>
      </div>

      {agents.length > 0 && (
        <div style={{ padding: '8px 14px', borderBottom: '1px solid var(--b1)', display: 'flex', gap: 6, overflowX: 'auto' }}>
          {agents.map((a) => (
            <div
              key={a.name}
              title={a.description}
              style={{
                padding: '3px 8px',
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 12,
                fontSize: 10,
                color: 'var(--tx)',
                whiteSpace: 'nowrap',
                flexShrink: 0,
              }}
            >
              {a.name}
            </div>
          ))}
        </div>
      )}

      <div ref={scrollRef} style={{ flex: 1, padding: 14, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {turns.length === 0 && !routePreview && (
          <div style={{ color: 'var(--mu)', fontSize: 11, textAlign: 'center', padding: '20px 0' }}>
            Ask a question. The council routes it to the right specialists.
          </div>
        )}

        {turns.map((t, i) => (
          <div
            key={i}
            style={{
              alignSelf: t.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '90%',
              padding: '8px 12px',
              borderRadius: 10,
              background: t.role === 'user' ? 'var(--p)' : 'var(--s2)',
              color: t.role === 'user' ? '#fff' : 'var(--tx)',
              fontSize: 12,
              lineHeight: 1.5,
              whiteSpace: 'pre-wrap',
              border: t.role === 'council' ? '1px solid var(--b1)' : 'none',
            }}
          >
            {t.text}
            {t.role === 'council' && t.agents && t.agents.length > 0 && (
              <div style={{ display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap' }}>
                {t.agents.map((name) => (
                  <span
                    key={name}
                    style={{ padding: '1px 6px', background: 'var(--b1)', borderRadius: 8, fontSize: 9, color: 'var(--mu)' }}
                  >
                    {name}
                  </span>
                ))}
                {t.mode && (
                  <span style={{ fontSize: 9, color: 'var(--mu)', marginLeft: 'auto' }}>
                    {t.mode}
                    {typeof t.confidence === 'number' ? ` · ${(t.confidence * 100).toFixed(0)}%` : ''}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}

        {routePreview && (
          <div style={{ background: 'var(--s2)', border: '1px dashed var(--b1)', borderRadius: 10, padding: 10 }}>
            <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 4 }}>
              Route preview · {routePreview.mode} · {(routePreview.confidence * 100).toFixed(0)}%
            </div>
            <div style={{ fontSize: 11, color: 'var(--tx)', marginBottom: 6 }}>{routePreview.reasoning}</div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {routePreview.selected_agents.map((name) => (
                <span key={name} style={{ padding: '2px 8px', background: 'var(--p)', color: '#fff', borderRadius: 10, fontSize: 10 }}>
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}

        {err && (
          <div style={{ color: '#f87171', fontSize: 11, padding: 8, background: 'rgba(248, 113, 113, 0.1)', borderRadius: 6 }}>
            {err}
          </div>
        )}
      </div>

      <div style={{ padding: 12, borderTop: '1px solid var(--b1)', display: 'flex', gap: 6 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder="Ask the council…"
          rows={2}
          style={{
            flex: 1,
            padding: '8px 10px',
            background: 'var(--s2)',
            border: '1px solid var(--b1)',
            borderRadius: 8,
            color: 'var(--tx)',
            fontSize: 12,
            fontFamily: 'inherit',
            resize: 'none',
          }}
        />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <button
            onClick={previewRoute}
            disabled={!input.trim() || previewing}
            title="Preview routing"
            style={{
              padding: '6px 10px',
              background: 'var(--s2)',
              border: '1px solid var(--b1)',
              borderRadius: 6,
              color: 'var(--mu)',
              cursor: previewing ? 'not-allowed' : 'pointer',
            }}
          >
            {previewing ? <Loader2 size={12} className="spin" /> : <Eye size={12} />}
          </button>
          <button
            onClick={send}
            disabled={!input.trim() || sending}
            style={{
              padding: '6px 10px',
              background: 'var(--p)',
              border: 'none',
              borderRadius: 6,
              color: '#fff',
              cursor: sending ? 'not-allowed' : 'pointer',
            }}
          >
            {sending ? <Loader2 size={12} className="spin" /> : <Send size={12} />}
          </button>
        </div>
      </div>
    </div>
  );
}
