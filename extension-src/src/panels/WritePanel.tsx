import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';
import { md } from '../markdown';

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

const TYPES = ['Essay', 'Article', 'Email', 'Story', 'Report', 'Letter', 'Poem', 'Blog Post', 'Newsletter', 'Social Post', 'Ad Copy', 'Product Description'];
const TONES = ['Formal', 'Casual', 'Persuasive', 'Informative', 'Creative'];
const LENS = ['Short', 'Medium', 'Long'];

type WriteMode = 'compose' | 'chat-draft' | 'research-write';

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

/* ------------------------------------------------------------------ */
/*  Chat Draft message type                                            */
/* ------------------------------------------------------------------ */

interface DraftMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

export default function WritePanel() {
  const { ws, wsReady, activeStream, setActiveStream, getModel } = useStore();

  // Top-level mode
  const [writeMode, setWriteMode] = useState<WriteMode>('compose');

  // Compose mode state
  const [tab, setTab] = useState<'write' | 'improve'>('write');
  const [type, setType] = useState('Essay');
  const [tone, setTone] = useState('Formal');
  const [len, setLen] = useState('Medium');
  const [result, setResult] = useState('');
  const [wordTarget, setWordTarget] = useState('');
  const [outlineFirst, setOutlineFirst] = useState(false);
  const [outline, setOutline] = useState('');
  const [outlineApproved, setOutlineApproved] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Chat Draft mode state
  const [draftMessages, setDraftMessages] = useState<DraftMessage[]>([]);
  const [draftInput, setDraftInput] = useState('');
  const [draftFinal, setDraftFinal] = useState('');
  const [draftAccepted, setDraftAccepted] = useState(false);
  const draftScrollRef = useRef<HTMLDivElement>(null);
  const draftInputRef = useRef<HTMLTextAreaElement>(null);

  // Research Write mode state
  const [rwTopic, setRwTopic] = useState('');
  const [rwPhase, setRwPhase] = useState<'idle' | 'researching' | 'writing' | 'done'>('idle');
  const [rwResearchResult, setRwResearchResult] = useState('');
  const [rwArticle, setRwArticle] = useState('');
  const [rwSources, setRwSources] = useState<Array<{ title?: string; url?: string; snippet?: string }>>([]);
  const [rwStatus, setRwStatus] = useState('');
  const rwAbortRef = useRef<AbortController | null>(null);
  const rwScrollRef = useRef<HTMLDivElement>(null);

  // Stream text reader for compose/chat-draft
  const stream = useStore(s => s.activeStream);
  const streamText = (stream && stream !== true && (stream.type === 'write')) ? stream.rawText : null;

  // Auto-scroll for draft chat
  useEffect(() => {
    if (draftScrollRef.current) {
      draftScrollRef.current.scrollTop = draftScrollRef.current.scrollHeight;
    }
  }, [draftMessages, streamText]);

  // Auto-scroll for research-write
  useEffect(() => {
    if (rwScrollRef.current) {
      rwScrollRef.current.scrollTop = rwScrollRef.current.scrollHeight;
    }
  }, [rwArticle, rwResearchResult, rwPhase]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (rwAbortRef.current) rwAbortRef.current.abort();
    };
  }, []);

  /* ================================================================ */
  /*  COMPOSE MODE                                                     */
  /* ================================================================ */

  const doCompose = useCallback(() => {
    const text = inputRef.current?.value.trim();
    if (!text) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { alert('AURA is offline.'); return; }
    if (activeStream) return;

    // If outline mode is on and we don't have an approved outline yet, generate outline first
    if (outlineFirst && !outlineApproved && tab === 'write') {
      setOutline('');
      setActiveStream({
        type: 'write',
        rawText: '',
        onFirstChunk: () => setOutline(''),
        onDone: (rawText) => setOutline(rawText),
      });

      const outlinePrompt = `Create a detailed outline for a ${tone.toLowerCase()}, ${len.toLowerCase()}-length ${type} about: ${text}${wordTarget ? `. Target approximately ${wordTarget} words.` : ''}. Output only the outline with numbered sections and bullet points.`;
      ws!.send(JSON.stringify({ type: 'chat', message: outlinePrompt, model: getModel('write'), conversation_id: null }));
      return;
    }

    setResult('');

    let prompt: string;
    if (tab === 'improve') {
      prompt = `Improve the following text. Make it ${tone.toLowerCase()} in tone and ${len.toLowerCase()} in length. Output only the improved text:\n\n${text}`;
    } else if (outlineFirst && outlineApproved && outline) {
      prompt = `Write a ${tone.toLowerCase()}, ${len.toLowerCase()}-length ${type} following this outline:\n\n${outline}\n\nTopic: ${text}${wordTarget ? `\n\nTarget word count: approximately ${wordTarget} words.` : ''}. Output only the final text.`;
    } else {
      prompt = `Write a ${tone.toLowerCase()}, ${len.toLowerCase()}-length ${type} about: ${text}${wordTarget ? `. Target approximately ${wordTarget} words.` : ''}`;
    }

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => setResult(''),
      onDone: (rawText) => setResult(rawText),
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('write'), conversation_id: null }));
  }, [ws, wsReady, activeStream, setActiveStream, getModel, tab, type, tone, len, wordTarget, outlineFirst, outlineApproved, outline]);

  const approveOutline = useCallback(() => {
    setOutlineApproved(true);
  }, []);

  const resetOutline = useCallback(() => {
    setOutline('');
    setOutlineApproved(false);
  }, []);

  // Display text for compose mode
  const composeStreamText = streamText;
  const composeDisplay = composeStreamText !== null ? composeStreamText : result;

  // Outline display
  const outlineStreamText = (stream && stream !== true && stream.type === 'write' && outlineFirst && !outlineApproved)
    ? stream.rawText
    : null;
  const outlineDisplay = outlineStreamText !== null ? outlineStreamText : outline;

  /* ================================================================ */
  /*  CHAT DRAFT MODE                                                  */
  /* ================================================================ */

  const sendDraftMessage = useCallback(() => {
    const text = draftInput.trim();
    if (!text) return;
    if (!wsReady || ws?.readyState !== WebSocket.OPEN) { alert('AURA is offline.'); return; }
    if (activeStream) return;

    const userMsg: DraftMessage = { id: 'u' + Date.now(), role: 'user', text };
    const aiMsg: DraftMessage = { id: 'a' + Date.now(), role: 'ai', text: '' };

    setDraftMessages(prev => [...prev, userMsg, aiMsg]);
    setDraftInput('');
    setDraftAccepted(false);

    // Build conversation context
    const history = [...draftMessages, userMsg]
      .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.text}`)
      .join('\n\n');

    const isFirst = draftMessages.length === 0;
    const systemContext = isFirst
      ? `You are a writing assistant. The user will describe what they want written. Generate a complete draft based on their description. Output only the draft text.`
      : `You are a writing assistant helping the user refine a document. Here is the conversation so far:\n\n${history}\n\nThe user is providing feedback. Revise the draft accordingly. Output only the revised draft text.`;

    const prompt = isFirst
      ? `${systemContext}\n\nUser request: ${text}`
      : `${systemContext}\n\nUser feedback: ${text}`;

    setActiveStream({
      type: 'write',
      rawText: '',
      onFirstChunk: () => {},
      onDone: (rawText) => {
        setDraftMessages(prev =>
          prev.map(m => m.id === aiMsg.id ? { ...m, text: rawText } : m)
        );
        setDraftFinal(rawText);
      },
    });

    ws!.send(JSON.stringify({ type: 'chat', message: prompt, model: getModel('write'), conversation_id: null }));
  }, [ws, wsReady, activeStream, setActiveStream, getModel, draftInput, draftMessages]);

  const acceptDraft = useCallback(() => {
    if (!draftFinal) return;
    navigator.clipboard.writeText(draftFinal).then(() => {
      setDraftAccepted(true);
    });
  }, [draftFinal]);

  const resetDraft = useCallback(() => {
    setDraftMessages([]);
    setDraftFinal('');
    setDraftAccepted(false);
    setDraftInput('');
  }, []);

  /* ================================================================ */
  /*  RESEARCH WRITE MODE                                              */
  /* ================================================================ */

  const doResearchWrite = useCallback(async () => {
    const topic = rwTopic.trim();
    if (!topic) return;

    setRwPhase('researching');
    setRwResearchResult('');
    setRwArticle('');
    setRwSources([]);
    setRwStatus('Researching topic...');

    const ctrl = new AbortController();
    rwAbortRef.current = ctrl;

    let researchReport = '';
    let collectedSources: Array<{ title?: string; url?: string; snippet?: string }> = [];

    try {
      // Phase 1: Research
      const resp = await fetch(`${HTTP}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ query: topic, depth: 'quick', model: getModel('research') }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setRwStatus('Error: ' + ((d as any).detail || resp.statusText));
        setRwPhase('idle');
        return;
      }

      const reader = resp.body!.getReader();
      const dec = new TextDecoder();
      let buf = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop()!;
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.status && ev.status !== 'done') setRwStatus(ev.message || ev.status);
            if (ev.status === 'done') {
              researchReport = ev.report || '';
              setRwResearchResult(researchReport);
              collectedSources = ev.sources || [];
              setRwSources(collectedSources);
            }
          } catch {}
        }
      }

      if (!researchReport) {
        setRwStatus('Research returned no results.');
        setRwPhase('idle');
        return;
      }

      // Phase 2: Write article using research
      setRwPhase('writing');
      setRwStatus('Writing article...');

      if (!wsReady || ws?.readyState !== WebSocket.OPEN) {
        setRwStatus('Error: AURA is offline.');
        setRwPhase('idle');
        return;
      }

      const sourcesList = collectedSources
        .slice(0, 10)
        .map((s, i) => `[${i + 1}] ${s.title || s.url || 'Source'}${s.url ? ` - ${s.url}` : ''}`)
        .join('\n');

      const writePrompt = `Based on the following research findings, write a comprehensive, well-structured long-form article about: ${topic}

RESEARCH FINDINGS:
${researchReport}

SOURCES:
${sourcesList}

Write a polished article with:
- An engaging introduction
- Well-organized sections with headers
- Key insights from the research integrated naturally
- Inline citations referencing the source numbers [1], [2], etc.
- A conclusion summarizing the key takeaways

Output only the article.`;

      setActiveStream({
        type: 'write',
        rawText: '',
        onFirstChunk: () => setRwArticle(''),
        onDone: (rawText) => {
          setRwArticle(rawText);
          setRwPhase('done');
          setRwStatus('');
        },
      });

      ws!.send(JSON.stringify({ type: 'chat', message: writePrompt, model: getModel('write'), conversation_id: null }));

    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setRwStatus('Error: ' + (err.message || 'Request failed'));
        setRwPhase('idle');
      }
    }
  }, [rwTopic, ws, wsReady, getModel, setActiveStream]);

  const cancelResearchWrite = useCallback(() => {
    rwAbortRef.current?.abort();
    setRwPhase('idle');
    setRwStatus('Cancelled');
  }, []);

  // Research-write article display
  const rwStreamText = (stream && stream !== true && stream.type === 'write' && writeMode === 'research-write' && rwPhase === 'writing')
    ? stream.rawText
    : null;
  const rwDisplay = rwStreamText !== null ? rwStreamText : rwArticle;

  /* ================================================================ */
  /*  Shared styles                                                    */
  /* ================================================================ */

  const pillBtn = (active: boolean, accent?: string) => ({
    padding: '3px 9px',
    background: active ? (accent || 'var(--pg2)') : 'var(--s2)',
    border: `1px solid ${active ? (accent ? accent : 'var(--p)') : 'var(--b1)'}`,
    borderRadius: 'var(--r-pill)' as const,
    color: active ? (accent ? '#fff' : 'var(--pl)') : 'var(--mu)',
    fontSize: '11px',
    cursor: 'pointer' as const,
    fontFamily: 'inherit',
    fontWeight: active ? 600 : 400,
    transition: 'all 0.15s ease',
  });

  const textAreaStyle: React.CSSProperties = {
    background: 'var(--s2)',
    border: '1px solid var(--b1)',
    borderRadius: 'var(--r-md)',
    color: 'var(--tx)',
    fontSize: '12.5px',
    padding: '8px 10px',
    resize: 'none',
    height: 80,
    outline: 'none',
    fontFamily: 'inherit',
    flexShrink: 0,
    width: '100%',
  };

  const resultBoxStyle: React.CSSProperties = {
    background: 'var(--s1)',
    border: '1px solid var(--b1)',
    borderRadius: 'var(--r-md)',
    padding: '10px',
  };

  /* ================================================================ */
  /*  RENDER                                                           */
  /* ================================================================ */

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* ============ Top mode tabs ============ */}
      <div className="flex flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
        {([
          { id: 'compose' as WriteMode, label: 'Compose' },
          { id: 'chat-draft' as WriteMode, label: 'Chat Draft' },
          { id: 'research-write' as WriteMode, label: 'Research Write' },
        ]).map(m => (
          <button
            key={m.id}
            onClick={() => setWriteMode(m.id)}
            style={{
              flex: 1,
              padding: '9px',
              background: 'none',
              border: 'none',
              borderBottom: writeMode === m.id ? '2px solid var(--p)' : '2px solid transparent',
              color: writeMode === m.id ? 'var(--pl)' : 'var(--mu)',
              fontSize: '12px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              fontWeight: writeMode === m.id ? 500 : 400,
              transition: 'all 0.15s ease',
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* ============ COMPOSE MODE ============ */}
      {writeMode === 'compose' && (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Write/Improve sub-tabs */}
          <div className="flex flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
            {(['write', 'improve'] as const).map(t => (
              <button
                key={t}
                onClick={() => { setTab(t); resetOutline(); }}
                style={{
                  flex: 1,
                  padding: '7px',
                  background: 'none',
                  border: 'none',
                  borderBottom: tab === t ? '2px solid var(--p)' : '2px solid transparent',
                  color: tab === t ? 'var(--pl)' : 'var(--mu)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  fontWeight: tab === t ? 500 : 400,
                }}
              >
                {t === 'write' ? 'Write' : 'Improve'}
              </button>
            ))}
          </div>

          <div className="flex-1 flex flex-col gap-2 p-3 overflow-y-auto">
            {/* Type pills */}
            {tab === 'write' && (
              <div className="flex flex-wrap gap-1 flex-shrink-0">
                {TYPES.map(t => (
                  <button key={t} onClick={() => setType(t)} style={pillBtn(type === t)}>
                    {t}
                  </button>
                ))}
              </div>
            )}

            {/* Tone pills */}
            <div className="flex flex-wrap gap-1 flex-shrink-0">
              {TONES.map(t => (
                <button key={t} onClick={() => setTone(t)} style={pillBtn(tone === t)}>
                  {t}
                </button>
              ))}
            </div>

            {/* Length pills */}
            <div className="flex flex-wrap gap-1 flex-shrink-0">
              {LENS.map(t => (
                <button key={t} onClick={() => setLen(t)} style={pillBtn(len === t)}>
                  {t}
                </button>
              ))}
            </div>

            {/* Word target + Outline toggle */}
            {tab === 'write' && (
              <div className="flex items-center gap-2 flex-shrink-0">
                <div className="flex items-center gap-1" style={{ flex: 1 }}>
                  <span style={{ fontSize: '10.5px', color: 'var(--mu)', whiteSpace: 'nowrap' }}>Target:</span>
                  <input
                    type="text"
                    placeholder="~500 words"
                    value={wordTarget}
                    onChange={e => setWordTarget(e.target.value.replace(/[^0-9]/g, ''))}
                    style={{
                      background: 'var(--s2)',
                      border: '1px solid var(--b1)',
                      borderRadius: 'var(--r-md)',
                      color: 'var(--tx)',
                      fontSize: '11px',
                      padding: '3px 7px',
                      outline: 'none',
                      fontFamily: 'inherit',
                      width: 80,
                    }}
                  />
                  {wordTarget && (
                    <span style={{ fontSize: '10px', color: 'var(--di)' }}>words</span>
                  )}
                </div>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 5,
                    cursor: 'pointer',
                    fontSize: '10.5px',
                    color: outlineFirst ? 'var(--pl)' : 'var(--mu)',
                    userSelect: 'none',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={outlineFirst}
                    onChange={e => { setOutlineFirst(e.target.checked); resetOutline(); }}
                    style={{ accentColor: 'var(--p)', width: 13, height: 13 }}
                  />
                  Outline first
                </label>
              </div>
            )}

            {/* Textarea */}
            <textarea
              ref={inputRef}
              placeholder={tab === 'improve' ? 'Paste text to improve...' : 'Describe what to write...'}
              onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); doCompose(); } }}
              style={textAreaStyle}
            />

            {/* Controls */}
            <div className="flex items-center justify-between flex-shrink-0">
              <ModelPill featureKey="write" />
              <button
                onClick={doCompose}
                disabled={!!activeStream}
                style={{
                  background: activeStream ? 'var(--s3)' : 'var(--p)',
                  border: 'none',
                  borderRadius: 'var(--r-md)',
                  color: 'white',
                  padding: '7px 18px',
                  cursor: activeStream ? 'not-allowed' : 'pointer',
                  fontSize: '12px',
                  fontFamily: 'inherit',
                  fontWeight: 500,
                }}
              >
                {activeStream ? '...' : outlineFirst && !outlineApproved ? 'Generate Outline' : 'Write'}
              </button>
            </div>

            {/* Outline preview (when outline-first is active) */}
            {outlineFirst && outlineDisplay && !outlineApproved && (
              <div style={{ ...resultBoxStyle, borderColor: 'var(--p)', borderStyle: 'dashed' }}>
                <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--pl)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>
                  Outline Preview
                </div>
                <div
                  className="md-body"
                  style={{ fontSize: '12px', lineHeight: 1.6 }}
                  dangerouslySetInnerHTML={{ __html: md(outlineDisplay) }}
                />
                {!activeStream && outline && (
                  <div className="flex items-center gap-2" style={{ marginTop: 8 }}>
                    <button
                      onClick={approveOutline}
                      style={{
                        background: 'var(--gr)',
                        border: 'none',
                        borderRadius: 'var(--r-md)',
                        color: 'white',
                        padding: '5px 14px',
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontFamily: 'inherit',
                        fontWeight: 600,
                      }}
                    >
                      Approve & Write
                    </button>
                    <button
                      onClick={resetOutline}
                      style={{
                        background: 'var(--s2)',
                        border: '1px solid var(--b1)',
                        borderRadius: 'var(--r-md)',
                        color: 'var(--mu)',
                        padding: '5px 12px',
                        cursor: 'pointer',
                        fontSize: '11px',
                        fontFamily: 'inherit',
                      }}
                    >
                      Discard
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Result */}
            {(composeDisplay || (activeStream && outlineApproved)) && !(!outlineApproved && outlineFirst && outlineDisplay) && (
              <div className="flex-1 overflow-y-auto" style={resultBoxStyle}>
                {composeDisplay ? (
                  <>
                    <div
                      className="md-body"
                      style={{ fontSize: '12.5px', lineHeight: 1.65 }}
                      dangerouslySetInnerHTML={{ __html: md(composeDisplay) }}
                    />
                    {/* Word count footer */}
                    {!activeStream && composeDisplay && (
                      <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid var(--b1)', fontSize: '10px', color: 'var(--di)', display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span>{countWords(composeDisplay)} words</span>
                        {wordTarget && (
                          <span style={{ color: Math.abs(countWords(composeDisplay) - parseInt(wordTarget)) < parseInt(wordTarget) * 0.15 ? 'var(--gr)' : 'var(--rd)' }}>
                            (target: {wordTarget})
                          </span>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="dots"><span /><span /><span /></div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ============ CHAT DRAFT MODE ============ */}
      {writeMode === 'chat-draft' && (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Messages area */}
          <div ref={draftScrollRef} className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
            {draftMessages.length === 0 && !activeStream && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 10, paddingTop: 32 }}>
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: '50%',
                    background: 'var(--pg)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid var(--b1)',
                  }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" stroke="var(--pl)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M9 10H15M9 13H12" stroke="var(--pl)" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ color: 'var(--tx)', fontSize: '13px', fontWeight: 600, marginBottom: 3 }}>Chat Draft</div>
                  <div style={{ color: 'var(--mu)', fontSize: '11px', lineHeight: 1.5, maxWidth: 220 }}>
                    Describe what you want written. Give feedback to iterate until it is perfect.
                  </div>
                </div>
              </div>
            )}

            {draftMessages.map((msg, i) => {
              const isLastAi = msg.role === 'ai' && i === draftMessages.length - 1;
              const displayContent = isLastAi && streamText !== null ? streamText : msg.text;

              return (
                <div
                  key={msg.id}
                  style={{
                    alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                    maxWidth: '90%',
                  }}
                >
                  <div
                    style={{
                      background: msg.role === 'user' ? 'var(--p)' : 'var(--s2)',
                      color: msg.role === 'user' ? '#fff' : 'var(--tx)',
                      borderRadius: msg.role === 'user' ? '12px 12px 4px 12px' : '12px 12px 12px 4px',
                      padding: '8px 12px',
                      fontSize: '12.5px',
                      lineHeight: 1.55,
                      border: msg.role === 'ai' ? '1px solid var(--b1)' : 'none',
                    }}
                  >
                    {msg.role === 'ai' && !displayContent ? (
                      <div className="dots"><span /><span /><span /></div>
                    ) : msg.role === 'ai' ? (
                      <div
                        className="md-body"
                        style={{ fontSize: '12.5px', lineHeight: 1.6 }}
                        dangerouslySetInnerHTML={{ __html: md(displayContent) }}
                      />
                    ) : (
                      <span>{displayContent}</span>
                    )}
                  </div>
                  {/* Word count on AI messages */}
                  {msg.role === 'ai' && displayContent && !activeStream && (
                    <div style={{ fontSize: '9.5px', color: 'var(--di)', marginTop: 3, marginLeft: 4 }}>
                      {countWords(displayContent)} words
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Accept / Reset buttons */}
          {draftFinal && !activeStream && (
            <div className="flex items-center gap-2 px-3 py-1" style={{ borderTop: '1px solid var(--b1)' }}>
              <button
                onClick={acceptDraft}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 5,
                  padding: '5px 14px',
                  background: draftAccepted ? 'var(--gr)' : 'var(--p)',
                  border: 'none',
                  borderRadius: 'var(--r-md)',
                  color: 'white',
                  fontSize: '11px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                  transition: 'all 0.15s ease',
                }}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
                  <path d="M3 8.5L6.5 12L13 4" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {draftAccepted ? 'Copied to clipboard!' : 'Accept & Copy'}
              </button>
              <button
                onClick={resetDraft}
                style={{
                  padding: '5px 12px',
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--mu)',
                  fontSize: '11px',
                  cursor: 'pointer',
                  fontFamily: 'inherit',
                }}
              >
                New Draft
              </button>
            </div>
          )}

          {/* Input area */}
          <div className="flex items-end gap-2 p-3 flex-shrink-0" style={{ borderTop: '1px solid var(--b1)' }}>
            <textarea
              ref={draftInputRef}
              value={draftInput}
              onChange={e => setDraftInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  sendDraftMessage();
                }
              }}
              placeholder={draftMessages.length === 0 ? 'Describe what to write...' : 'Give feedback to refine...'}
              rows={2}
              style={{
                ...textAreaStyle,
                height: 50,
                flex: 1,
              }}
            />
            <button
              onClick={sendDraftMessage}
              disabled={!!activeStream || !draftInput.trim()}
              style={{
                background: (activeStream || !draftInput.trim()) ? 'var(--s3)' : 'var(--p)',
                border: 'none',
                borderRadius: 'var(--r-md)',
                color: 'white',
                padding: '7px 14px',
                cursor: (activeStream || !draftInput.trim()) ? 'not-allowed' : 'pointer',
                fontSize: '11px',
                fontFamily: 'inherit',
                fontWeight: 600,
                flexShrink: 0,
                height: 50,
              }}
            >
              {activeStream ? '...' : 'Send'}
            </button>
          </div>
        </div>
      )}

      {/* ============ RESEARCH WRITE MODE ============ */}
      {writeMode === 'research-write' && (
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Input */}
          <div className="flex flex-col gap-2 p-3 flex-shrink-0" style={{ borderBottom: '1px solid var(--b1)' }}>
            <textarea
              value={rwTopic}
              onChange={e => setRwTopic(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                  e.preventDefault();
                  if (rwPhase === 'idle' || rwPhase === 'done') doResearchWrite();
                }
              }}
              placeholder="Enter a topic to research & write about..."
              rows={2}
              style={{ ...textAreaStyle, height: 56 }}
            />
            <div className="flex items-center justify-between">
              <ModelPill featureKey="write" />
              <div className="flex items-center gap-2">
                {(rwPhase === 'researching' || rwPhase === 'writing') && (
                  <button
                    onClick={cancelResearchWrite}
                    style={{
                      background: 'var(--rd)',
                      border: 'none',
                      borderRadius: 'var(--r-md)',
                      color: 'white',
                      padding: '6px 12px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      fontFamily: 'inherit',
                      fontWeight: 500,
                    }}
                  >
                    Cancel
                  </button>
                )}
                <button
                  onClick={doResearchWrite}
                  disabled={rwPhase === 'researching' || rwPhase === 'writing' || !rwTopic.trim()}
                  style={{
                    background: (rwPhase === 'researching' || rwPhase === 'writing' || !rwTopic.trim()) ? 'var(--s3)' : 'var(--p)',
                    border: 'none',
                    borderRadius: 'var(--r-md)',
                    color: 'white',
                    padding: '6px 16px',
                    cursor: (rwPhase === 'researching' || rwPhase === 'writing' || !rwTopic.trim()) ? 'not-allowed' : 'pointer',
                    fontSize: '12px',
                    fontFamily: 'inherit',
                    fontWeight: 600,
                  }}
                >
                  {rwPhase === 'researching' || rwPhase === 'writing' ? '...' : 'Research & Write'}
                </button>
              </div>
            </div>
          </div>

          {/* Output area */}
          <div ref={rwScrollRef} className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
            {/* Phase indicator */}
            {(rwPhase === 'researching' || rwPhase === 'writing') && (
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 12px',
                  background: 'var(--s2)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  animation: 'panelFadeIn 0.3s ease',
                }}
              >
                <div style={{ display: 'flex', gap: 4 }}>
                  {/* Phase dots */}
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: rwPhase === 'researching' ? 'var(--p)' : 'var(--gr)',
                      transition: 'all 0.3s ease',
                      animation: rwPhase === 'researching' ? 'dotPulse 1.2s ease-in-out infinite' : undefined,
                    }}
                  />
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: rwPhase === 'writing' ? 'var(--p)' : 'var(--s3)',
                      transition: 'all 0.3s ease',
                      animation: rwPhase === 'writing' ? 'dotPulse 1.2s ease-in-out infinite' : undefined,
                    }}
                  />
                </div>
                <span style={{ fontSize: '11.5px', color: 'var(--tx)', fontWeight: 500 }}>
                  {rwPhase === 'researching' ? 'Researching...' : 'Writing article...'}
                </span>
                {rwStatus && rwPhase === 'researching' && (
                  <span style={{ fontSize: '10px', color: 'var(--mu)' }}>{rwStatus}</span>
                )}
              </div>
            )}

            {/* Status for errors */}
            {rwStatus && rwStatus.startsWith('Error') && (
              <div style={{ color: 'var(--rd)', fontSize: '11.5px' }}>{rwStatus}</div>
            )}

            {/* Article output */}
            {rwDisplay && (
              <>
                <div style={resultBoxStyle}>
                  <div
                    className="md-body"
                    style={{ fontSize: '12.5px', lineHeight: 1.65 }}
                    dangerouslySetInnerHTML={{ __html: md(rwDisplay) }}
                  />
                  {/* Word count */}
                  {rwPhase === 'done' && rwDisplay && (
                    <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px solid var(--b1)', fontSize: '10px', color: 'var(--di)' }}>
                      {countWords(rwDisplay)} words
                    </div>
                  )}
                </div>
                {rwPhase === 'writing' && <span className="streaming-cursor" />}
              </>
            )}

            {/* Sources at bottom */}
            {rwSources.length > 0 && rwPhase === 'done' && (
              <div style={{ animation: 'fadeIn 0.3s ease' }}>
                <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
                  Sources ({rwSources.length})
                </div>
                {rwSources.slice(0, 10).map((s, i) => (
                  <div
                    key={i}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 6,
                      padding: '4px 0',
                      fontSize: '11px',
                      lineHeight: 1.4,
                    }}
                  >
                    <span style={{ fontSize: '9px', fontWeight: 700, color: 'var(--p)', background: 'var(--pg)', borderRadius: 3, padding: '1px 4px', flexShrink: 0 }}>
                      {i + 1}
                    </span>
                    <div style={{ minWidth: 0 }}>
                      {s.url ? (
                        <a href={s.url} target="_blank" rel="noopener" style={{ color: 'var(--pl)', textDecoration: 'none', wordBreak: 'break-all' }}>
                          {s.title || s.url}
                        </a>
                      ) : (
                        <span style={{ color: 'var(--tx)' }}>{s.title || 'Source'}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Empty state */}
            {rwPhase === 'idle' && !rwArticle && !rwStatus && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, gap: 10, paddingTop: 32 }}>
                <div
                  style={{
                    width: 44,
                    height: 44,
                    borderRadius: '50%',
                    background: 'var(--pg)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid var(--b1)',
                  }}
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                    <circle cx="11" cy="11" r="6" stroke="var(--pl)" strokeWidth="1.6" />
                    <path d="M15.5 15.5L20 20" stroke="var(--pl)" strokeWidth="1.6" strokeLinecap="round" />
                    <path d="M4 20L8 4L12 14L16 8L20 16" stroke="var(--pl)" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" opacity="0.4" />
                  </svg>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ color: 'var(--tx)', fontSize: '13px', fontWeight: 600, marginBottom: 3 }}>Research Write</div>
                  <div style={{ color: 'var(--mu)', fontSize: '11px', lineHeight: 1.5, maxWidth: 220 }}>
                    Enter a topic. We will research it first, then write a long-form article with citations.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
