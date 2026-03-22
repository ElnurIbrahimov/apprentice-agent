import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Terminal, ChevronRight, Copy, Play, Upload, Check, Pencil, Bug, RotateCcw, Zap, Square } from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import { HTTP, getAuthHeaders } from '../api';
import ModelPill from '../components/ModelPill';

/* ── Types ── */
interface OutputBlock {
  type: 'stdout' | 'image' | 'html' | 'error' | 'result';
  // stdout: text field; image: mime+data; html: content; error: ename+evalue+traceback; result: repr+type_name
  text?: string;
  mime?: string;
  data?: string;
  content?: string;
  ename?: string;
  evalue?: string;
  traceback?: string;
  repr?: string;
  type_name?: string;
}

interface VariableInfo {
  name: string;
  type_name: string;
  repr: string;
}

interface Exchange {
  id: string;
  prompt: string;
  code: string;
  outputs: OutputBlock[];
  variables: VariableInfo[];
  codeVisible: boolean;
  editing: boolean;
  editCode: string;
  phase: 'idle' | 'generating' | 'executing';
  executionStartTime?: number;
}

const SYSTEM_PROMPT =
  'You are a Python data analyst. Write and explain code to accomplish the user\'s request. ' +
  'Use matplotlib for charts (call plt.show()). Use print() for output. ' +
  'Return ONLY a JSON object: {"code": "...", "explanation": "..."}. No markdown fences.';

const QUICK_ACTIONS = [
  { label: 'Analyze CSV', icon: '📊', action: 'csv' },
  { label: 'Create Chart', icon: '📈', action: 'chart' },
  { label: 'Solve Math', icon: '🔢', action: 'math' },
  { label: 'Run Python', icon: '🐍', action: 'python' },
] as const;

let _exchangeCounter = 0;
function newId() { return `ex-${Date.now()}-${++_exchangeCounter}`; }

/* ── Sub-components ── */

function OutputBlockRenderer({ block, idx, exchangeId, code, onFix }: {
  block: OutputBlock; idx: number; exchangeId: string; code: string;
  onFix: (id: string, error: string, code: string) => void;
}) {
  if (block.type === 'image') {
    return (
      <div key={idx} style={{ marginTop: 8 }}>
        <img
          src={`data:${block.mime || 'image/png'};base64,${block.data}`}
          alt="Chart output"
          style={{ maxWidth: '100%', borderRadius: 'var(--r-sm)', border: '1px solid var(--b1)' }}
        />
      </div>
    );
  }
  if (block.type === 'html') {
    return (
      <div
        key={idx}
        className="code-table-wrap"
        style={{ marginTop: 8, overflow: 'auto', maxHeight: 300 }}
        dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(block.content || '') }}
      />
    );
  }
  if (block.type === 'error') {
    return (
      <div key={idx} style={{
        marginTop: 8, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
        borderRadius: 'var(--r-sm)', padding: '8px 10px',
      }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#ef4444', marginBottom: 4 }}>
          {block.ename}: {block.evalue}
        </div>
        {block.traceback && (
          <div style={{ fontFamily: 'monospace', fontSize: '10.5px', color: '#f87171', whiteSpace: 'pre-wrap', wordBreak: 'break-word', maxHeight: 200, overflow: 'auto' }}>
            {block.traceback}
          </div>
        )}
        <button
          onClick={() => onFix(exchangeId, `${block.ename}: ${block.evalue}\n${block.traceback || ''}`, code)}
          style={{
            marginTop: 6, display: 'inline-flex', alignItems: 'center', gap: 4,
            background: 'rgba(239,68,68,0.12)', border: '1px solid rgba(239,68,68,0.3)',
            borderRadius: 'var(--r-sm)', color: '#ef4444', padding: '4px 10px',
            fontSize: '11px', cursor: 'pointer', fontFamily: 'inherit',
          }}
        >
          <Bug size={12} /> Fix Error
        </button>
      </div>
    );
  }
  if (block.type === 'result') {
    return (
      <div key={idx} style={{
        marginTop: 8, display: 'flex', gap: 6, alignItems: 'baseline',
      }}>
        <span style={{ color: '#f97316', fontFamily: 'monospace', fontSize: '11px', fontWeight: 600 }}>Out:</span>
        <pre style={{
          margin: 0, fontFamily: 'monospace', fontSize: '11.5px', color: '#e6edf3',
          whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        }}>
          {block.repr}
        </pre>
        {block.type_name && (
          <span style={{ color: '#8b949e', fontSize: '10px' }}>({block.type_name})</span>
        )}
      </div>
    );
  }
  // stdout
  return (
    <pre key={idx} style={{
      marginTop: 8, background: '#0d1117', border: '1px solid var(--b1)',
      borderRadius: 'var(--r-sm)', padding: '10px 12px', fontFamily: 'monospace',
      fontSize: '11.5px', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
      overflow: 'auto', maxHeight: 300,
    }}>
      {block.text}
    </pre>
  );
}

function VariableInspector({ variables }: { variables: VariableInfo[] }) {
  const [open, setOpen] = useState(false);
  if (!variables.length) return null;
  return (
    <div style={{ marginTop: 6 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          background: 'none', border: 'none', color: '#8b949e', fontSize: '10.5px',
          cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4, padding: 0,
        }}
      >
        <ChevronRight size={10} style={{ transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
        Variables ({variables.length})
      </button>
      {open && (
        <div style={{
          marginTop: 4, background: '#161b22', border: '1px solid #30363d', borderRadius: 'var(--r-sm)',
          padding: '6px 8px', fontSize: '10.5px', fontFamily: 'monospace',
        }}>
          {variables.map((v, i) => (
            <div key={i} style={{ display: 'flex', gap: 8, padding: '2px 0', color: '#e6edf3' }}>
              <span style={{ color: '#79c0ff', minWidth: 60 }}>{v.name}</span>
              <span style={{ color: '#8b949e' }}>{v.type_name}</span>
              <span style={{ color: '#a5d6ff', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 160 }}>
                {v.repr}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ExecutionIndicator({ phase, startTime, onStop }: {
  phase: 'idle' | 'generating' | 'executing'; startTime?: number;
  onStop: () => void;
}) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (phase !== 'executing' || !startTime) { setElapsed(0); return; }
    const iv = setInterval(() => setElapsed(Math.floor((Date.now() - startTime) / 1000)), 200);
    return () => clearInterval(iv);
  }, [phase, startTime]);

  if (phase === 'idle') return null;
  return (
    <div style={{ padding: '12px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
      <div className="dots"><span /><span /><span /></div>
      <span style={{ fontSize: '11px', color: 'var(--mu)' }}>
        {phase === 'generating' ? 'Generating code…' : `Executing… ${elapsed}s`}
      </span>
      {phase === 'executing' && (
        <button onClick={onStop} style={{
          background: 'none', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
          color: '#ef4444', padding: '2px 6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
          fontSize: '10px', fontFamily: 'inherit',
        }}>
          <Square size={9} /> Stop
        </button>
      )}
    </div>
  );
}

/* ── Component ── */
export default function CodePanel() {
  const { getModel } = useStore();
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const exchangesRef = useRef<Exchange[]>([]);
  exchangesRef.current = exchanges;
  const [sessionId, setSessionId] = useState(() => `code-${Date.now()}`);
  const [hasSessionState, setHasSessionState] = useState(false);
  const [runOnlyMode, setRunOnlyMode] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generateAbortRef = useRef<AbortController | null>(null);
  const executeAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      if (generateAbortRef.current) generateAbortRef.current.abort();
      if (executeAbortRef.current) executeAbortRef.current.abort();
    };
  }, []);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  }, []);

  const updateExchange = useCallback((id: string, patch: Partial<Exchange>) => {
    setExchanges(prev => prev.map(e => e.id === id ? { ...e, ...patch } : e));
  }, []);

  const copyCode = useCallback(async (code: string, id: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopiedId(id);
      if (copiedTimerRef.current) clearTimeout(copiedTimerRef.current);
      copiedTimerRef.current = setTimeout(() => setCopiedId(null), 1500);
    } catch { /* clipboard not available (unfocused, permissions) — skip visual feedback */ }
  }, []);

  /* ── Generate code via chat endpoint ── */
  const generateCode = useCallback(async (prompt: string): Promise<{ code: string; explanation: string }> => {
    const model = getModel('code');
    if (generateAbortRef.current) generateAbortRef.current.abort();
    const ctrl = new AbortController();
    generateAbortRef.current = ctrl;
    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          message: `${SYSTEM_PROMPT}\n\n${prompt}`,
          model: model || undefined,
        }),
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const text = data.response || data.text || data.content || '';
      try {
        const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return { code: parsed.code || '', explanation: parsed.explanation || '' };
      } catch {
        if (text.includes('import ') || text.includes('def ') || text.includes('print(')) {
          const fenceMatch = text.match(/```(?:python)?\s*([\s\S]*?)```/);
          return { code: fenceMatch ? fenceMatch[1].trim() : text, explanation: '' };
        }
        return { code: text, explanation: '' };
      }
    } catch (err: any) {
      if (err.name === 'AbortError') throw err; // let caller handle cancellation
      throw new Error('Failed to generate code: ' + (err.message || err));
    }
  }, [getModel]);

  /* ── Execute code via real backend executor ── */
  const executeCode = useCallback(async (code: string): Promise<{ outputs: OutputBlock[]; variables: VariableInfo[] }> => {
    if (executeAbortRef.current) executeAbortRef.current.abort();
    const ctrl = new AbortController();
    executeAbortRef.current = ctrl;
    try {
      const resp = await fetch(`${HTTP}/api/code/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ code, session_id: sessionId, timeout: 60 }),
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        return {
          outputs: [{ type: 'error', ename: 'HTTPError', evalue: (d as any).detail || `HTTP ${resp.status}`, traceback: '' }],
          variables: [],
        };
      }
      const data = await resp.json();
      if (data.variables?.length) setHasSessionState(true);
      const outputs: OutputBlock[] = data.outputs || [];
      if (!outputs.length && data.success) {
        outputs.push({ type: 'stdout', text: '(no output)' });
      }
      return { outputs, variables: data.variables || [] };
    } catch (err: any) {
      if (err.name === 'AbortError') {
        return { outputs: [{ type: 'stdout', text: 'Execution cancelled.' }], variables: [] };
      }
      return {
        outputs: [{ type: 'error', ename: 'Error', evalue: err.message || 'Unknown error', traceback: '' }],
        variables: [],
      };
    }
  }, [sessionId]);

  /* ── Main submit (generate + execute) ── */
  const submit = useCallback(async (prompt: string) => {
    if (!prompt.trim()) return;
    const id = newId();
    const exchange: Exchange = {
      id, prompt: prompt.trim(), code: '', outputs: [], variables: [],
      codeVisible: false, editing: false, editCode: '', phase: 'generating',
    };
    setExchanges(prev => [...prev, exchange]);
    setInputValue('');
    scrollToBottom();

    try {
      const recentExchanges = exchangesRef.current.slice(-3);
      const contextParts = recentExchanges.map(e =>
        `Previous code:\n${e.code}\nOutput: ${e.outputs.map(o => o.text || o.evalue || '').join('\n')}`
      );
      const fullPrompt = contextParts.length
        ? `${contextParts.join('\n---\n')}\n\nNew request: ${prompt.trim()}`
        : prompt.trim();

      const { code } = await generateCode(fullPrompt);
      updateExchange(id, { code, phase: 'executing', executionStartTime: Date.now() });
      scrollToBottom();

      const { outputs, variables } = await executeCode(code);
      updateExchange(id, { outputs, variables, phase: 'idle' });
    } catch (err: any) {
      if (err.name === 'AbortError') {
        updateExchange(id, { outputs: [{ type: 'stdout', text: 'Cancelled.' }], phase: 'idle' });
      } else {
        updateExchange(id, {
          outputs: [{ type: 'error', ename: 'Error', evalue: err.message || 'Unknown error', traceback: '' }],
          phase: 'idle',
        });
      }
    }
    scrollToBottom();
  }, [generateCode, executeCode, updateExchange, scrollToBottom]);

  /* ── Run Only (Shift+Enter) — skip LLM, run code directly ── */
  const submitDirect = useCallback(async (code: string) => {
    if (!code.trim()) return;
    const id = newId();
    const exchange: Exchange = {
      id, prompt: '(direct execution)', code: code.trim(), outputs: [], variables: [],
      codeVisible: true, editing: false, editCode: '', phase: 'executing',
      executionStartTime: Date.now(),
    };
    setExchanges(prev => [...prev, exchange]);
    setInputValue('');
    scrollToBottom();

    const { outputs, variables } = await executeCode(code.trim());
    updateExchange(id, { outputs, variables, phase: 'idle' });
    scrollToBottom();
  }, [executeCode, updateExchange, scrollToBottom]);

  /* ── Re-run edited code ── */
  const rerun = useCallback(async (id: string, code: string) => {
    updateExchange(id, { code, editing: false, phase: 'executing', outputs: [], variables: [], executionStartTime: Date.now() });
    scrollToBottom();
    const { outputs, variables } = await executeCode(code);
    updateExchange(id, { outputs, variables, phase: 'idle' });
    scrollToBottom();
  }, [executeCode, updateExchange, scrollToBottom]);

  /* ── Fix error ── */
  const fixError = useCallback(async (id: string, errorMsg: string, originalCode: string) => {
    updateExchange(id, { phase: 'generating', outputs: [], variables: [] });
    scrollToBottom();
    try {
      const fixPrompt = `The following Python code produced an error. Fix it.\n\nCode:\n${originalCode}\n\nError:\n${errorMsg}`;
      const { code } = await generateCode(fixPrompt);
      updateExchange(id, { code, phase: 'executing', executionStartTime: Date.now() });
      const { outputs, variables } = await executeCode(code);
      updateExchange(id, { outputs, variables, phase: 'idle' });
    } catch (err: any) {
      updateExchange(id, {
        outputs: [{ type: 'error', ename: 'Error', evalue: err.message || 'Fix failed', traceback: '' }],
        phase: 'idle',
      });
    }
    scrollToBottom();
  }, [generateCode, executeCode, updateExchange, scrollToBottom]);

  /* ── CSV upload ── */
  const handleCsvUpload = useCallback(async (file: File) => {
    const id = newId();
    setExchanges(prev => [...prev, {
      id, prompt: `Analyze uploaded CSV: ${file.name}`, code: '', outputs: [], variables: [],
      codeVisible: false, editing: false, editCode: '', phase: 'generating' as const,
    }]);
    scrollToBottom();

    try {
      const form = new FormData();
      form.append('file', file);
      let filePath = file.name;
      try {
        const upResp = await fetch(`${HTTP}/api/upload`, { method: 'POST', body: form, headers: getAuthHeaders() });
        if (upResp.ok) {
          const upData = await upResp.json();
          filePath = upData.path || upData.filename || file.name;
        }
      } catch { /* use filename as fallback */ }

      const analysisPrompt =
        `Analyze the CSV file at "${filePath}". ` +
        'Show: shape, column names and dtypes, basic statistics (describe()), missing values count. ' +
        'Print everything clearly. Suggest 3 follow-up analyses.';

      const { code } = await generateCode(analysisPrompt);
      updateExchange(id, { code, phase: 'executing', executionStartTime: Date.now() });
      const { outputs, variables } = await executeCode(code);
      updateExchange(id, { outputs, variables, phase: 'idle' });
    } catch (err: any) {
      updateExchange(id, {
        outputs: [{ type: 'error', ename: 'Error', evalue: err.message || 'CSV analysis failed', traceback: '' }],
        phase: 'idle',
      });
    }
    scrollToBottom();
  }, [generateCode, executeCode, updateExchange, scrollToBottom]);

  /* ── Quick action handlers ── */
  const handleQuickAction = useCallback((action: string) => {
    if (action === 'csv') {
      fileRef.current?.click();
      return;
    }
    const prompts: Record<string, string> = {
      chart: 'Create a sample bar chart with matplotlib showing monthly sales data for 2024. Use a clean style.',
      math: '',
      python: '',
    };
    const prompt = prompts[action];
    if (prompt) {
      submit(prompt);
    } else {
      inputRef.current?.focus();
    }
  }, [submit]);

  /* ── Reset session ── */
  const resetSession = useCallback(async () => {
    try {
      await fetch(`${HTTP}/api/code/session/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch { /* ignore */ }
    setExchanges([]);
    setSessionId(`code-${Date.now()}`);
    setHasSessionState(false);
  }, [sessionId]);

  const stopExecution = useCallback(() => {
    if (generateAbortRef.current) generateAbortRef.current.abort();
    if (executeAbortRef.current) executeAbortRef.current.abort();
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pt-3 pb-1">
        <Terminal size={16} style={{ color: 'var(--pl)' }} />
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>Code Interpreter</span>
        {hasSessionState && (
          <span style={{
            fontSize: '10px', color: '#3fb950', background: 'rgba(63,185,80,0.1)',
            border: '1px solid rgba(63,185,80,0.3)', borderRadius: 'var(--r-pill)',
            padding: '1px 6px', fontWeight: 500,
          }}>
            Session
          </span>
        )}
        <span style={{ flex: 1 }} />
        <ModelPill featureKey="code" />
        {exchanges.length > 0 && (
          <button
            onClick={resetSession}
            title="Reset session"
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--mu)', padding: '3px 6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
              fontSize: '10px', fontFamily: 'inherit',
            }}
          >
            <RotateCcw size={10} /> Reset
          </button>
        )}
      </div>

      {/* Conversation thread */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto panel-scroll-root" style={{ padding: '8px 12px' }}>
        {exchanges.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 16, opacity: 0.7 }}>
            <Terminal size={36} style={{ color: 'var(--mu)' }} />
            <div style={{ fontSize: '12px', color: 'var(--mu)', textAlign: 'center', lineHeight: 1.5 }}>
              Ask me to analyze data, create charts, or run code.
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', maxWidth: 280 }}>
              {QUICK_ACTIONS.map(qa => (
                <button
                  key={qa.action}
                  onClick={() => handleQuickAction(qa.action)}
                  style={{
                    background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-pill)',
                    color: 'var(--tx)', padding: '6px 12px', fontSize: '11px', cursor: 'pointer',
                    fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 5,
                    transition: 'border-color 0.15s, background 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--p)'; e.currentTarget.style.background = 'var(--pg)'; }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--b1)'; e.currentTarget.style.background = 'var(--s2)'; }}
                >
                  <span>{qa.icon}</span> {qa.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {exchanges.map(ex => (
          <div key={ex.id} style={{ marginBottom: 16 }}>
            {/* User prompt */}
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 8 }}>
              <ChevronRight size={14} style={{ color: 'var(--pl)', marginTop: 2, flexShrink: 0 }} />
              <div style={{ fontSize: '12.5px', color: 'var(--tx)', fontWeight: 500 }}>{ex.prompt}</div>
            </div>

            {/* Code block (collapsible) */}
            {ex.code && (
              <div style={{
                background: '#0d1117', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                overflow: 'hidden', marginBottom: 8,
              }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px',
                  background: '#161b22', borderBottom: ex.codeVisible ? '1px solid #30363d' : 'none',
                }}>
                  <button
                    onClick={() => updateExchange(ex.id, { codeVisible: !ex.codeVisible })}
                    style={{
                      background: 'none', border: 'none', color: '#8b949e', fontSize: '11px',
                      cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4, padding: 0,
                    }}
                  >
                    <ChevronRight size={12} style={{
                      transform: ex.codeVisible ? 'rotate(90deg)' : 'rotate(0deg)',
                      transition: 'transform 0.15s ease',
                    }} />
                    {ex.codeVisible ? 'Hide Code' : 'Show Code'}
                  </button>
                  <span style={{ flex: 1 }} />
                  <button
                    onClick={() => copyCode(ex.code, ex.id)}
                    style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', padding: '2px', display: 'flex', alignItems: 'center' }}
                    title="Copy code"
                  >
                    {copiedId === ex.id ? <Check size={12} style={{ color: '#3fb950' }} /> : <Copy size={12} />}
                  </button>
                  <button
                    onClick={() => updateExchange(ex.id, { editing: !ex.editing, editCode: ex.code, codeVisible: true })}
                    style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', padding: '2px', display: 'flex', alignItems: 'center' }}
                    title="Edit & Re-run"
                  >
                    <Pencil size={12} />
                  </button>
                </div>

                {ex.codeVisible && !ex.editing && (
                  <pre style={{
                    margin: 0, padding: '10px 12px', fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
                    fontSize: '11px', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                    overflow: 'auto', maxHeight: 300, lineHeight: 1.5,
                  }}>
                    {highlightPython(ex.code)}
                  </pre>
                )}

                {ex.editing && (
                  <div style={{ padding: '8px' }}>
                    <textarea
                      value={ex.editCode}
                      onChange={e => updateExchange(ex.id, { editCode: e.target.value })}
                      style={{
                        width: '100%', minHeight: 120, background: '#0d1117', color: '#e6edf3',
                        border: '1px solid #30363d', borderRadius: 'var(--r-sm)', padding: '8px 10px',
                        fontFamily: "'JetBrains Mono', 'Fira Code', monospace", fontSize: '11px',
                        resize: 'vertical', outline: 'none', lineHeight: 1.5,
                      }}
                    />
                    <div style={{ display: 'flex', gap: 6, marginTop: 6, justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => updateExchange(ex.id, { editing: false })}
                        style={{
                          background: '#21262d', border: '1px solid #30363d', borderRadius: 'var(--r-sm)',
                          color: '#8b949e', padding: '4px 12px', fontSize: '11px', cursor: 'pointer', fontFamily: 'inherit',
                        }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => rerun(ex.id, ex.editCode)}
                        style={{
                          background: 'var(--p)', border: 'none', borderRadius: 'var(--r-sm)',
                          color: 'white', padding: '4px 12px', fontSize: '11px', cursor: 'pointer',
                          fontFamily: 'inherit', display: 'flex', alignItems: 'center', gap: 4,
                        }}
                      >
                        <Play size={11} /> Re-run
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Execution indicator */}
            <ExecutionIndicator phase={ex.phase} startTime={ex.executionStartTime} onStop={stopExecution} />

            {/* Outputs */}
            {ex.outputs.map((out, i) => (
              <OutputBlockRenderer key={i} block={out} idx={i} exchangeId={ex.id} code={ex.code} onFix={fixError} />
            ))}

            {/* Variable inspector */}
            <VariableInspector variables={ex.variables} />
          </div>
        ))}
      </div>

      {/* Input area */}
      <div style={{ padding: '8px 12px 12px', borderTop: '1px solid var(--b1)' }}>
        {exchanges.length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginBottom: 6, flexWrap: 'wrap' }}>
            {QUICK_ACTIONS.map(qa => (
              <button
                key={qa.action}
                onClick={() => handleQuickAction(qa.action)}
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-pill)',
                  color: 'var(--mu)', padding: '3px 8px', fontSize: '10px', cursor: 'pointer', fontFamily: 'inherit',
                }}
              >
                {qa.icon} {qa.label}
              </button>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
          <button
            onClick={() => fileRef.current?.click()}
            title="Upload CSV"
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--mu)', padding: '7px', cursor: 'pointer', display: 'flex', alignItems: 'center', flexShrink: 0,
            }}
          >
            <Upload size={14} />
          </button>
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            placeholder={runOnlyMode ? 'Enter Python code… (Shift+Enter to run)' : 'Ask me to analyze data, create charts, or run code...'}
            onKeyDown={e => {
              if (e.key === 'Enter' && e.shiftKey) {
                e.preventDefault();
                submitDirect(inputValue);
              } else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                runOnlyMode ? submitDirect(inputValue) : submit(inputValue);
              }
            }}
            rows={1}
            style={{
              flex: 1, background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
              padding: '8px 10px', resize: 'none', outline: 'none', fontFamily: runOnlyMode ? "'JetBrains Mono', monospace" : 'inherit',
              minHeight: 36, maxHeight: 100,
            }}
          />
          <button
            onClick={() => setRunOnlyMode(!runOnlyMode)}
            title={runOnlyMode ? 'Run Only mode (click for AI mode)' : 'AI mode (click for Run Only)'}
            style={{
              background: runOnlyMode ? 'rgba(249,115,22,0.15)' : 'var(--s2)',
              border: `1px solid ${runOnlyMode ? 'rgba(249,115,22,0.4)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-sm)', color: runOnlyMode ? '#f97316' : 'var(--mu)',
              padding: '7px', cursor: 'pointer', display: 'flex', alignItems: 'center', flexShrink: 0,
            }}
          >
            <Zap size={14} />
          </button>
          <button
            onClick={() => runOnlyMode ? submitDirect(inputValue) : submit(inputValue)}
            disabled={!inputValue.trim()}
            style={{
              background: inputValue.trim() ? 'var(--p)' : 'var(--s3)',
              border: 'none', borderRadius: 'var(--r-sm)', color: 'white',
              padding: '7px 12px', cursor: inputValue.trim() ? 'pointer' : 'not-allowed',
              display: 'flex', alignItems: 'center', gap: 4, fontSize: '12px',
              fontFamily: 'inherit', flexShrink: 0,
            }}
          >
            <Play size={13} /> Run
          </button>
        </div>
        <div style={{ fontSize: '9.5px', color: 'var(--mu)', marginTop: 4, opacity: 0.7 }}>
          {runOnlyMode ? 'Shift+Enter: run code directly' : 'Ctrl+Enter: generate & run · Shift+Enter: run directly'}
        </div>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept=".csv,.tsv,.xlsx,.xls"
        style={{ display: 'none' }}
        onChange={e => {
          const file = e.target.files?.[0];
          if (file) handleCsvUpload(file);
          e.target.value = '';
        }}
      />
    </div>
  );
}

/* ── Minimal Python syntax highlighting ── */
function highlightPython(code: string): React.ReactNode[] {
  const lines = code.split('\n');
  return lines.map((line, i) => (
    <React.Fragment key={i}>
      {i > 0 && '\n'}
      {highlightLine(line)}
    </React.Fragment>
  ));
}

function highlightLine(line: string): React.ReactNode[] {
  const tokens: React.ReactNode[] = [];
  // Note: triple-quoted strings removed — they span multiple lines and can't match in per-line highlighting
  const regex = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(import|from|as|def|class|return|if|elif|else|for|while|in|not|and|or|is|with|try|except|finally|raise|yield|lambda|pass|break|continue|True|False|None|print|len|range|list|dict|set|tuple|int|float|str|bool|type|isinstance|open|self)\b|(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(line)) !== null) {
    if (match.index > lastIndex) {
      tokens.push(line.slice(lastIndex, match.index));
    }
    if (match[1]) {
      // Comment
      tokens.push(<span key={match.index} style={{ color: '#8b949e', fontStyle: 'italic' }}>{match[1]}</span>);
    } else if (match[2]) {
      // String
      tokens.push(<span key={match.index} style={{ color: '#a5d6ff' }}>{match[2]}</span>);
    } else if (match[3]) {
      // Keyword
      tokens.push(<span key={match.index} style={{ color: '#ff7b72' }}>{match[3]}</span>);
    } else if (match[4]) {
      // Number
      tokens.push(<span key={match.index} style={{ color: '#79c0ff' }}>{match[4]}</span>);
    } else {
      tokens.push(match[0]);
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < line.length) {
    tokens.push(line.slice(lastIndex));
  }

  return tokens;
}
