import React, { useCallback, useRef, useState } from 'react';
import { Terminal, ChevronRight, Copy, Play, Upload, X, Check, Pencil, Bug } from 'lucide-react';
import { useStore } from '../store';
import { HTTP } from '../api';
import ModelPill from '../components/ModelPill';

/* ── Types ── */
interface CodeOutput {
  type: 'text' | 'image' | 'table' | 'error';
  content: string;       // text, base64 data-uri, HTML table, or error message
}

interface Exchange {
  id: string;
  prompt: string;
  code: string;
  outputs: CodeOutput[];
  codeVisible: boolean;
  editing: boolean;
  editCode: string;
  loading: boolean;
}

const SYSTEM_PROMPT =
  'You are a Python data analyst. Write and explain code to accomplish the user\'s request. ' +
  'Use matplotlib for charts (call plt.savefig to a temp file). Format output clearly. ' +
  'Return ONLY a JSON object: {"code": "...", "explanation": "..."}. No markdown fences.';

const QUICK_ACTIONS = [
  { label: 'Analyze CSV', icon: '📊', action: 'csv' },
  { label: 'Create Chart', icon: '📈', action: 'chart' },
  { label: 'Solve Math', icon: '🔢', action: 'math' },
  { label: 'Run Python', icon: '🐍', action: 'python' },
] as const;

let _exchangeCounter = 0;
function newId() { return `ex-${Date.now()}-${++_exchangeCounter}`; }

/* ── Component ── */
export default function CodePanel() {
  const { getModel } = useStore();
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [sessionId] = useState(() => `code-${Date.now()}`);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    });
  }, []);

  /* ── Helpers ── */
  const updateExchange = useCallback((id: string, patch: Partial<Exchange>) => {
    setExchanges(prev => prev.map(e => e.id === id ? { ...e, ...patch } : e));
  }, []);

  const copyCode = useCallback((code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  }, []);

  /* ── Generate code via chat endpoint ── */
  const generateCode = useCallback(async (prompt: string): Promise<{ code: string; explanation: string }> => {
    const model = getModel('code');
    try {
      const resp = await fetch(`${HTTP}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          system_prompt: SYSTEM_PROMPT,
          model: model || undefined,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const text = data.response || data.text || data.content || '';
      // Try parsing JSON from the response
      try {
        const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return { code: parsed.code || '', explanation: parsed.explanation || '' };
      } catch {
        // If not JSON, treat whole response as code if it looks like Python
        if (text.includes('import ') || text.includes('def ') || text.includes('print(')) {
          // Extract code from markdown fences if present
          const fenceMatch = text.match(/```(?:python)?\s*([\s\S]*?)```/);
          return { code: fenceMatch ? fenceMatch[1].trim() : text, explanation: '' };
        }
        return { code: text, explanation: '' };
      }
    } catch (err: any) {
      throw new Error('Failed to generate code: ' + (err.message || err));
    }
  }, [getModel]);

  /* ── Execute code ── */
  const executeCode = useCallback(async (code: string): Promise<CodeOutput[]> => {
    try {
      const resp = await fetch(`${HTTP}/api/tools/code/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, session_id: sessionId }),
      });
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        return [{ type: 'error', content: (d as any).detail || `Execution failed (HTTP ${resp.status})` }];
      }
      const data = await resp.json();
      const outputs: CodeOutput[] = [];
      if (data.stdout) outputs.push({ type: 'text', content: data.stdout });
      if (data.image || data.plot) outputs.push({ type: 'image', content: data.image || data.plot });
      if (data.table) outputs.push({ type: 'table', content: data.table });
      if (data.error) outputs.push({ type: 'error', content: data.error });
      if (outputs.length === 0 && data.result) outputs.push({ type: 'text', content: String(data.result) });
      if (outputs.length === 0) outputs.push({ type: 'text', content: 'Code executed successfully (no output).' });
      return outputs;
    } catch {
      // Backend endpoint doesn't exist — run via chat fallback
      return [{ type: 'error', content: 'Code execution endpoint not available. Backend may need /api/tools/code/execute.' }];
    }
  }, [sessionId]);

  /* ── Main submit ── */
  const submit = useCallback(async (prompt: string) => {
    if (!prompt.trim()) return;
    const id = newId();
    const exchange: Exchange = {
      id, prompt: prompt.trim(), code: '', outputs: [],
      codeVisible: false, editing: false, editCode: '', loading: true,
    };
    setExchanges(prev => [...prev, exchange]);
    setInputValue('');
    scrollToBottom();

    try {
      // Build context from previous exchanges
      const contextParts = exchanges.slice(-3).map(e =>
        `Previous code:\n${e.code}\nOutput: ${e.outputs.map(o => o.content).join('\n')}`
      );
      const fullPrompt = contextParts.length
        ? `${contextParts.join('\n---\n')}\n\nNew request: ${prompt.trim()}`
        : prompt.trim();

      const { code } = await generateCode(fullPrompt);
      updateExchange(id, { code });
      scrollToBottom();

      const outputs = await executeCode(code);
      updateExchange(id, { outputs, loading: false });
    } catch (err: any) {
      updateExchange(id, {
        outputs: [{ type: 'error', content: err.message || 'Unknown error' }],
        loading: false,
      });
    }
    scrollToBottom();
  }, [exchanges, generateCode, executeCode, updateExchange, scrollToBottom]);

  /* ── Re-run edited code ── */
  const rerun = useCallback(async (id: string, code: string) => {
    updateExchange(id, { code, editing: false, loading: true, outputs: [] });
    scrollToBottom();
    const outputs = await executeCode(code);
    updateExchange(id, { outputs, loading: false });
    scrollToBottom();
  }, [executeCode, updateExchange, scrollToBottom]);

  /* ── Fix error ── */
  const fixError = useCallback(async (id: string, errorMsg: string, originalCode: string) => {
    updateExchange(id, { loading: true, outputs: [] });
    scrollToBottom();
    try {
      const fixPrompt = `The following Python code produced an error. Fix it.\n\nCode:\n${originalCode}\n\nError:\n${errorMsg}`;
      const { code } = await generateCode(fixPrompt);
      updateExchange(id, { code });
      const outputs = await executeCode(code);
      updateExchange(id, { outputs, loading: false });
    } catch (err: any) {
      updateExchange(id, {
        outputs: [{ type: 'error', content: err.message || 'Fix failed' }],
        loading: false,
      });
    }
    scrollToBottom();
  }, [generateCode, executeCode, updateExchange, scrollToBottom]);

  /* ── CSV upload ── */
  const handleCsvUpload = useCallback(async (file: File) => {
    const id = newId();
    setExchanges(prev => [...prev, {
      id, prompt: `Analyze uploaded CSV: ${file.name}`, code: '', outputs: [],
      codeVisible: false, editing: false, editCode: '', loading: true,
    }]);
    scrollToBottom();

    try {
      // Try uploading to backend
      const form = new FormData();
      form.append('file', file);
      let filePath = file.name;
      try {
        const upResp = await fetch(`${HTTP}/api/upload`, { method: 'POST', body: form });
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
      updateExchange(id, { code });
      const outputs = await executeCode(code);
      updateExchange(id, { outputs, loading: false });
    } catch (err: any) {
      updateExchange(id, {
        outputs: [{ type: 'error', content: err.message || 'CSV analysis failed' }],
        loading: false,
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

  /* ── Render output block ── */
  const renderOutput = (output: CodeOutput, idx: number, exchangeId: string, code: string) => {
    if (output.type === 'image') {
      const src = output.content.startsWith('data:') ? output.content : `data:image/png;base64,${output.content}`;
      return (
        <div key={idx} style={{ marginTop: 8 }}>
          <img
            src={src}
            alt="Chart output"
            style={{ maxWidth: '100%', borderRadius: 'var(--r-sm)', border: '1px solid var(--b1)' }}
          />
        </div>
      );
    }
    if (output.type === 'table') {
      return (
        <div
          key={idx}
          className="code-table-wrap"
          style={{ marginTop: 8, overflow: 'auto', maxHeight: 300 }}
          dangerouslySetInnerHTML={{ __html: output.content }}
        />
      );
    }
    if (output.type === 'error') {
      return (
        <div key={idx} style={{
          marginTop: 8, background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)',
          borderRadius: 'var(--r-sm)', padding: '8px 10px',
        }}>
          <div style={{ fontFamily: 'monospace', fontSize: '11.5px', color: '#ef4444', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
            {output.content}
          </div>
          <button
            onClick={() => fixError(exchangeId, output.content, code)}
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
    // text
    return (
      <pre key={idx} style={{
        marginTop: 8, background: '#0d1117', border: '1px solid var(--b1)',
        borderRadius: 'var(--r-sm)', padding: '10px 12px', fontFamily: 'monospace',
        fontSize: '11.5px', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
        overflow: 'auto', maxHeight: 300,
      }}>
        {output.content}
      </pre>
    );
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pt-3 pb-1">
        <Terminal size={16} style={{ color: 'var(--pl)' }} />
        <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>Code Interpreter</span>
        <span style={{ flex: 1 }} />
        <ModelPill featureKey="code" />
        {exchanges.length > 0 && (
          <button
            onClick={() => setExchanges([])}
            title="Clear session"
            style={{
              background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-sm)',
              color: 'var(--mu)', padding: '3px 6px', cursor: 'pointer', display: 'flex', alignItems: 'center',
            }}
          >
            <X size={12} />
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
            {/* Quick actions */}
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
            <div style={{
              display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 8,
            }}>
              <ChevronRight size={14} style={{ color: 'var(--pl)', marginTop: 2, flexShrink: 0 }} />
              <div style={{ fontSize: '12.5px', color: 'var(--tx)', fontWeight: 500 }}>{ex.prompt}</div>
            </div>

            {/* Code block (collapsible) */}
            {ex.code && (
              <div style={{
                background: '#0d1117', border: '1px solid var(--b1)', borderRadius: 'var(--r-md)',
                overflow: 'hidden', marginBottom: 8,
              }}>
                {/* Code header */}
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
                    <ChevronRight
                      size={12}
                      style={{
                        transform: ex.codeVisible ? 'rotate(90deg)' : 'rotate(0deg)',
                        transition: 'transform 0.15s ease',
                      }}
                    />
                    {ex.codeVisible ? 'Hide Code' : 'Show Code'}
                  </button>
                  <span style={{ flex: 1 }} />
                  <button
                    onClick={() => copyCode(ex.code, ex.id)}
                    style={{
                      background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer',
                      padding: '2px', display: 'flex', alignItems: 'center',
                    }}
                    title="Copy code"
                  >
                    {copiedId === ex.id ? <Check size={12} style={{ color: '#3fb950' }} /> : <Copy size={12} />}
                  </button>
                  <button
                    onClick={() => updateExchange(ex.id, { editing: !ex.editing, editCode: ex.code, codeVisible: true })}
                    style={{
                      background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer',
                      padding: '2px', display: 'flex', alignItems: 'center',
                    }}
                    title="Edit & Re-run"
                  >
                    <Pencil size={12} />
                  </button>
                </div>

                {/* Code content */}
                {ex.codeVisible && !ex.editing && (
                  <pre style={{
                    margin: 0, padding: '10px 12px', fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
                    fontSize: '11px', color: '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                    overflow: 'auto', maxHeight: 300, lineHeight: 1.5,
                  }}>
                    {highlightPython(ex.code)}
                  </pre>
                )}

                {/* Edit mode */}
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

            {/* Loading indicator */}
            {ex.loading && (
              <div style={{ padding: '12px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
                <div className="dots"><span /><span /><span /></div>
                <span style={{ fontSize: '11px', color: 'var(--mu)' }}>
                  {ex.code ? 'Executing…' : 'Generating code…'}
                </span>
              </div>
            )}

            {/* Outputs */}
            {ex.outputs.map((out, i) => renderOutput(out, i, ex.id, ex.code))}
          </div>
        ))}
      </div>

      {/* Input area */}
      <div style={{ padding: '8px 12px 12px', borderTop: '1px solid var(--b1)' }}>
        {/* Quick actions row when conversation started */}
        {exchanges.length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginBottom: 6, flexWrap: 'wrap' }}>
            {QUICK_ACTIONS.map(qa => (
              <button
                key={qa.action}
                onClick={() => handleQuickAction(qa.action)}
                style={{
                  background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 'var(--r-pill)',
                  color: 'var(--mu)', padding: '3px 8px', fontSize: '10px', cursor: 'pointer',
                  fontFamily: 'inherit',
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
              color: 'var(--mu)', padding: '7px', cursor: 'pointer', display: 'flex', alignItems: 'center',
              flexShrink: 0,
            }}
          >
            <Upload size={14} />
          </button>
          <textarea
            ref={inputRef}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            placeholder="Ask me to analyze data, create charts, or run code..."
            onKeyDown={e => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                submit(inputValue);
              }
            }}
            rows={1}
            style={{
              flex: 1, background: 'var(--s2)', border: '1px solid var(--b1)',
              borderRadius: 'var(--r-md)', color: 'var(--tx)', fontSize: '12px',
              padding: '8px 10px', resize: 'none', outline: 'none', fontFamily: 'inherit',
              minHeight: 36, maxHeight: 100,
            }}
          />
          <button
            onClick={() => submit(inputValue)}
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
      </div>

      {/* Hidden file input */}
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
  // Simple token-based highlighting
  const regex = /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|("""[\s\S]*?"""|'''[\s\S]*?''')|\b(import|from|as|def|class|return|if|elif|else|for|while|in|not|and|or|is|with|try|except|finally|raise|yield|lambda|pass|break|continue|True|False|None|print|len|range|list|dict|set|tuple|int|float|str|bool|type|isinstance|open|self)\b|(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(line)) !== null) {
    // Plain text before match
    if (match.index > lastIndex) {
      tokens.push(line.slice(lastIndex, match.index));
    }

    if (match[1]) {
      // Comment
      tokens.push(<span key={match.index} style={{ color: '#8b949e', fontStyle: 'italic' }}>{match[1]}</span>);
    } else if (match[2] || match[3]) {
      // String
      tokens.push(<span key={match.index} style={{ color: '#a5d6ff' }}>{match[2] || match[3]}</span>);
    } else if (match[4]) {
      // Keyword
      tokens.push(<span key={match.index} style={{ color: '#ff7b72' }}>{match[4]}</span>);
    } else if (match[5]) {
      // Number
      tokens.push(<span key={match.index} style={{ color: '#79c0ff' }}>{match[5]}</span>);
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
