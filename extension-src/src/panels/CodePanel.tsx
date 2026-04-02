import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Terminal, ChevronRight, Copy, Play, Upload, Check, Pencil, Bug, RotateCcw, Zap, Square, Globe, Server, FilePlus, X, FileCode2, ExternalLink, Sparkles, Presentation } from 'lucide-react';
import DOMPurify from 'dompurify';
import { useStore } from '../store';
import { HTTP, getAuthHeaders } from '../api';
import CodeEditor from '../components/CodeEditor';
import ModelPill from '../components/ModelPill';
import { getPyodideExecutor, PyodideExecutor } from '../utils/PyodideExecutor';
import type { OutputBlock as PyOutputBlock, VariableInfo as PyVariableInfo } from '../utils/PyodideExecutor';
import OfflineBanner from '../components/OfflineBanner';

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

/* ── Multi-file Python support ── */
interface PythonFile {
  name: string;
  content: string;
}

const DEFAULT_FILE = 'main.py';

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

const handoffBtnStyle: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: 3,
  background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.25)',
  borderRadius: 'var(--r-sm)', color: '#a78bfa', padding: '3px 8px',
  fontSize: '9.5px', cursor: 'pointer', fontFamily: 'inherit', marginTop: 4, marginRight: 4,
};

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
        <div>
          <button
            style={handoffBtnStyle}
            onClick={() => {
              const { handoffToPanel } = useStore.getState();
              const imgHtml = `<img src="data:${block.mime || 'image/png'};base64,${block.data}" style="max-width:100%;" />`;
              handoffToPanel('webcreator', { code: `<!DOCTYPE html><html><body style="margin:0;display:flex;justify-content:center;padding:20px;">${imgHtml}</body></html>`, from: 'Code' });
            }}
          >
            <ExternalLink size={9} /> Web Creator
          </button>
          <button
            style={handoffBtnStyle}
            onClick={() => {
              const { handoffToPanel } = useStore.getState();
              const imgHtml = `<img src="data:${block.mime || 'image/png'};base64,${block.data}" style="max-width:100%;" />`;
              handoffToPanel('artifacts', { code: imgHtml, type: 'html', from: 'Code' });
            }}
          >
            <Sparkles size={9} /> Artifact
          </button>
        </div>
      </div>
    );
  }
  if (block.type === 'html') {
    return (
      <div key={idx}>
        <div
          className="code-table-wrap"
          style={{ marginTop: 8, overflow: 'auto', maxHeight: 300 }}
          dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(block.content || '') }}
        />
        <div>
          <button
            style={handoffBtnStyle}
            onClick={() => {
              const { handoffToPanel } = useStore.getState();
              handoffToPanel('webcreator', { code: block.content || '', from: 'Code' });
            }}
          >
            <ExternalLink size={9} /> Web Creator
          </button>
          <button
            style={handoffBtnStyle}
            onClick={() => {
              const { handoffToPanel } = useStore.getState();
              handoffToPanel('artifacts', { code: block.content || '', type: 'html', from: 'Code' });
            }}
          >
            <Sparkles size={9} /> Artifact
          </button>
        </div>
      </div>
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
  const [pyodideState, setPyodideState] = useState<string>('idle');
  const [forceBackend, setForceBackend] = useState(false);
  const executorRef = useRef(getPyodideExecutor(setPyodideState));
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const copiedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const generateAbortRef = useRef<AbortController | null>(null);
  const executeAbortRef = useRef<AbortController | null>(null);

  /* ── Multi-file state ── */
  const [pythonFiles, setPythonFiles] = useState<PythonFile[]>([{ name: DEFAULT_FILE, content: '' }]);
  const [activeFileName, setActiveFileName] = useState(DEFAULT_FILE);
  const [showFileTabs, setShowFileTabs] = useState(false);

  const activeFile = pythonFiles.find(f => f.name === activeFileName) || pythonFiles[0];

  const addPythonFile = useCallback((name?: string) => {
    const baseName = name || 'untitled';
    let finalName = baseName.endsWith('.py') ? baseName : `${baseName}.py`;
    const existing = new Set(pythonFiles.map(f => f.name));
    let counter = 1;
    while (existing.has(finalName)) {
      finalName = baseName.replace(/\.py$/, '') + `_${counter}.py`;
      counter++;
    }
    setPythonFiles(prev => [...prev, { name: finalName, content: '' }]);
    setActiveFileName(finalName);
    setShowFileTabs(true);
  }, [pythonFiles]);

  const removePythonFile = useCallback((name: string) => {
    if (pythonFiles.length <= 1) return;
    setPythonFiles(prev => prev.filter(f => f.name !== name));
    if (activeFileName === name) {
      setActiveFileName(pythonFiles.find(f => f.name !== name)?.name || DEFAULT_FILE);
    }
  }, [pythonFiles, activeFileName]);

  const updateFileContent = useCallback((name: string, content: string) => {
    setPythonFiles(prev => prev.map(f => f.name === name ? { ...f, content } : f));
  }, []);

  /** Build import preamble: make sibling .py files importable by writing them into Pyodide's virtual FS */
  const buildMultiFilePreamble = useCallback((): string => {
    if (pythonFiles.length <= 1) return '';
    const parts = pythonFiles
      .filter(f => f.name !== activeFileName && f.content.trim())
      .map(f => {
        const moduleName = f.name.replace(/\.py$/, '');
        const escaped = f.content.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/\n/g, '\\n');
        return `open('${f.name}', 'w').write('${escaped}')`;
      });
    return parts.length ? parts.join('\n') + '\n' : '';
  }, [pythonFiles, activeFileName]);

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

  /* ── Generate code via raw LLM endpoint (bypasses agent pipeline) ── */
  const generateCode = useCallback(async (prompt: string): Promise<{ code: string; explanation: string }> => {
    const model = getModel('code');
    if (generateAbortRef.current) generateAbortRef.current.abort();
    const ctrl = new AbortController();
    generateAbortRef.current = ctrl;
    try {
      const resp = await fetch(`${HTTP}/api/generate/raw`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          message: prompt,
          system_prompt: SYSTEM_PROMPT,
          model: model || undefined,
        }),
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      // Collect SSE stream into full text
      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buf = '', text = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'chunk' && parsed.content) text += parsed.content;
            else if (parsed.type === 'error') throw new Error(parsed.error || 'Generation error');
          } catch (e: any) { if (!(e instanceof SyntaxError)) throw e; }
        }
      }
      // Parse JSON response from LLM
      try {
        const cleaned = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
        const parsed = JSON.parse(cleaned);
        return { code: parsed.code || '', explanation: parsed.explanation || '' };
      } catch {
        // Fallback: extract code from fences or raw text
        if (text.includes('import ') || text.includes('def ') || text.includes('print(')) {
          const fenceMatch = text.match(/```(?:python)?\s*([\s\S]*?)```/);
          return { code: fenceMatch ? fenceMatch[1].trim() : text, explanation: '' };
        }
        return { code: text, explanation: '' };
      }
    } catch (err: any) {
      if (err.name === 'AbortError') throw err;
      throw new Error('Failed to generate code: ' + (err.message || err));
    }
  }, [getModel]);

  /* ── Execute code via Pyodide (browser) or backend (server) ── */
  const executeCode = useCallback(async (code: string): Promise<{ outputs: OutputBlock[]; variables: VariableInfo[] }> => {
    if (executeAbortRef.current) executeAbortRef.current.abort();
    const ctrl = new AbortController();
    executeAbortRef.current = ctrl;

    return new Promise((resolve) => {
      let resolved = false;
      const outputs: OutputBlock[] = [];
      let variables: VariableInfo[] = [];

      const finish = (success: boolean) => {
        if (resolved) return;
        resolved = true;
        if (!outputs.length && success) outputs.push({ type: 'stdout', text: '(no output)' });
        resolve({ outputs, variables });
      };

      // Handle abort
      ctrl.signal.addEventListener('abort', () => {
        if (!resolved) { resolved = true; resolve({ outputs: [{ type: 'stdout', text: 'Execution cancelled.' }], variables: [] }); }
      });

      // Safety timeout: 5 minutes — prevents infinite hang if executor crashes
      const safetyTimer = setTimeout(() => {
        if (!resolved) {
          outputs.push({ type: 'error', ename: 'Timeout', evalue: 'Execution timed out after 5 minutes', traceback: '' });
          finish(false);
        }
      }, 300000);

      // Prepend sibling Python files so cross-file imports work in Pyodide
      const preamble = buildMultiFilePreamble();
      const fullCode = preamble ? preamble + code : code;

      executorRef.current.execute(
        fullCode,
        {
          onOutput: (block) => { if (!resolved) outputs.push(block as OutputBlock); },
          onVariables: (vars) => {
            if (resolved) return;
            variables = vars as VariableInfo[];
            if (vars.length) setHasSessionState(true);
          },
          onDone: (success, _executionTime) => { clearTimeout(safetyTimer); finish(success); },
          onStatus: (msg) => { if (!resolved) outputs.push({ type: 'stdout', text: msg + '\n' }); },
        },
        { sessionId, forceBackend }
      );
    });
  }, [sessionId, forceBackend]);

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
    // Reset backend session
    try {
      await fetch(`${HTTP}/api/code/session/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch { /* ignore */ }
    // Reset Pyodide runtime
    executorRef.current.reset();
    setExchanges([]);
    setSessionId(`code-${Date.now()}`);
    setHasSessionState(false);
    setPythonFiles([{ name: DEFAULT_FILE, content: '' }]);
    setActiveFileName(DEFAULT_FILE);
    setShowFileTabs(false);
  }, [sessionId]);

  const stopExecution = useCallback(() => {
    if (generateAbortRef.current) generateAbortRef.current.abort();
    if (executeAbortRef.current) executeAbortRef.current.abort();
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <OfflineBanner hint="Browser Python available" />
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
        {/* Pyodide / Backend toggle */}
        <button
          onClick={() => setForceBackend(!forceBackend)}
          title={forceBackend ? 'Using server — click for browser Python' : 'Using browser Python — click for server'}
          aria-label={forceBackend ? 'Switch to browser execution' : 'Switch to server execution'}
          style={{
            background: forceBackend ? 'rgba(245,158,11,0.1)' : 'rgba(16,185,129,0.1)',
            border: `1px solid ${forceBackend ? 'rgba(245,158,11,0.3)' : 'rgba(16,185,129,0.3)'}`,
            borderRadius: 'var(--r-pill)', color: forceBackend ? '#f59e0b' : '#10b981',
            padding: '1px 7px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
            fontSize: '9.5px', fontFamily: 'inherit', fontWeight: 500,
          }}
        >
          {forceBackend ? <Server size={10} /> : <Globe size={10} />}
          {forceBackend ? 'Server' : 'Browser'}
        </button>
        {pyodideState === 'loading' && (
          <span style={{ fontSize: '9px', color: 'var(--pl)', fontStyle: 'italic' }}>Loading Python...</span>
        )}
        {pyodideState === 'error' && (
          <span style={{ fontSize: '9px', color: 'var(--rd)' }}>Python unavailable</span>
        )}
        <button
          onClick={() => { setShowFileTabs(!showFileTabs); if (!showFileTabs && pythonFiles.length === 1) addPythonFile('utils.py'); }}
          title={showFileTabs ? 'Hide file tabs' : 'Multi-file mode'}
          style={{
            background: showFileTabs ? 'rgba(124,58,237,0.15)' : 'var(--s2)',
            border: `1px solid ${showFileTabs ? 'rgba(124,58,237,0.3)' : 'var(--b1)'}`,
            borderRadius: 'var(--r-pill)', color: showFileTabs ? 'var(--pl)' : 'var(--mu)',
            padding: '1px 7px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 3,
            fontSize: '9.5px', fontFamily: 'inherit', fontWeight: 500,
          }}
        >
          <FileCode2 size={10} />
          Files{pythonFiles.length > 1 ? ` (${pythonFiles.length})` : ''}
        </button>
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

      {/* Multi-file tabs */}
      {showFileTabs && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 0, borderBottom: '1px solid var(--b1)',
          padding: '0 8px', overflow: 'auto', flexShrink: 0,
        }}>
          {pythonFiles.map(f => (
            <div
              key={f.name}
              style={{
                display: 'flex', alignItems: 'center', gap: 4,
                padding: '5px 10px', fontSize: '10.5px', cursor: 'pointer',
                color: f.name === activeFileName ? 'var(--tx)' : 'var(--mu)',
                background: f.name === activeFileName ? 'rgba(124,58,237,0.1)' : 'transparent',
                borderBottom: f.name === activeFileName ? '2px solid var(--p)' : '2px solid transparent',
              }}
              onClick={() => setActiveFileName(f.name)}
            >
              <FileCode2 size={11} />
              <span>{f.name}</span>
              {pythonFiles.length > 1 && (
                <button
                  onClick={(e) => { e.stopPropagation(); removePythonFile(f.name); }}
                  style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer', padding: 0, display: 'flex', marginLeft: 2 }}
                  title="Close file"
                >
                  <X size={10} />
                </button>
              )}
            </div>
          ))}
          <button
            onClick={() => {
              const name = prompt('File name:', 'utils.py');
              if (name) addPythonFile(name);
            }}
            style={{
              background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer',
              padding: '5px 8px', display: 'flex', alignItems: 'center',
            }}
            title="New Python file"
          >
            <FilePlus size={12} />
          </button>
        </div>
      )}

      {/* File editor — shows when a non-main file is active and has content or is being edited */}
      {showFileTabs && activeFileName !== DEFAULT_FILE && (
        <div style={{
          borderBottom: '1px solid var(--b1)', flexShrink: 0,
          height: Math.max(80, Math.min(200, (activeFile.content.split('\n').length + 1) * 20 + 20)),
        }}>
          <CodeEditor
            code={activeFile.content}
            language="python"
            onChange={(val) => updateFileContent(activeFileName, val)}
          />
        </div>
      )}

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
                  <div style={{ height: `${Math.max(3, Math.min(20, ex.code.split('\n').length)) * 22 + 24}px` }}>
                    <CodeEditor
                      code={ex.code}
                      language="python"
                      readOnly
                    />
                  </div>
                )}

                {ex.editing && (
                  <div style={{ padding: '8px' }}>
                    <div style={{ height: `${Math.max(3, Math.min(20, ex.editCode.split('\n').length)) * 22 + 24}px`, border: '1px solid #30363d', borderRadius: 'var(--r-sm)', overflow: 'hidden' }}>
                      <CodeEditor
                        code={ex.editCode}
                        language="python"
                        onChange={(nextCode) => updateExchange(ex.id, { editCode: nextCode })}
                      />
                    </div>
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
