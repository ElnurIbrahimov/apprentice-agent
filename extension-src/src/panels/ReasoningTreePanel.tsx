/**
 * ReasoningTreePanel — front-end for the MCTS-style deep reasoning tool.
 *
 * Modes:
 *  - Think: POST /api/reasoning-tree/think { problem, context, max_iterations, max_depth }
 *  - Explore: POST /api/reasoning-tree/explore { question, num_options }
 *  - Sessions: GET /sessions to list recent runs, click to load tree viz
 *
 * Tree is rendered as a collapsible nested list (not a graph lib — stays lightweight).
 */

import React, { useCallback, useEffect, useState } from 'react';
import { GitBranch, Play, Loader2, ChevronRight, ChevronDown, List, Compass } from 'lucide-react';
import { reasoning } from '../api/client';
import type { ReasoningResponse, TreeNode, ReasoningSessionListItem } from '../api/types';

type Mode = 'think' | 'explore';

export default function ReasoningTreePanel() {
  const [mode, setMode] = useState<Mode>('think');
  const [problem, setProblem] = useState('');
  const [context, setContext] = useState('');
  const [maxIter, setMaxIter] = useState(20);
  const [maxDepth, setMaxDepth] = useState(6);
  const [numOptions, setNumOptions] = useState(4);

  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<ReasoningResponse | null>(null);
  const [tree, setTree] = useState<TreeNode | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [showSessions, setShowSessions] = useState(false);
  const [sessions, setSessions] = useState<ReasoningSessionListItem[]>([]);

  const loadSessions = useCallback(async () => {
    try {
      const r = await reasoning.sessions();
      setSessions(r.sessions ?? []);
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    if (showSessions) loadSessions();
  }, [showSessions, loadSessions]);

  const run = useCallback(async () => {
    if (!problem.trim()) return;
    setRunning(true);
    setErr(null);
    setResponse(null);
    setTree(null);
    try {
      const resp = mode === 'think'
        ? await reasoning.think(problem.trim(), { context: context.trim() || undefined, max_iterations: maxIter, max_depth: maxDepth })
        : await reasoning.explore(problem.trim(), { num_options: numOptions, context: context.trim() || undefined });
      setResponse(resp);
      if (resp.session_id) {
        try {
          const viz = await reasoning.tree(resp.session_id);
          if (viz.success && viz.tree) setTree(viz.tree);
        } catch { /* silent, tree is optional */ }
      }
    } catch (e: any) {
      setErr(e?.message || 'Reasoning failed');
    }
    setRunning(false);
  }, [mode, problem, context, maxIter, maxDepth, numOptions]);

  const loadSession = useCallback(async (sid: string) => {
    setRunning(true);
    setErr(null);
    try {
      const viz = await reasoning.tree(sid);
      if (viz.success && viz.tree) setTree(viz.tree);
      setShowSessions(false);
    } catch (e: any) {
      setErr(e?.message || 'Could not load session');
    }
    setRunning(false);
  }, []);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <GitBranch size={14} style={{ color: 'var(--p)' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)' }}>Reasoning Tree</span>
        </div>
        <button
          onClick={() => setShowSessions((s) => !s)}
          style={{ background: 'none', border: '1px solid var(--b1)', borderRadius: 6, padding: '4px 8px', fontSize: 10, color: 'var(--mu)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4 }}
        >
          <List size={11} /> Sessions
        </button>
      </div>

      {showSessions && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10, maxHeight: 160, overflowY: 'auto' }}>
          {sessions.length === 0 && <div style={{ fontSize: 11, color: 'var(--mu)' }}>No sessions yet.</div>}
          {sessions.map((s) => (
            <button
              key={s.session_id}
              onClick={() => loadSession(s.session_id)}
              style={{ display: 'block', width: '100%', textAlign: 'left', background: 'none', border: 'none', color: 'var(--tx)', fontSize: 11, padding: '4px 6px', borderRadius: 4, cursor: 'pointer' }}
            >
              <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {s.problem}
              </div>
              {typeof s.confidence === 'number' && (
                <div style={{ fontSize: 9, color: 'var(--mu)' }}>confidence {(s.confidence * 100).toFixed(0)}%</div>
              )}
            </button>
          ))}
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, padding: 3, background: 'var(--s2)', borderRadius: 8, border: '1px solid var(--b1)' }}>
        <ModeButton label="Think" icon={<GitBranch size={11} />} active={mode === 'think'} onClick={() => setMode('think')} />
        <ModeButton label="Explore" icon={<Compass size={11} />} active={mode === 'explore'} onClick={() => setMode('explore')} />
      </div>

      <textarea
        value={problem}
        onChange={(e) => setProblem(e.target.value)}
        placeholder={mode === 'think' ? 'State the problem…' : 'Decision question (e.g. should I refactor X)…'}
        rows={3}
        style={{
          width: '100%',
          padding: '8px 10px',
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 8,
          color: 'var(--tx)',
          fontSize: 12,
          fontFamily: 'inherit',
          resize: 'vertical',
        }}
      />

      <textarea
        value={context}
        onChange={(e) => setContext(e.target.value)}
        placeholder="Context (optional)"
        rows={2}
        style={{
          width: '100%',
          padding: '8px 10px',
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 8,
          color: 'var(--tx)',
          fontSize: 11,
          fontFamily: 'inherit',
          resize: 'vertical',
        }}
      />

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', fontSize: 10, color: 'var(--mu)' }}>
        {mode === 'think' ? (
          <>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              iter
              <input type="number" min={5} max={100} value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} style={numInput} />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              depth
              <input type="number" min={3} max={20} value={maxDepth} onChange={(e) => setMaxDepth(Number(e.target.value))} style={numInput} />
            </label>
          </>
        ) : (
          <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            options
            <input type="number" min={2} max={10} value={numOptions} onChange={(e) => setNumOptions(Number(e.target.value))} style={numInput} />
          </label>
        )}
        <button
          onClick={run}
          disabled={running || !problem.trim()}
          style={{
            marginLeft: 'auto',
            padding: '6px 14px',
            background: running ? 'var(--s2)' : 'var(--p)',
            border: 'none',
            borderRadius: 6,
            color: '#fff',
            fontSize: 11,
            fontWeight: 600,
            cursor: running ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          {running ? <Loader2 size={12} className="spin" /> : <Play size={12} />}
          {running ? 'Thinking…' : 'Run'}
        </button>
      </div>

      {err && (
        <div style={{ color: '#f87171', fontSize: 11, padding: 8, background: 'rgba(248, 113, 113, 0.1)', borderRadius: 6 }}>
          {err}
        </div>
      )}

      {response && response.success && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)' }}>
              Answer
            </span>
            {typeof response.confidence === 'number' && (
              <span style={{ fontSize: 10, color: 'var(--p)', fontWeight: 600 }}>
                {(response.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div style={{ fontSize: 12, color: 'var(--tx)', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
            {response.answer || response.summary || '(no answer)'}
          </div>
          {response.reasoning_steps && response.reasoning_steps.length > 0 && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ fontSize: 10, color: 'var(--mu)', cursor: 'pointer' }}>Reasoning steps ({response.reasoning_steps.length})</summary>
              <ol style={{ fontSize: 11, color: 'var(--tx)', marginTop: 6, paddingLeft: 16, lineHeight: 1.4 }}>
                {response.reasoning_steps.map((s) => (
                  <li key={s.step} style={{ marginBottom: 4 }}>{s.thought}</li>
                ))}
              </ol>
            </details>
          )}
        </div>
      )}

      {tree && (
        <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 12 }}>
          <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 8 }}>
            Tree
          </div>
          <TreeNodeView node={tree} depth={0} />
        </div>
      )}
    </div>
  );
}

// Memoized recursive node — toggling one node doesn't re-render siblings.
const TreeNodeView = React.memo(function TreeNodeView({ node, depth }: { node: TreeNode; depth: number }) {
  const [open, setOpen] = useState(depth < 2);
  const hasChildren = (node.children?.length ?? 0) > 0;
  return (
    <div style={{ marginLeft: depth === 0 ? 0 : 10 }}>
      <div
        onClick={() => hasChildren && setOpen((o) => !o)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          padding: '3px 0',
          cursor: hasChildren ? 'pointer' : 'default',
          fontSize: 11,
          color: 'var(--tx)',
        }}
      >
        {hasChildren ? (
          open ? <ChevronDown size={11} style={{ color: 'var(--mu)' }} /> : <ChevronRight size={11} style={{ color: 'var(--mu)' }} />
        ) : (
          <span style={{ width: 11 }} />
        )}
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {node.thought || node.id}
        </span>
        {typeof node.score === 'number' && (
          <span style={{ fontSize: 9, color: 'var(--p)' }}>{node.score.toFixed(2)}</span>
        )}
      </div>
      {open && hasChildren && node.children!.map((c) => (
        <TreeNodeView key={c.id} node={c} depth={depth + 1} />
      ))}
    </div>
  );
});

function ModeButton({ label, icon, active, onClick }: { label: string; icon: React.ReactNode; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        padding: '5px 8px',
        border: 'none',
        borderRadius: 6,
        background: active ? 'var(--p)' : 'transparent',
        color: active ? '#fff' : 'var(--mu)',
        fontSize: 10,
        fontWeight: 600,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
      }}
    >
      {icon}
      {label}
    </button>
  );
}

const numInput: React.CSSProperties = {
  width: 48,
  padding: '3px 6px',
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 4,
  color: 'var(--tx)',
  fontSize: 10,
};
