import React, { useEffect, useRef, useState } from 'react';
import { useStore } from '../store';
import ModelPill from '../components/ModelPill';
import { HTTP, getAuthHeaders } from '../api';

export default function MathPanel() {
  const { getModel } = useStore();
  const [mode, setMode] = useState<'solve' | 'graph' | 'stats'>('solve');
  const [status, setStatus] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Cleanup abort on unmount
  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
    };
  }, []);

  const solve = async () => {
    const problem = inputRef.current?.value.trim();
    if (!problem) return;
    setLoading(true);
    setStatus('Solving…');
    setResult(null);
    if (abortRef.current) abortRef.current.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const resp = await fetch(`${HTTP}/api/math/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({ problem, mode, model: getModel('math') }),
        signal: ctrl.signal,
      });
      if (!resp.ok) {
        const d = await resp.json().catch(() => ({}));
        setStatus('⚠ ' + ((d as any).detail || resp.statusText));
        return;
      }
      const data = await resp.json();
      setResult(data);
      setStatus('');
    } catch (err: any) {
      setStatus('⚠ ' + (err.message || 'Request failed'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden p-3 gap-3">
      {/* Mode */}
      <div className="flex gap-1">
        {(['solve', 'graph', 'stats'] as const).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={{
              flex: 1,
              padding: '7px',
              background: mode === m ? 'var(--pg2)' : 'var(--s2)',
              border: `1px solid ${mode === m ? 'var(--p)' : 'var(--b1)'}`,
              borderRadius: 'var(--r-md)',
              color: mode === m ? 'var(--pl)' : 'var(--mu)',
              fontSize: '11px',
              cursor: 'pointer',
              fontFamily: 'inherit',
              textTransform: 'capitalize',
            }}
          >
            {m === 'solve' ? '= Solve' : m === 'graph' ? '📈 Graph' : '📊 Stats'}
          </button>
        ))}
      </div>

      <textarea
        ref={inputRef}
        placeholder="Enter math problem… (Ctrl+Enter to solve)"
        onKeyDown={e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); solve(); } }}
        style={{
          background: 'var(--s2)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-md)',
          color: 'var(--tx)',
          fontSize: '12.5px',
          padding: '8px 10px',
          resize: 'none',
          height: 90,
          outline: 'none',
          fontFamily: 'monospace',
        }}
      />

      <div className="flex items-center justify-between">
        <ModelPill featureKey="math" />
        <button
          onClick={solve}
          disabled={loading}
          style={{
            background: loading ? 'var(--s3)' : 'var(--p)',
            border: 'none',
            borderRadius: 'var(--r-md)',
            color: 'white',
            padding: '7px 20px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '12px',
            fontFamily: 'inherit',
          }}
        >
          {loading ? '…' : 'Solve'}
        </button>
      </div>

      {status && (
        <div style={{ color: status.startsWith('⚠') ? 'var(--rd)' : 'var(--mu)', fontSize: '12px' }}>{status}</div>
      )}

      {result && (
        <div
          className="flex-1 overflow-y-auto flex flex-col gap-3"
          style={{
            background: 'var(--s1)',
            border: '1px solid var(--b1)',
            borderRadius: 'var(--r-md)',
            padding: '12px',
          }}
        >
          {result.solution && (
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
                Solution
              </div>
              <div style={{ fontSize: '16px', fontWeight: 600, color: 'var(--pl)', fontFamily: 'monospace' }}>
                {result.solution}
              </div>
            </div>
          )}

          {result.latex && (
            <div
              style={{
                background: 'var(--s2)',
                border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)',
                padding: '8px 10px',
                fontFamily: 'monospace',
                fontSize: '12px',
                color: 'var(--mu)',
              }}
            >
              {result.latex}
            </div>
          )}

          {result.steps?.length > 0 && (
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--mu)', marginBottom: 6 }}>
                Steps
              </div>
              <div className="flex flex-col gap-2">
                {result.steps.map((step: string, i: number) => (
                  <div key={i} className="flex gap-2 items-start">
                    <span
                      style={{
                        flexShrink: 0,
                        width: 20,
                        height: 20,
                        borderRadius: '50%',
                        background: 'var(--pg)',
                        border: '1px solid var(--p)',
                        fontSize: '10px',
                        color: 'var(--pl)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      {i + 1}
                    </span>
                    <span style={{ fontSize: '12px', color: 'var(--tx)', flex: 1, paddingTop: 2 }}>{step}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
