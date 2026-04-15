/**
 * EvolutionPanel — GEPA skill evolution + self-improvement control.
 *
 * Tabs:
 *  - Evolve: POST /api/evolution/preview | run, GET /api/evolution/status
 *  - Self-improve: GET /self-improvement/{status, report, params}; POST /cycle, /tune
 */

import React, { useCallback, useEffect, useState } from 'react';
import { Dna, Play, Eye, Loader2, Gauge, RefreshCw } from 'lucide-react';
import { evolution, selfImprovement } from '../api/client';
import type { EvolutionStatusResponse, SelfImprovementStatus, SelfImprovementParams } from '../api/types';

type Tab = 'evolve' | 'self';

export default function EvolutionPanel() {
  const [tab, setTab] = useState<Tab>('evolve');

  // Evolve state
  const [skillIds, setSkillIds] = useState('');
  const [maxIter, setMaxIter] = useState(10);
  const [dryRun, setDryRun] = useState(true);
  const [timeout, setTimeoutSec] = useState(600);
  const [status, setStatus] = useState<EvolutionStatusResponse | null>(null);
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Self-improve state
  const [siStatus, setSiStatus] = useState<SelfImprovementStatus | null>(null);
  const [siParams, setSiParams] = useState<SelfImprovementParams | null>(null);
  const [siBusy, setSiBusy] = useState(false);

  const loadStatus = useCallback(async () => {
    try {
      setStatus(await evolution.status());
    } catch { /* silent */ }
  }, []);

  const loadSelf = useCallback(async () => {
    try {
      const [s, p] = await Promise.all([
        selfImprovement.status(),
        selfImprovement.params(),
      ]);
      setSiStatus(s);
      setSiParams(p);
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    if (tab === 'evolve') {
      loadStatus();
      // Poll fast (5s) only while an evolution run is in-flight; otherwise
      // back off to 15s. Stop polling entirely on terminal states; a final
      // manual refresh is enough.
      const currentStatus = status?.status;
      if (currentStatus === 'complete' || currentStatus === 'error') {
        return; // no interval — user can click refresh
      }
      const interval = currentStatus === 'running' ? 5000 : 15000;
      const id = setInterval(loadStatus, interval);
      return () => clearInterval(id);
    } else {
      loadSelf();
    }
  }, [tab, loadStatus, loadSelf, status?.status]);

  const buildBody = () => ({
    skill_ids: skillIds.trim() ? skillIds.split(',').map((s) => s.trim()).filter(Boolean) : undefined,
    max_iterations: maxIter,
    dry_run: dryRun,
    timeout_seconds: timeout,
  });

  const runPreview = useCallback(async () => {
    setBusy(true);
    setErr(null);
    setPreview(null);
    try {
      const r = await evolution.preview(buildBody());
      setPreview(r.preview);
    } catch (e: any) {
      setErr(e?.message || 'Preview failed');
    }
    setBusy(false);
  }, [skillIds, maxIter, dryRun, timeout]);

  const runEvolve = useCallback(async () => {
    setBusy(true);
    setErr(null);
    try {
      await evolution.run(buildBody());
      setPreview(null);
      loadStatus();
    } catch (e: any) {
      setErr(e?.message || 'Evolve failed');
    }
    setBusy(false);
  }, [skillIds, maxIter, dryRun, timeout, loadStatus]);

  const runCycle = useCallback(async () => {
    setSiBusy(true);
    try {
      await selfImprovement.cycle();
      await loadSelf();
    } catch { /* silent */ }
    setSiBusy(false);
  }, [loadSelf]);

  const tuneParam = useCallback(async (name: string, value: number) => {
    try {
      await selfImprovement.tune(name, value);
      await loadSelf();
    } catch { /* silent */ }
  }, [loadSelf]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Dna size={14} style={{ color: 'var(--p)' }} />
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)', flex: 1 }}>Evolution</span>
        <button onClick={() => (tab === 'evolve' ? loadStatus() : loadSelf())} style={{ background: 'none', border: 'none', color: 'var(--mu)', cursor: 'pointer' }}>
          <RefreshCw size={12} />
        </button>
      </div>

      <div style={{ display: 'flex', gap: 4, padding: 3, background: 'var(--s2)', borderRadius: 8, border: '1px solid var(--b1)' }}>
        <TabButton label="Evolve (GEPA)" icon={<Dna size={11} />} active={tab === 'evolve'} onClick={() => setTab('evolve')} />
        <TabButton label="Self-improve" icon={<Gauge size={11} />} active={tab === 'self'} onClick={() => setTab('self')} />
      </div>

      {tab === 'evolve' && (
        <>
          <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <input
              value={skillIds}
              onChange={(e) => setSkillIds(e.target.value)}
              placeholder="Skill IDs (comma-separated, blank = all)"
              style={{ padding: '6px 10px', background: 'var(--bg)', border: '1px solid var(--b1)', borderRadius: 6, color: 'var(--tx)', fontSize: 11 }}
            />
            <div style={{ display: 'flex', gap: 8, fontSize: 10, color: 'var(--mu)', alignItems: 'center', flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                iter <input type="number" min={1} max={50} value={maxIter} onChange={(e) => setMaxIter(Number(e.target.value))} style={numInput} />
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                timeout <input type="number" min={60} max={3600} value={timeout} onChange={(e) => setTimeoutSec(Number(e.target.value))} style={numInput} />s
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
                dry run
              </label>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <button onClick={runPreview} disabled={busy} style={btnSecondary}>
                {busy ? <Loader2 size={11} className="spin" /> : <Eye size={11} />} Preview
              </button>
              <button onClick={runEvolve} disabled={busy} style={btnPrimary}>
                {busy ? <Loader2 size={11} className="spin" /> : <Play size={11} />} Run
              </button>
            </div>
          </div>

          {status && (
            <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 4 }}>
                Status
              </div>
              <div style={{ fontSize: 11, color: 'var(--tx)' }}>
                {status.status}{status.run_id ? ` · ${status.run_id.slice(0, 8)}` : ''}
              </div>
              {status.result && (
                <pre style={{ fontSize: 9, color: 'var(--mu)', marginTop: 6, overflow: 'auto', maxHeight: 120 }}>
                  {JSON.stringify(status.result, null, 2)}
                </pre>
              )}
            </div>
          )}

          {preview && (
            <div style={{ background: 'var(--s2)', border: '1px dashed var(--b1)', borderRadius: 10, padding: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 4 }}>
                Preview
              </div>
              <pre style={{ fontSize: 9, color: 'var(--tx)', overflow: 'auto', maxHeight: 160 }}>
                {JSON.stringify(preview, null, 2)}
              </pre>
            </div>
          )}

          {err && (
            <div style={{ color: '#f87171', fontSize: 11, padding: 8, background: 'rgba(248, 113, 113, 0.1)', borderRadius: 6 }}>
              {err}
            </div>
          )}
        </>
      )}

      {tab === 'self' && (
        <>
          <button onClick={runCycle} disabled={siBusy} style={{ ...btnPrimary, alignSelf: 'flex-start' }}>
            {siBusy ? <Loader2 size={11} className="spin" /> : <Play size={11} />} Trigger cycle
          </button>

          {siStatus && (
            <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 4 }}>
                Status
              </div>
              <pre style={{ fontSize: 9, color: 'var(--tx)', overflow: 'auto', maxHeight: 120 }}>
                {JSON.stringify(siStatus, null, 2)}
              </pre>
            </div>
          )}

          {siParams?.params && (
            <div style={{ background: 'var(--s2)', border: '1px solid var(--b1)', borderRadius: 10, padding: 10 }}>
              <div style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--mu)', marginBottom: 6 }}>
                Tunable parameters
              </div>
              {Object.entries(siParams.params).map(([name, value]) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', fontSize: 11 }}>
                  <span style={{ flex: 1, color: 'var(--tx)' }}>{name}</span>
                  {typeof value === 'number' ? (
                    <input
                      type="number"
                      defaultValue={value}
                      onBlur={(e) => {
                        const v = Number(e.target.value);
                        if (!Number.isNaN(v) && v !== value) tuneParam(name, v);
                      }}
                      style={{ ...numInput, width: 80 }}
                    />
                  ) : (
                    <span style={{ color: 'var(--mu)' }}>{String(value)}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function TabButton({ label, icon, active, onClick }: { label: string; icon: React.ReactNode; active: boolean; onClick: () => void }) {
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
  width: 60,
  padding: '3px 6px',
  background: 'var(--bg)',
  border: '1px solid var(--b1)',
  borderRadius: 4,
  color: 'var(--tx)',
  fontSize: 10,
};

const btnPrimary: React.CSSProperties = {
  padding: '6px 14px',
  background: 'var(--p)',
  border: 'none',
  borderRadius: 6,
  color: '#fff',
  fontSize: 11,
  fontWeight: 600,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
};

const btnSecondary: React.CSSProperties = {
  padding: '6px 14px',
  background: 'var(--s2)',
  border: '1px solid var(--b1)',
  borderRadius: 6,
  color: 'var(--tx)',
  fontSize: 11,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  gap: 4,
};
