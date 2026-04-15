/**
 * BanditPanel — live telemetry for the strategy bandit.
 *
 * Shows Beta-distribution arms per category with mean reward, alpha/beta, pulls.
 * Read-only; polls /api/bandit/state every 10s.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { BarChart2, RefreshCw } from 'lucide-react';
import { bandit } from '../api/client';
import type { BanditState, BanditArm } from '../api/types';

const CATEGORY_COLORS: Record<string, string> = {
  reasoning: '#a78bfa',
  retrieval: '#60a5fa',
  generation: '#f472b6',
  planning: '#34d399',
  default: '#fbbf24',
};

function categoryColor(name: string): string {
  return CATEGORY_COLORS[name.toLowerCase()] ?? CATEGORY_COLORS.default;
}

function ArmRow({ arm, color }: { arm: BanditArm; color: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, arm.mean_reward)) * 100);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0' }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
          <span style={{ fontSize: 11, color: 'var(--tx)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {arm.strategy}
          </span>
          <span style={{ fontSize: 10, color, fontWeight: 600, marginLeft: 8, flexShrink: 0 }}>
            {pct}%
          </span>
        </div>
        <div style={{ height: 4, borderRadius: 2, background: 'var(--b1)', overflow: 'hidden' }}>
          <div
            style={{
              height: '100%',
              width: `${pct}%`,
              background: color,
              transition: 'width 0.6s ease',
            }}
          />
        </div>
        <div style={{ display: 'flex', gap: 10, marginTop: 3, fontSize: 9, color: 'var(--mu)' }}>
          <span>α {arm.alpha.toFixed(1)}</span>
          <span>β {arm.beta.toFixed(1)}</span>
          <span>{arm.total_pulls} pulls</span>
        </div>
      </div>
    </div>
  );
}

export default function BanditPanel() {
  const [state, setState] = useState<BanditState | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const fetchState = useCallback(async () => {
    try {
      const s = await bandit.state();
      setState(s);
      setErr(null);
    } catch (e: any) {
      setErr(e?.message || 'Failed to load bandit state');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchState();
    const id = setInterval(fetchState, 10_000);
    return () => clearInterval(id);
  }, [fetchState]);

  return (
    <div className="panel-scroll-root" style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 10, height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <BarChart2 size={14} style={{ color: 'var(--p)' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx)' }}>Strategy Bandit</span>
        </div>
        <button
          onClick={fetchState}
          aria-label="Refresh"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--mu)', padding: 4 }}
        >
          <RefreshCw size={12} />
        </button>
      </div>

      {state?.summary && (
        <div style={{ display: 'flex', gap: 10, fontSize: 10, color: 'var(--mu)' }}>
          <span><b style={{ color: 'var(--tx)' }}>{state.summary.total_arms}</b> arms</span>
          <span><b style={{ color: 'var(--tx)' }}>{state.summary.total_outcomes}</b> outcomes</span>
        </div>
      )}

      {loading && !state && (
        <div style={{ color: 'var(--mu)', fontSize: 11 }}>Loading…</div>
      )}

      {err && !state && (
        <div style={{ color: '#f87171', fontSize: 11, padding: 8, background: 'rgba(248, 113, 113, 0.1)', borderRadius: 6 }}>
          {err}
        </div>
      )}

      {state && Object.entries(state.categories ?? {}).map(([catName, arms]) => {
        const color = categoryColor(catName);
        const sorted = [...arms].sort((a, b) => b.mean_reward - a.mean_reward);
        return (
          <div key={catName} style={{ background: 'var(--s2)', borderRadius: 10, padding: '10px 12px', border: '1px solid var(--b1)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
              <span style={{ fontSize: 10, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--tx)' }}>
                {catName}
              </span>
              <span style={{ fontSize: 9, color: 'var(--mu)', marginLeft: 'auto' }}>
                {sorted.length} arms
              </span>
            </div>
            {sorted.map((arm) => (
              <ArmRow key={arm.strategy} arm={arm} color={color} />
            ))}
          </div>
        );
      })}

      {state && Object.keys(state.categories ?? {}).length === 0 && (
        <div style={{ color: 'var(--mu)', fontSize: 11, textAlign: 'center', padding: 16 }}>
          No bandit arms yet. Strategies appear as Aura runs.
        </div>
      )}
    </div>
  );
}
