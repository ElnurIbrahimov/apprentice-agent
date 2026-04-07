import React, { useEffect, useState, useMemo } from 'react';
import { useStore } from '../store';
import { FEATURE_DEFS } from '../types';
import { Check, ChevronDown, Zap, Cpu, Cloud, Bot } from 'lucide-react';

/* ── Tier helpers (same logic as ModelPill) ── */
const FAST_PATTERNS = [
  'nemotron', 'glm-5:',
];
const POWER_PATTERNS = [
  '397b', '480b', '120b', 'qwen3.5', 'qwen3-coder', 'gpt-oss',
  'kimi-k2', 'minimax-m2.7', 'minimax-m2.5', 'glm-5.1', 'gemma4',
];

function classifyModel(name: string, source: 'cloud' | 'local' | 'chatgpt'): string {
  if (source === 'chatgpt') return 'chatgpt';
  const lower = name.toLowerCase();
  if (FAST_PATTERNS.some(p => lower.includes(p))) return 'fast';
  if (POWER_PATTERNS.some(p => lower.includes(p))) return 'power';
  return 'standard';
}

interface ModelEntry {
  name: string;
  displayName: string;
  source: 'cloud' | 'local' | 'chatgpt';
  tier: string;
}

const TIER_META: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  fast: { label: 'Fast', color: '#10b981', icon: <Zap size={13} /> },
  standard: { label: 'Standard', color: '#a78bfa', icon: <Cpu size={13} /> },
  power: { label: 'Power', color: '#f59e0b', icon: <Cloud size={13} /> },
  chatgpt: { label: 'ChatGPT', color: '#74aa9c', icon: <Bot size={13} /> },
};

const TIER_ORDER = ['fast', 'standard', 'power', 'chatgpt'];

function getBadgeColor(source: 'cloud' | 'local' | 'chatgpt'): string {
  if (source === 'chatgpt') return '#74aa9c';
  if (source === 'cloud') return '#60a5fa';
  return '#a78bfa';
}
function getBadgeLabel(source: 'cloud' | 'local' | 'chatgpt'): string {
  if (source === 'chatgpt') return 'chatgpt';
  if (source === 'cloud') return 'cloud';
  return 'local';
}

/* ── Quick-set dropdown ── */
function QuickSetDropdown({ allModels, onSet }: { allModels: ModelEntry[]; onSet: (model: string) => void }) {
  const [open, setOpen] = useState(false);

  if (allModels.length === 0) return null;

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          background: 'var(--pg)',
          border: '1px solid var(--b1)',
          borderRadius: 'var(--r-pill)',
          padding: '4px 10px',
          fontSize: '10.5px',
          color: 'var(--pl)',
          cursor: 'pointer',
          fontFamily: 'inherit',
          fontWeight: 500,
          display: 'flex', alignItems: 'center', gap: 4,
          transition: 'all 0.15s',
        }}
        onMouseEnter={e => { e.currentTarget.style.background = 'var(--pg2)'; }}
        onMouseLeave={e => { e.currentTarget.style.background = 'var(--pg)'; }}
      >
        Set all to...
        <ChevronDown size={10} style={{
          transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
          transition: 'transform 0.2s ease',
        }} />
      </button>
      {open && (
        <>
          {/* Backdrop to close */}
          <div
            style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
            onClick={() => setOpen(false)}
          />
          <div style={{
            position: 'absolute',
            top: '100%', right: 0, marginTop: 4,
            minWidth: 200, maxWidth: 260, maxHeight: 280,
            overflowY: 'auto',
            background: 'var(--glass)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: '1px solid var(--b2)',
            borderRadius: 'var(--r-md)',
            boxShadow: 'var(--sh-lg)',
            zIndex: 9999,
            padding: '4px 0',
            animation: 'mdlDropIn 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
            transformOrigin: 'top right',
          }}>
            {/* Auto option */}
            <div
              onClick={() => { onSet(''); setOpen(false); }}
              style={{
                padding: '6px 12px', fontSize: '11.5px', cursor: 'pointer',
                color: 'var(--tx)', display: 'flex', alignItems: 'center', gap: 6,
                transition: 'background 0.1s',
              }}
              onMouseEnter={e => (e.currentTarget.style.background = 'var(--b1)')}
              onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
            >
              Auto (reset all)
            </div>
            <div style={{ height: 1, background: 'var(--b1)', margin: '2px 8px' }} />
            {allModels.map(m => (
              <div
                key={m.name}
                title={m.name}
                onClick={() => { onSet(m.name); setOpen(false); }}
                style={{
                  padding: '5px 12px', fontSize: '11px', cursor: 'pointer',
                  color: 'var(--tx)', display: 'flex', alignItems: 'center', gap: 6,
                  transition: 'background 0.1s',
                }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--b1)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
              >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                  {m.displayName}
                </span>
                <span style={{
                  fontSize: '8.5px', padding: '1px 5px',
                  borderRadius: 'var(--r-pill)',
                  background: `${getBadgeColor(m.source)}18`,
                  color: getBadgeColor(m.source),
                  fontWeight: 500, flexShrink: 0,
                }}>
                  {getBadgeLabel(m.source)}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ── Main Panel ── */
export default function ModelsPanel() {
  const {
    featureModels, setModel, mdlCloudList, mdlLocalList, mdlChatgptList,
    loadModels: storeLoadModels, activePanel, mdlListsLoaded,
    routingPreference, setRoutingPreference,
  } = useStore();
  const [loading, setLoading] = useState(false);
  const [tab, setTab] = useState<'features' | 'models'>('features');

  useEffect(() => {
    if (activePanel !== 'models') return;
    if (mdlListsLoaded && (mdlCloudList.length || mdlLocalList.length || mdlChatgptList.length)) return;
    setLoading(true);
    storeLoadModels().finally(() => setLoading(false));
  }, [activePanel]);

  /* Build model entries */
  const allModels = useMemo<ModelEntry[]>(() => {
    const entries: ModelEntry[] = [];
    for (const m of mdlChatgptList) {
      entries.push({ name: m, displayName: m.replace(/^chatgpt:/, ''), source: 'chatgpt', tier: 'chatgpt' });
    }
    for (const m of mdlCloudList) {
      entries.push({ name: m, displayName: m.replace(/:cloud$/, ''), source: 'cloud', tier: classifyModel(m, 'cloud') });
    }
    for (const m of mdlLocalList) {
      entries.push({ name: m, displayName: m, source: 'local', tier: classifyModel(m, 'local') });
    }
    return entries;
  }, [mdlCloudList, mdlLocalList, mdlChatgptList]);

  const grouped = useMemo(() => {
    const map = new Map<string, ModelEntry[]>();
    for (const t of TIER_ORDER) map.set(t, []);
    for (const m of allModels) {
      const arr = map.get(m.tier);
      if (arr) arr.push(m);
    }
    return TIER_ORDER.filter(t => (map.get(t)?.length || 0) > 0).map(t => ({
      ...TIER_META[t],
      id: t,
      models: map.get(t)!,
    }));
  }, [allModels]);

  const handleSetAll = (model: string) => {
    if (!model) {
      // Clear all
      for (const def of FEATURE_DEFS) setModel(def.key, null);
    } else {
      for (const def of FEATURE_DEFS) setModel(def.key, model);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div style={{
        padding: '12px 14px 10px',
        borderBottom: '1px solid var(--b1)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--tx)' }}>
            Models
          </div>
          <QuickSetDropdown allModels={allModels} onSet={handleSetAll} />
        </div>

        {/* Preference tier */}
        <div style={{
          display: 'flex', gap: 2,
          background: 'var(--s2)', borderRadius: 'var(--r-sm)',
          padding: 2, border: '1px solid var(--b1)',
          marginBottom: 8,
        }}>
          {([
            { key: 'prefer-fast', label: '\u26A1 Fast', color: '#10b981' },
            { key: 'balanced', label: '\u2696 Balanced', color: '#a78bfa' },
            { key: 'prefer-quality', label: '\uD83D\uDC8E Quality', color: '#f59e0b' },
          ] as const).map(t => (
            <button
              key={t.key}
              onClick={() => setRoutingPreference(t.key as any)}
              style={{
                flex: 1, padding: '5px 0',
                fontSize: '10.5px', fontWeight: 500, fontFamily: 'inherit',
                borderRadius: 'calc(var(--r-sm) - 2px)',
                border: 'none', cursor: 'pointer',
                background: routingPreference === t.key ? 'var(--pg2)' : 'transparent',
                color: routingPreference === t.key ? t.color : 'var(--mu)',
                transition: 'all 0.15s',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab switcher */}
        <div style={{
          display: 'flex', gap: 2,
          background: 'var(--s2)', borderRadius: 'var(--r-sm)',
          padding: 2, border: '1px solid var(--b1)',
        }}>
          {(['features', 'models'] as const).map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                flex: 1, padding: '5px 0',
                fontSize: '11px', fontWeight: 500, fontFamily: 'inherit',
                borderRadius: 'calc(var(--r-sm) - 2px)',
                border: 'none', cursor: 'pointer',
                background: tab === t ? 'var(--pg2)' : 'transparent',
                color: tab === t ? 'var(--pl)' : 'var(--mu)',
                transition: 'all 0.15s',
              }}
            >
              {t === 'features' ? 'Per-Feature' : 'All Models'}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <div className="flex justify-center mt-8">
          <div className="dots"><span /><span /><span /></div>
        </div>
      )}

      {/* ── Per-Feature Tab ── */}
      {!loading && tab === 'features' && (
        <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-2">
          <div style={{ fontSize: '10.5px', color: 'var(--mu)', marginBottom: 4, padding: '0 2px' }}>
            Assign a model to each feature. "Auto" lets the backend pick the best one.
          </div>
          {FEATURE_DEFS.map(def => {
            const current = featureModels[def.key] || '';
            const currentDisplay = current
              ? current.replace(/:cloud$/, '').replace(/^chatgpt:/, '')
              : '';
            return (
              <div
                key={def.key}
                style={{
                  background: 'var(--glass)',
                  backdropFilter: 'blur(12px)',
                  WebkitBackdropFilter: 'blur(12px)',
                  border: '1px solid var(--b1)',
                  borderRadius: 'var(--r-md)',
                  padding: '10px 12px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  transition: 'border-color 0.15s',
                }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--b2)')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--b1)')}
              >
                <span style={{ fontSize: '15px', flexShrink: 0, width: 22, textAlign: 'center' }}>
                  {def.icon}
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: '12px', fontWeight: 500, color: 'var(--tx)' }}>
                    {def.label}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--mu)', lineHeight: '14px' }}>
                    {def.desc}
                  </div>
                </div>
                <select
                  value={current}
                  onChange={e => setModel(def.key, e.target.value || null)}
                  style={{
                    background: 'var(--s3)',
                    border: '1px solid var(--b1)',
                    borderRadius: 'var(--r-sm)',
                    color: current ? 'var(--pl)' : 'var(--mu)',
                    fontSize: '10.5px',
                    padding: '4px 6px',
                    maxWidth: 120,
                    fontFamily: 'inherit',
                    cursor: 'pointer',
                    transition: 'border-color 0.15s',
                  }}
                  onFocus={e => (e.currentTarget.style.borderColor = 'var(--b2)')}
                  onBlur={e => (e.currentTarget.style.borderColor = 'var(--b1)')}
                >
                  <option value="">Auto</option>
                  {grouped.map(g => (
                    <optgroup key={g.id} label={`${g.label} (${g.models.length})`}>
                      {g.models.map(m => (
                        <option key={m.name} value={m.name}>
                          {m.displayName} [{getBadgeLabel(m.source)}]
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
              </div>
            );
          })}

          {allModels.length === 0 && (
            <div style={{ color: 'var(--mu)', fontSize: '11px', textAlign: 'center', marginTop: 20 }}>
              No models found — is Ollama running?
            </div>
          )}
        </div>
      )}

      {/* ── All Models Tab ── */}
      {!loading && tab === 'models' && (
        <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
          {/* Stats bar */}
          <div style={{
            display: 'flex', gap: 8, flexWrap: 'wrap',
            padding: '0 2px', marginBottom: 2,
          }}>
            {grouped.map(g => (
              <span key={g.id} style={{
                fontSize: '10px', display: 'flex', alignItems: 'center', gap: 4,
                color: g.color,
              }}>
                <span style={{ width: 5, height: 5, borderRadius: '50%', background: g.color }} />
                {g.label}: {g.models.length}
              </span>
            ))}
            <span style={{ fontSize: '10px', color: 'var(--mu)' }}>
              Total: {allModels.length}
            </span>
          </div>

          {grouped.map(g => (
            <div key={g.id}>
              {/* Tier header */}
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                marginBottom: 6, padding: '0 2px',
              }}>
                <span style={{ color: g.color, display: 'flex' }}>{g.icon}</span>
                <span style={{
                  fontSize: '11px', fontWeight: 600,
                  color: g.color, letterSpacing: '0.03em',
                }}>
                  {g.label}
                </span>
                <span style={{
                  fontSize: '9.5px', color: 'var(--mu)', fontWeight: 400,
                }}>
                  ({g.models.length})
                </span>
              </div>

              {/* Model cards */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginBottom: 8 }}>
                {g.models.map(m => {
                  // Find which features use this model
                  const assignedFeatures = FEATURE_DEFS.filter(f => featureModels[f.key] === m.name);
                  return (
                    <div
                      key={m.name}
                      style={{
                        background: 'var(--glass)',
                        backdropFilter: 'blur(12px)',
                        WebkitBackdropFilter: 'blur(12px)',
                        border: '1px solid var(--b1)',
                        borderRadius: 'var(--r-sm)',
                        padding: '8px 10px',
                        transition: 'border-color 0.15s',
                      }}
                      onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--b2)')}
                      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--b1)')}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{
                          width: 6, height: 6, borderRadius: '50%',
                          background: g.color, flexShrink: 0,
                        }} />
                        <span style={{
                          fontSize: '12px', fontWeight: 500, color: 'var(--tx)',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          flex: 1, minWidth: 0,
                        }}>
                          {m.displayName}
                        </span>
                        <span style={{
                          fontSize: '8.5px', padding: '1px 6px',
                          borderRadius: 'var(--r-pill)',
                          background: `${getBadgeColor(m.source)}18`,
                          color: getBadgeColor(m.source),
                          fontWeight: 500, flexShrink: 0,
                        }}>
                          {getBadgeLabel(m.source)}
                        </span>
                      </div>

                      {/* Assigned features */}
                      {assignedFeatures.length > 0 && (
                        <div style={{
                          display: 'flex', flexWrap: 'wrap', gap: 4,
                          marginTop: 6, paddingLeft: 14,
                        }}>
                          {assignedFeatures.map(f => (
                            <span key={f.key} style={{
                              fontSize: '9px', padding: '1px 6px',
                              borderRadius: 'var(--r-pill)',
                              background: 'var(--pg)',
                              color: 'var(--pl)',
                              fontWeight: 500,
                              display: 'flex', alignItems: 'center', gap: 3,
                            }}>
                              <Check size={8} />
                              {f.label}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {allModels.length === 0 && (
            <div style={{ color: 'var(--mu)', fontSize: '11px', textAlign: 'center', marginTop: 20 }}>
              No models found — is Ollama running?
            </div>
          )}
        </div>
      )}
    </div>
  );
}
