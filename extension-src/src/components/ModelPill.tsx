import React, { useRef, useState, useEffect, useMemo } from 'react';
import { ChevronDown, Check, Search } from 'lucide-react';
import { useStore } from '../store';

/* ── Tier classification ── */
interface TierDef {
  id: string;
  label: string;
  color: string;
  badge: string;
}

const TIERS: TierDef[] = [
  { id: 'fast', label: 'Fast', color: '#10b981', badge: 'fast' },
  { id: 'standard', label: 'Standard', color: '#a78bfa', badge: 'std' },
  { id: 'power', label: 'Power', color: '#f59e0b', badge: 'power' },
  { id: 'chatgpt', label: 'ChatGPT', color: '#74aa9c', badge: 'chatgpt' },
  { id: 'api', label: 'API Providers', color: '#ec4899', badge: 'api' },
];

const FAST_PATTERNS = [
  'nemotron', 'glm-5',
];
const POWER_PATTERNS = [
  '397b', '480b', '120b', 'qwen3.5', 'qwen3-coder', 'gpt-oss',
  'kimi-k2', 'minimax-m2.7', 'minimax-m2.5',
];

function classifyModel(name: string, source: 'cloud' | 'local' | 'chatgpt'): string {
  if (source === 'chatgpt') return 'chatgpt';
  const lower = name.toLowerCase();
  if (FAST_PATTERNS.some(p => lower.includes(p))) return 'fast';
  if (POWER_PATTERNS.some(p => lower.includes(p))) return 'power';
  return 'standard';
}

function getBadgeLabel(source: string): string {
  if (source === 'chatgpt') return 'chatgpt';
  if (source === 'cloud') return 'cloud';
  if (source === 'local') return 'local';
  return source; // provider name for API models (e.g. 'anthropic')
}

function getBadgeColor(source: string): string {
  if (source === 'chatgpt') return '#74aa9c';
  if (source === 'cloud') return '#60a5fa';
  if (source === 'local') return '#a78bfa';
  return '#ec4899'; // pink for API provider models
}

interface ModelEntry {
  name: string;
  displayName: string;
  source: string;
  tier: string;
}

/* ── Component ── */
interface Props {
  featureKey: string;
}

export default function ModelPill({ featureKey }: Props) {
  const { featureModels, setModel, mdlCloudList, mdlLocalList, mdlChatgptList, mdlDirectList, loadModels } = useStore();
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const btnRef = useRef<HTMLButtonElement>(null);
  const dropRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const current = featureModels[featureKey] || null;
  const displayName = current
    ? current.replace(/:cloud$/, '').replace(/^chatgpt:/, '').replace(/^[^:]+:/, '')
    : 'Auto';

  /* Build tiered model list */
  const allModels = useMemo<ModelEntry[]>(() => {
    const entries: ModelEntry[] = [];
    for (const m of mdlChatgptList) {
      entries.push({ name: m, displayName: m.replace(/^chatgpt:/, ''), source: 'chatgpt', tier: 'chatgpt' });
    }
    for (const m of (mdlDirectList || [])) {
      const provider = m.split(':')[0] || 'api';
      entries.push({ name: m, displayName: m.replace(/^[^:]+:/, ''), source: provider, tier: 'api' });
    }
    for (const m of mdlCloudList) {
      entries.push({ name: m, displayName: m.replace(/:cloud$/, ''), source: 'cloud', tier: classifyModel(m, 'cloud') });
    }
    for (const m of mdlLocalList) {
      entries.push({ name: m, displayName: m, source: 'local', tier: classifyModel(m, 'local') });
    }
    return entries;
  }, [mdlCloudList, mdlLocalList, mdlChatgptList, mdlDirectList]);

  const showSearch = allModels.length > 10;

  const filtered = useMemo(() => {
    if (!search) return allModels;
    const q = search.toLowerCase();
    return allModels.filter(m => m.displayName.toLowerCase().includes(q) || m.source.includes(q));
  }, [allModels, search]);

  const grouped = useMemo(() => {
    const map = new Map<string, ModelEntry[]>();
    for (const tier of TIERS) map.set(tier.id, []);
    for (const m of filtered) {
      const arr = map.get(m.tier);
      if (arr) arr.push(m);
    }
    return TIERS.filter(t => (map.get(t.id)?.length || 0) > 0).map(t => ({
      tier: t,
      models: map.get(t.id)!,
    }));
  }, [filtered]);

  /* Open/close */
  const toggle = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (open) { setOpen(false); setSearch(''); return; }
    // If lists are empty, force load. Otherwise open immediately (lists refresh in background).
    if (!mdlCloudList.length && !mdlLocalList.length && !mdlChatgptList.length) {
      setLoading(true);
      await loadModels();
      setLoading(false);
    }
    setOpen(true);
  };

  /* Focus search on open */
  useEffect(() => {
    if (open && searchRef.current) {
      setTimeout(() => searchRef.current?.focus(), 60);
    }
  }, [open]);

  /* Position dropdown + click-outside */
  useEffect(() => {
    if (!open || !btnRef.current || !dropRef.current) return;
    const pRect = btnRef.current.getBoundingClientRect();
    const drop = dropRef.current;
    const sidebarW = window.innerWidth;
    // Fill sidebar width with 4px margins on each side
    drop.style.left = '4px';
    drop.style.right = '4px';
    drop.style.top = (pRect.bottom + 6) + 'px';
    drop.style.bottom = 'auto';
    drop.style.width = 'auto'; // let left+right control width
    // Flip upward if needed
    requestAnimationFrame(() => {
      const dropRect = drop.getBoundingClientRect();
      if (window.innerHeight - pRect.bottom < dropRect.height + 16) {
        drop.style.top = 'auto';
        drop.style.bottom = (window.innerHeight - pRect.top + 6) + 'px';
      }
    });

    const outside = (e: MouseEvent) => {
      if (!dropRef.current?.contains(e.target as Node) && !btnRef.current?.contains(e.target as Node)) {
        setOpen(false);
        setSearch('');
      }
    };
    document.addEventListener('mousedown', outside);
    return () => document.removeEventListener('mousedown', outside);
  }, [open]);

  const pick = (model: string | null) => {
    setModel(featureKey, model);
    setOpen(false);
    setSearch('');
  };

  return (
    <span style={{ position: 'relative', display: 'inline-block', verticalAlign: 'middle' }}>
      <button
        ref={btnRef}
        onClick={toggle}
        className="flex items-center gap-1 transition-all duration-150"
        style={{
          background: current ? 'var(--pg)' : 'var(--s2)',
          border: `1px solid ${current ? 'rgba(124,58,237,0.25)' : 'var(--b2)'}`,
          borderRadius: 'var(--r-pill)',
          color: current ? 'var(--pl)' : 'var(--tx)',
          fontSize: '12px',
          fontWeight: 500,
          cursor: 'pointer',
          fontFamily: 'inherit',
          padding: '4px 10px',
          gap: 5,
        }}
      >
        <span
          className="w-[6px] h-[6px] rounded-full flex-shrink-0"
          style={{ background: current ? 'var(--pl)' : 'var(--di)' }}
        />
        <span style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {displayName}
        </span>
        <ChevronDown
          size={11}
          style={{
            opacity: 0.7,
            marginLeft: 2,
            transform: open ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
          }}
        />
      </button>

      {open && (
        <div
          ref={dropRef}
          style={{
            position: 'fixed',
            maxHeight: 320,
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
          }}
        >
          {/* Search input */}
          {showSearch && (
            <div style={{ padding: '6px 8px 4px', position: 'sticky', top: 0, zIndex: 1, background: 'var(--glass)' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                background: 'var(--s3)', border: '1px solid var(--b1)',
                borderRadius: 'var(--r-sm)', padding: '4px 8px',
              }}>
                <Search size={11} style={{ color: 'var(--mu)', flexShrink: 0 }} />
                <input
                  ref={searchRef}
                  type="text"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Filter models..."
                  onClick={e => e.stopPropagation()}
                  style={{
                    background: 'transparent', border: 'none', outline: 'none',
                    color: 'var(--tx)', fontSize: '11px', fontFamily: 'inherit',
                    width: '100%', padding: 0,
                  }}
                />
              </div>
            </div>
          )}

          {/* Auto option */}
          <div
            onClick={() => pick(null)}
            style={{
              padding: '7px 12px',
              fontSize: '12px',
              cursor: 'pointer',
              color: !current ? 'var(--pl)' : 'var(--tx)',
              fontWeight: !current ? 500 : 400,
              display: 'flex', alignItems: 'center', gap: 8,
              transition: 'background 0.1s',
            }}
            onMouseEnter={e => (e.currentTarget.style.background = 'var(--b1)')}
            onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
          >
            {!current && <Check size={12} style={{ color: 'var(--pl)' }} />}
            <span style={{ marginLeft: current ? 20 : 0 }}>Auto</span>
            <span style={{
              marginLeft: 'auto', fontSize: '9px', padding: '1px 5px',
              borderRadius: 'var(--r-pill)', background: 'var(--pg)',
              color: 'var(--pl)', fontWeight: 500,
            }}>
              smart
            </span>
          </div>

          {/* Separator */}
          <div style={{ height: 1, background: 'var(--b1)', margin: '2px 8px' }} />

          {loading && (
            <div style={{ padding: '16px', textAlign: 'center' }}>
              <div className="dots"><span /><span /><span /></div>
            </div>
          )}

          {!loading && grouped.length === 0 && allModels.length === 0 && (
            <div style={{ padding: '14px 12px', fontSize: '11px', color: 'var(--mu)', textAlign: 'center' }}>
              No models found — is Ollama running?
            </div>
          )}

          {!loading && grouped.length === 0 && allModels.length > 0 && search && (
            <div style={{ padding: '14px 12px', fontSize: '11px', color: 'var(--mu)', textAlign: 'center' }}>
              No matches for "{search}"
            </div>
          )}

          {/* Tiered groups */}
          {!loading && grouped.map(({ tier, models }) => (
            <div key={tier.id}>
              <div style={{
                padding: '8px 12px 3px',
                fontSize: '9.5px',
                fontWeight: 600,
                letterSpacing: '0.07em',
                textTransform: 'uppercase',
                color: tier.color,
                display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{
                  width: 5, height: 5, borderRadius: '50%',
                  background: tier.color, display: 'inline-block',
                }} />
                {tier.label}
                <span style={{ fontSize: '8.5px', color: 'var(--mu)', fontWeight: 400, textTransform: 'none' }}>
                  ({models.length})
                </span>
              </div>
              {models.map(m => {
                const isActive = current === m.name;
                return (
                  <div
                    key={m.name}
                    title={m.name}
                    onClick={() => pick(m.name)}
                    style={{
                      padding: '6px 12px',
                      fontSize: '11.5px',
                      cursor: 'pointer',
                      color: isActive ? 'var(--pl)' : 'var(--tx)',
                      fontWeight: isActive ? 500 : 400,
                      display: 'flex', alignItems: 'center', gap: 8,
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--b1)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    {isActive && <Check size={11} style={{ color: 'var(--pl)', flexShrink: 0 }} />}
                    <span style={{
                      marginLeft: isActive ? 0 : 19,
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                      flex: 1, minWidth: 0,
                    }}>
                      {m.displayName}
                    </span>
                    <span style={{
                      fontSize: '8.5px',
                      padding: '1px 5px',
                      borderRadius: 'var(--r-pill)',
                      background: `${getBadgeColor(m.source)}18`,
                      color: getBadgeColor(m.source),
                      fontWeight: 500,
                      flexShrink: 0,
                      lineHeight: '14px',
                    }}>
                      {getBadgeLabel(m.source)}
                    </span>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      )}
    </span>
  );
}
