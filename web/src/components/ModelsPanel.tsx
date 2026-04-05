import { useState, useEffect, useCallback } from 'react';
import {
  MagnifyingGlassIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { StarIcon as StarSolid } from '@heroicons/react/24/solid';

/* ── Types ── */
interface ModelEntry {
  name: string;
  category: 'chatgpt' | 'cloud' | 'local' | 'direct';
}

interface RoleAssignments {
  fast?: string;
  reason?: string;
  code?: string;
  vision?: string;
}

interface StatusInfo {
  active_model?: string;
  model?: string;
  [key: string]: unknown;
}

const CATEGORY_LABELS: Record<ModelEntry['category'], string> = {
  chatgpt: 'ChatGPT',
  cloud: 'Cloud',
  local: 'Local',
  direct: 'Direct API',
};

const CATEGORY_COLORS: Record<ModelEntry['category'], string> = {
  chatgpt: '#10a37f',
  cloud: '#7c3aed',
  local: '#0ea5e9',
  direct: '#f59e0b',
};

const ROLE_LABELS: Record<keyof RoleAssignments, string> = {
  fast: 'Fast',
  reason: 'Reason',
  code: 'Code',
  vision: 'Vision',
};

/* ── CategoryBadge ── */
function CategoryBadge({ category }: { category: ModelEntry['category'] }) {
  return (
    <span
      className="text-[10px] px-1.5 py-0.5 rounded-full font-medium"
      style={{
        background: CATEGORY_COLORS[category] + '22',
        color: CATEGORY_COLORS[category],
        border: `1px solid ${CATEGORY_COLORS[category]}44`,
      }}
    >
      {CATEGORY_LABELS[category]}
    </span>
  );
}

/* ── RoleBadge ── */
function RoleBadge({ role }: { role: string }) {
  return (
    <span className="text-[10px] px-1.5 py-0.5 rounded-full font-medium bg-chat-accent/10 text-chat-accent border border-chat-accent/20">
      {role}
    </span>
  );
}

/* ── Main Component ── */
export function ModelsPanel() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [activeModel, setActiveModel] = useState<string | null>(null);
  const [roles, setRoles] = useState<RoleAssignments>({});
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [settingDefault, setSettingDefault] = useState<string | null>(null);
  const [successModel, setSuccessModel] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [modRes, statusRes, rolesRes] = await Promise.all([
        fetch('/api/models'),
        fetch('/api/status').catch(() => null),
        fetch('/api/models/roles').catch(() => null),
      ]);

      if (!modRes.ok) throw new Error(`Failed to fetch models (${modRes.status})`);
      const modData = await modRes.json();
      const entries: ModelEntry[] = [
        ...(modData.chatgpt_models || []).map((n: string) => ({ name: n, category: 'chatgpt' as const })),
        ...(modData.cloud_models || []).map((n: string) => ({ name: n, category: 'cloud' as const })),
        ...(modData.local_models || []).map((n: string) => ({ name: n, category: 'local' as const })),
        ...(modData.direct_api_models || []).map((n: string) => ({ name: n, category: 'direct' as const })),
      ];
      setModels(entries);

      if (statusRes?.ok) {
        const statusData: StatusInfo = await statusRes.json();
        setActiveModel(statusData.active_model || statusData.model || null);
      }

      if (rolesRes?.ok) {
        const rolesData = await rolesRes.json();
        setRoles(rolesData);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to load models');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSetDefault = useCallback(async (modelName: string) => {
    setSettingDefault(modelName);
    try {
      const res = await fetch('/api/models/default', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelName }),
      });
      if (res.ok) {
        setActiveModel(modelName);
        setSuccessModel(modelName);
        setTimeout(() => setSuccessModel(null), 2000);
      }
    } catch {
      /* non-fatal */
    } finally {
      setSettingDefault(null);
    }
  }, []);

  /* Derived */
  const filtered = models.filter(m =>
    m.name.toLowerCase().includes(search.toLowerCase()),
  );

  const grouped = (['chatgpt', 'cloud', 'local', 'direct'] as ModelEntry['category'][])
    .map(cat => ({
      category: cat,
      items: filtered.filter(m => m.category === cat),
    }))
    .filter(g => g.items.length > 0);

  /* Role map: model name → role names */
  const roleMap: Record<string, string[]> = {};
  for (const [role, name] of Object.entries(roles)) {
    if (name) {
      if (!roleMap[name]) roleMap[name] = [];
      roleMap[name].push(ROLE_LABELS[role as keyof RoleAssignments] || role);
    }
  }

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--surface-0)' }}>
      {/* Header */}
      <div className="flex-shrink-0 border-b border-chat-border px-4 py-3 space-y-3" style={{ background: 'var(--surface-0)' }}>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-chat-text">Models</h2>
            {!loading && !error && (
              <p className="text-[11px] text-chat-text-secondary mt-0.5">
                {models.length} model{models.length !== 1 ? 's' : ''} available
                {activeModel && (
                  <> &middot; Active: <span className="text-chat-accent">{activeModel.split('/').pop()}</span></>
                )}
              </p>
            )}
          </div>
          <button
            onClick={fetchData}
            disabled={loading}
            title="Refresh"
            className="p-1.5 rounded-lg border border-chat-border text-chat-text-secondary hover:text-chat-text disabled:opacity-40 transition-colors"
            style={{ background: 'var(--surface-2)' }}
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <MagnifyingGlassIcon className="w-4 h-4 absolute left-2.5 top-1/2 -translate-y-1/2 text-chat-text-secondary pointer-events-none" />
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter models…"
            className="w-full pl-8 pr-3 py-2 rounded-lg border border-chat-border text-chat-text text-sm outline-none focus:border-chat-accent placeholder-chat-text-secondary/50"
            style={{ background: 'var(--surface-2)' }}
          />
        </div>

        {/* Role summary */}
        {Object.keys(roles).length > 0 && (
          <div className="flex flex-wrap gap-2">
            {(Object.entries(roles) as [keyof RoleAssignments, string | undefined][]).map(([role, name]) => name ? (
              <div key={role} className="flex items-center gap-1.5 text-[11px]">
                <span className="text-chat-text-secondary capitalize">{ROLE_LABELS[role]}:</span>
                <span className="text-chat-text font-medium truncate max-w-[120px]">{name.split('/').pop()}</span>
              </div>
            ) : null)}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {loading && (
          <div className="flex items-center justify-center h-32">
            <div className="animate-pulse flex flex-col gap-3 w-64">
              <div className="h-3 bg-surface-2 rounded w-3/4" />
              <div className="h-3 bg-surface-2 rounded w-1/2" />
              <div className="h-3 bg-surface-2 rounded w-5/6" />
            </div>
          </div>
        )}

        {error && (
          <div className="text-sm text-red-400 p-4 rounded-lg border border-red-400/20" style={{ background: 'rgba(239,68,68,0.05)' }}>
            {error}
          </div>
        )}

        {!loading && !error && filtered.length === 0 && (
          <p className="text-sm text-chat-text-secondary text-center py-8">
            {search ? 'No models match your search.' : 'No models found.'}
          </p>
        )}

        {!loading && grouped.map(({ category, items }) => (
          <div key={category}>
            <div className="flex items-center gap-2 mb-2">
              <CategoryBadge category={category} />
              <span className="text-[11px] text-chat-text-secondary">{items.length}</span>
            </div>
            <div className="space-y-1">
              {items.map(m => {
                const isActive = m.name === activeModel;
                const isSuccess = m.name === successModel;
                const isSetting = m.name === settingDefault;
                const modelRoles = roleMap[m.name] || [];

                return (
                  <div
                    key={m.name}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-colors"
                    style={{
                      background: isActive ? 'var(--surface-3)' : 'var(--surface-1)',
                      borderColor: isActive ? 'var(--chat-accent)' : 'var(--border-default)',
                    }}
                  >
                    {/* Active indicator */}
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-green-400' : 'bg-transparent border border-chat-border'}`} />

                    {/* Name */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span className="text-xs font-medium text-chat-text truncate">{m.name}</span>
                        {modelRoles.map(r => <RoleBadge key={r} role={r} />)}
                      </div>
                    </div>

                    {/* Set as default button */}
                    <button
                      onClick={() => handleSetDefault(m.name)}
                      disabled={isActive || isSetting}
                      title={isActive ? 'Current default' : 'Set as default'}
                      className="flex-shrink-0 p-1 rounded transition-colors disabled:opacity-50"
                    >
                      {isSuccess ? (
                        <CheckCircleIcon className="w-4 h-4 text-green-400" />
                      ) : isActive ? (
                        <StarSolid className="w-4 h-4 text-chat-accent" />
                      ) : isSetting ? (
                        <ArrowPathIcon className="w-4 h-4 text-chat-text-secondary animate-spin" />
                      ) : (
                        <StarIcon className="w-4 h-4 text-chat-text-secondary hover:text-chat-accent transition-colors" />
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
