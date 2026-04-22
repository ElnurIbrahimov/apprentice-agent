import { useState, useEffect, useRef, useMemo } from 'react';
import type { Tool, VoiceStatus, SessionCosts } from '../types';
import {
  WrenchScrewdriverIcon,
  SpeakerWaveIcon,
  MagnifyingGlassIcon,
  CurrencyDollarIcon,
  PuzzlePieceIcon,
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/apiFetch';

export function ToolsPanel() {
  const [tools, setTools] = useState<Tool[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [voice, setVoice] = useState<VoiceStatus | null>(null);
  const [costs, setCosts] = useState<SessionCosts | null>(null);
  const [, setLoading] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [reloadMsg, setReloadMsg] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [toolsRes, voiceRes, costsRes] = await Promise.all([
        apiFetch('/api/tools'),
        apiFetch('/api/voice'),
        apiFetch('/api/costs/summary'),
      ]);
      if (!mountedRef.current) return;
      if (toolsRes.ok) {
        const data = await toolsRes.json();
        setTools(data.tools || []);
        setCategories(data.categories || []);
      }
      if (voiceRes.ok) setVoice(await voiceRes.json());
      if (costsRes.ok) {
        const c = await costsRes.json();
        if (c.success) setCosts(c);
      }
    } catch {
      if (mountedRef.current) setReloadMsg('Failed to load tools data');
    }
    if (mountedRef.current) setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleReloadPlugins = async () => {
    setReloading(true);
    setReloadMsg(null);
    try {
      const res = await apiFetch('/api/plugins/reload', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        if (data.success) {
          setReloadMsg(`+${data.new_tools} new tool${data.new_tools !== 1 ? 's' : ''} loaded (${data.tools_after} total)`);
          fetchData(); // refresh tool list
        } else {
          setReloadMsg(`Error: ${data.error}`);
        }
      }
    } catch (e) {
      setReloadMsg('Reload failed');
    } finally {
      setReloading(false);
    }
  };

  // Filtered tool list
  const filteredTools = useMemo(() => {
    const q = search.toLowerCase();
    return tools.filter((t) => {
      const matchesSearch = !q || t.name.includes(q) || t.description.toLowerCase().includes(q);
      const matchesCategory = !activeCategory || t.category === activeCategory;
      return matchesSearch && matchesCategory;
    });
  }, [tools, search, activeCategory]);

  const CATEGORY_COLORS: Record<string, string> = {
    Core: 'bg-blue-900/40 text-blue-300 hover:bg-blue-800/50',
    Memory: 'bg-purple-900/40 text-purple-300 hover:bg-purple-800/50',
    Communication: 'bg-green-900/40 text-green-300 hover:bg-green-800/50',
    Productivity: 'bg-yellow-900/40 text-yellow-300 hover:bg-yellow-800/50',
    'Smart Home': 'bg-orange-900/40 text-orange-300 hover:bg-orange-800/50',
    Media: 'bg-pink-900/40 text-pink-300 hover:bg-pink-800/50',
    Development: 'bg-cyan-900/40 text-cyan-300 hover:bg-cyan-800/50',
    AI: 'bg-violet-900/40 text-violet-300 hover:bg-violet-800/50',
    Monitoring: 'bg-emerald-900/40 text-emerald-300 hover:bg-emerald-800/50',
    System: 'bg-slate-700/60 text-slate-300 hover:bg-slate-600/60',
    Analytics: 'bg-teal-900/40 text-teal-300 hover:bg-teal-800/50',
    Other: 'bg-gray-800/40 text-gray-400 hover:bg-gray-700/40',
  };

  const getCategoryColor = (cat: string) =>
    CATEGORY_COLORS[cat] || CATEGORY_COLORS['Other'];

  const getToolBadgeColor = (cat: string) => {
    const map: Record<string, string> = {
      Core: 'bg-blue-900/20 text-blue-400',
      Memory: 'bg-purple-900/20 text-purple-400',
      Communication: 'bg-green-900/20 text-green-400',
      Productivity: 'bg-yellow-900/20 text-yellow-400',
      'Smart Home': 'bg-orange-900/20 text-orange-400',
      Media: 'bg-pink-900/20 text-pink-400',
      Development: 'bg-cyan-900/20 text-cyan-400',
      AI: 'bg-violet-900/20 text-violet-400',
      Monitoring: 'bg-emerald-900/20 text-emerald-400',
      System: 'bg-slate-700/30 text-slate-400',
      Analytics: 'bg-teal-900/20 text-teal-400',
    };
    return map[cat] || 'bg-gray-800/20 text-gray-400';
  };

  return (
    <div className="space-y-4">
      {/* Voice */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2 mb-3">
          <SpeakerWaveIcon className="w-5 h-5 text-orange-400" />
          Voice / TTS
        </h3>

        {voice?.available ? (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Engine</span>
              <span className="text-chat-text">{voice.engine}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-chat-text-secondary">Sesame</span>
              <span className={voice.sesame_loaded ? 'text-green-400' : 'text-chat-text-secondary'}>
                {voice.sesame_loaded ? 'Loaded' : 'Not loaded'}
              </span>
            </div>
          </div>
        ) : (
          <div className="text-chat-text-secondary text-sm">Voice not available</div>
        )}
      </div>

      {/* Session Costs */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2 mb-3">
          <CurrencyDollarIcon className="w-5 h-5 text-green-400" />
          Session Cost
        </h3>

        {costs ? (
          <div className="grid grid-cols-2 gap-2">
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-green-400">
                ${costs.cost_usd < 0.001 ? costs.cost_usd.toFixed(6) : costs.cost_usd.toFixed(4)}
              </div>
              <div className="text-xs text-chat-text-secondary">Est. Cost USD</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-lg font-bold text-chat-text">
                {costs.total_tokens.toLocaleString()}
              </div>
              <div className="text-xs text-chat-text-secondary">Total Tokens</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-sm font-bold text-blue-400">
                {costs.input_tokens.toLocaleString()}
              </div>
              <div className="text-xs text-chat-text-secondary">Input</div>
            </div>
            <div className="bg-chat-assistant rounded p-2 text-center">
              <div className="text-sm font-bold text-purple-400">
                {costs.output_tokens.toLocaleString()}
              </div>
              <div className="text-xs text-chat-text-secondary">Output</div>
            </div>
            <div className="col-span-2 flex justify-between text-xs text-chat-text-secondary px-1">
              <span>{costs.queries} queries this session</span>
              <button onClick={fetchData} className="text-blue-400 hover:text-blue-300">refresh</button>
            </div>
          </div>
        ) : (
          <div className="text-chat-text-secondary text-sm">No session data yet</div>
        )}
      </div>

      {/* Plugin Hot-Reload */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-chat-text font-medium flex items-center gap-2">
            <PuzzlePieceIcon className="w-5 h-5 text-yellow-400" />
            Plugins
          </h3>
          <button
            onClick={handleReloadPlugins}
            disabled={reloading}
            className="px-2 py-1 text-xs bg-yellow-900/30 text-yellow-300 hover:bg-yellow-800/40 rounded transition-colors disabled:opacity-50"
          >
            {reloading ? 'Reloading...' : 'Hot Reload'}
          </button>
        </div>
        {reloadMsg && (
          <div className="text-xs text-green-400 mt-1">{reloadMsg}</div>
        )}
        <div className="text-xs text-chat-text-secondary">
          Reload custom tools from registry without restarting AURA.
        </div>
      </div>

      {/* Tools Discovery */}
      <div className="bg-chat-sidebar rounded-lg p-4">
        <h3 className="text-chat-text font-medium flex items-center gap-2 mb-3">
          <WrenchScrewdriverIcon className="w-5 h-5 text-yellow-400" />
          Tools ({filteredTools.length}/{tools.length})
        </h3>

        {/* Search */}
        <div className="relative mb-3">
          <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-chat-text-secondary" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search tools..."
            className="w-full bg-chat-assistant border border-chat-border rounded pl-8 pr-3 py-1.5 text-sm text-chat-text placeholder-chat-text-secondary/60 focus:outline-none focus:border-yellow-500/50"
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-chat-text-secondary hover:text-chat-text text-xs"
            >
              ✕
            </button>
          )}
        </div>

        {/* Category filter */}
        {categories.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            <button
              onClick={() => setActiveCategory(null)}
              className={`px-2 py-0.5 rounded text-xs transition-colors ${
                activeCategory === null
                  ? 'bg-yellow-600 text-white'
                  : 'bg-chat-assistant text-chat-text-secondary hover:text-chat-text'
              }`}
            >
              All
            </button>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat === activeCategory ? null : cat)}
                className={`px-2 py-0.5 rounded text-xs transition-colors ${
                  activeCategory === cat
                    ? 'ring-1 ring-white/30 ' + getCategoryColor(cat)
                    : getCategoryColor(cat)
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        )}

        {/* Tool grid */}
        <div className="space-y-1.5 max-h-96 overflow-y-auto pr-0.5">
          {filteredTools.length === 0 && (
            <div className="text-chat-text-secondary text-sm text-center py-4">
              No tools match "{search}"
            </div>
          )}
          {filteredTools.map((tool) => (
            <div
              key={tool.name}
              className="bg-chat-assistant rounded p-2 text-xs flex items-start gap-2"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5 mb-0.5">
                  <span className="text-chat-text font-medium truncate">{tool.name}</span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] shrink-0 ${getToolBadgeColor(tool.category)}`}>
                    {tool.category}
                  </span>
                </div>
                <div className="text-chat-text-secondary leading-relaxed" title={tool.description}>
                  {tool.description}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
