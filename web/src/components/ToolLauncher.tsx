import { useState, useMemo } from 'react';
import { MagnifyingGlassIcon, ArrowLeftIcon, WrenchScrewdriverIcon } from '@heroicons/react/24/outline';

export type ToolId =
  | 'ask' | 'search' | 'research' | 'agent' | 'compare'
  | 'write' | 'translate' | 'summary' | 'grammar'
  | 'pdf' | 'ocr' | 'capture' | 'youtube'
  | 'voice' | 'record'
  | 'math' | 'slides' | 'wisebase' | 'models';

interface Tool {
  id: ToolId;
  label: string;
  icon: string;
  bg: string;   // Tailwind bg class for icon container
  desc: string;
}

interface ToolGroup {
  label: string;
  tools: Tool[];
}

const TOOL_GROUPS: ToolGroup[] = [
  {
    label: 'AI Tools',
    tools: [
      { id: 'ask',      label: 'Ask',      icon: '💬', bg: 'bg-blue-500/15',    desc: 'Quick Q&A' },
      { id: 'search',   label: 'Search',   icon: '🔍', bg: 'bg-cyan-500/15',    desc: 'Web search' },
      { id: 'research', label: 'Research', icon: '🔬', bg: 'bg-violet-500/15',  desc: 'Deep analysis' },
      { id: 'agent',    label: 'Agent',    icon: '🤖', bg: 'bg-emerald-500/15', desc: 'Autonomous tasks' },
      { id: 'compare',  label: 'Compare',  icon: '⚖️', bg: 'bg-amber-500/15',   desc: 'Model comparison' },
    ],
  },
  {
    label: 'Writing',
    tools: [
      { id: 'write',     label: 'Write',     icon: '✏️', bg: 'bg-orange-500/15',  desc: 'Content creation' },
      { id: 'translate', label: 'Translate', icon: '🌐', bg: 'bg-sky-500/15',     desc: 'Multi-language' },
      { id: 'summary',   label: 'Summary',   icon: '📋', bg: 'bg-teal-500/15',    desc: 'Summarize text' },
      { id: 'grammar',   label: 'Grammar',   icon: '📝', bg: 'bg-green-500/15',   desc: 'Grammar check' },
    ],
  },
  {
    label: 'Media',
    tools: [
      { id: 'pdf',     label: 'PDF',     icon: '📄', bg: 'bg-red-500/15',     desc: 'PDF analysis' },
      { id: 'ocr',     label: 'OCR',     icon: '🔎', bg: 'bg-indigo-500/15',  desc: 'Image text' },
      { id: 'capture', label: 'Capture', icon: '📸', bg: 'bg-fuchsia-500/15', desc: 'Screenshot analysis' },
      { id: 'youtube', label: 'YouTube', icon: '▶️', bg: 'bg-rose-500/15',    desc: 'Video analysis' },
    ],
  },
  {
    label: 'Audio',
    tools: [
      { id: 'voice',  label: 'Voice',  icon: '🔊', bg: 'bg-purple-500/15',  desc: 'Text to speech' },
      { id: 'record', label: 'Record', icon: '🎙️', bg: 'bg-pink-500/15',    desc: 'Audio recording' },
    ],
  },
  {
    label: 'Productivity',
    tools: [
      { id: 'math',     label: 'Math',     icon: '🧮', bg: 'bg-blue-500/15',   desc: 'Math solver' },
      { id: 'slides',   label: 'Slides',   icon: '📊', bg: 'bg-amber-500/15',  desc: 'Presentations' },
      { id: 'wisebase', label: 'Wisebase', icon: '📚', bg: 'bg-violet-500/15', desc: 'Knowledge base' },
      { id: 'models',   label: 'Models',   icon: '⚙️', bg: 'bg-slate-500/15',  desc: 'Model manager' },
    ],
  },
];

// Flat list of all tools for search
const ALL_TOOLS: Tool[] = TOOL_GROUPS.flatMap((g) => g.tools);

interface ToolLauncherProps {
  onSelect: (id: ToolId | 'tools') => void;
}

export function ToolLauncher({ onSelect }: ToolLauncherProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    if (!query.trim()) return null;
    const q = query.toLowerCase();
    return ALL_TOOLS.filter(
      (t) => t.label.toLowerCase().includes(q) || t.desc.toLowerCase().includes(q),
    );
  }, [query]);

  return (
    <div className="h-full overflow-y-auto tab-panel-scroll" style={{ background: 'var(--bg-base)' }}>
      <div className="max-w-3xl mx-auto px-4 py-6">

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-chat-text mb-1">Tools</h2>
          <p className="text-sm text-chat-text-secondary">Pick a tool to get started</p>
        </div>

        {/* Search */}
        <div className="relative mb-6">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-chat-text-secondary pointer-events-none" />
          <input
            type="text"
            placeholder="Search tools…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 rounded-xl text-sm text-chat-text placeholder:text-chat-text-secondary border border-chat-border focus:outline-none focus:border-chat-accent transition-colors"
            style={{ background: 'var(--surface-2)' }}
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-chat-text-secondary hover:text-chat-text"
              aria-label="Clear search"
            >
              ✕
            </button>
          )}
        </div>

        {/* System Tools special card */}
        {!filtered && (
          <button
            onClick={() => onSelect('tools')}
            className="w-full flex items-center gap-3 px-4 py-3 rounded-xl border border-chat-border mb-6 hover:border-chat-accent transition-colors group text-left"
            style={{ background: 'var(--surface-2)' }}
          >
            <span className="flex items-center justify-center w-10 h-10 rounded-lg shrink-0"
              style={{ background: 'var(--surface-3)' }}>
              <WrenchScrewdriverIcon className="w-5 h-5 text-chat-accent" />
            </span>
            <div>
              <div className="text-sm font-semibold text-chat-text group-hover:text-chat-accent transition-colors">System Tools</div>
              <div className="text-xs text-chat-text-secondary">Active tools &amp; integrations</div>
            </div>
          </button>
        )}

        {/* Search results */}
        {filtered && (
          filtered.length === 0 ? (
            <p className="text-center text-sm text-chat-text-secondary py-8">No tools match "{query}"</p>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {filtered.map((tool) => (
                <ToolCard key={tool.id} tool={tool} onSelect={onSelect} />
              ))}
            </div>
          )
        )}

        {/* Grouped tool grid */}
        {!filtered && TOOL_GROUPS.map((group) => (
          <div key={group.label} className="mb-8">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-chat-text-secondary mb-3">
              {group.label}
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {group.tools.map((tool) => (
                <ToolCard key={tool.id} tool={tool} onSelect={onSelect} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ToolCard({ tool, onSelect }: { tool: Tool; onSelect: (id: ToolId | 'tools') => void }) {
  return (
    <button
      onClick={() => onSelect(tool.id)}
      className="flex flex-col items-start gap-3 p-4 rounded-xl border border-chat-border hover:border-chat-accent/50 transition-all group text-left hover:shadow-lg hover:shadow-chat-accent/5"
      style={{ background: 'var(--surface-2)' }}
    >
      <span className={`flex items-center justify-center w-10 h-10 rounded-lg text-xl ${tool.bg}`}>
        {tool.icon}
      </span>
      <div>
        <div className="text-sm font-semibold text-chat-text group-hover:text-chat-accent transition-colors leading-tight">
          {tool.label}
        </div>
        <div className="text-xs text-chat-text-secondary mt-0.5 leading-snug">
          {tool.desc}
        </div>
      </div>
    </button>
  );
}

// Compact sub-nav shown while a tool is active
interface ToolSubNavProps {
  activeId: ToolId | 'tools';
  onSelect: (id: ToolId | 'tools') => void;
  onBack: () => void;
}

export function ToolSubNav({ activeId, onSelect, onBack }: ToolSubNavProps) {
  const activeTool = ALL_TOOLS.find((t) => t.id === activeId);
  const activeGroup = TOOL_GROUPS.find((g) => g.tools.some((t) => t.id === activeId));

  return (
    <div
      className="flex items-center gap-2 px-4 py-2 border-b border-chat-border overflow-x-auto scrollbar-hide"
      style={{ background: 'var(--bg-panel)' }}
    >
      {/* Back button */}
      <button
        onClick={onBack}
        className="flex items-center gap-1.5 shrink-0 px-2.5 py-1.5 rounded-lg text-xs font-medium text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant transition-colors"
        aria-label="Back to launcher"
      >
        <ArrowLeftIcon className="w-3.5 h-3.5" />
        <span>All Tools</span>
      </button>

      <span className="text-chat-border shrink-0">|</span>

      {/* Current tool label */}
      {activeTool && (
        <span className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-chat-accent text-white">
          <span>{activeTool.icon}</span>
          <span>{activeTool.label}</span>
        </span>
      )}
      {activeId === 'tools' && (
        <span className="shrink-0 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-chat-accent text-white">
          System Tools
        </span>
      )}

      {/* Other tools in the same group as quick-switch */}
      {activeGroup && activeGroup.tools
        .filter((t) => t.id !== activeId)
        .map((tool) => (
          <button
            key={tool.id}
            onClick={() => onSelect(tool.id)}
            className="shrink-0 flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-chat-text-secondary hover:text-chat-text hover:bg-chat-assistant transition-colors"
          >
            <span>{tool.icon}</span>
            <span>{tool.label}</span>
          </button>
        ))}
    </div>
  );
}
