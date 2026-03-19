import { useState, useEffect } from 'react';
import type { ToolTrace as ToolTraceType } from '../types';

const TOOL_ICONS: Record<string, string> = {
  tavily_search: '🔍',
  web_search: '🌐',
  code_executor: '⚡',
  filesystem: '📁',
  git_tool: '🌿',
  browser: '🌍',
  default: '🔧',
};

function getIcon(tool: string): string {
  return TOOL_ICONS[tool] || TOOL_ICONS.default;
}

// Friendly descriptions for tool names
const TOOL_DESCRIPTIONS: Record<string, string> = {
  tavily_search: 'Searching the web',
  web_search: 'Web search',
  code_executor: 'Running code',
  filesystem: 'File operation',
  git_tool: 'Git operation',
  browser: 'Browsing page',
};

function getDescription(tool: string): string {
  return TOOL_DESCRIPTIONS[tool] || 'Tool action';
}

interface ToolCardProps {
  trace: ToolTraceType;
  defaultExpanded: boolean;
}

function ToolCard({ trace, defaultExpanded }: ToolCardProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const isDone = trace.event === 'done';
  const isError = trace.event === 'error';
  const isRunning = trace.event === 'start';

  // Auto-expand when running, keep user choice otherwise
  useEffect(() => {
    if (isRunning) setExpanded(true);
  }, [isRunning]);

  const borderColor = isError
    ? 'border-l-red-400'
    : isRunning
      ? 'border-l-amber-400'
      : 'border-l-green-400';

  const dotColor = isError
    ? 'bg-red-400'
    : isRunning
      ? 'bg-amber-400'
      : 'bg-green-400';

  return (
    <div
      className={`rounded-lg border border-white/[0.06] border-l-2 ${borderColor} overflow-hidden transition-colors`}
      style={{ background: 'rgba(255,255,255,0.03)' }}
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2.5 px-3 py-2 text-left text-gray-300 hover:text-gray-100 transition-colors"
      >
        {/* Status dot */}
        <span className={`flex-shrink-0 w-2 h-2 rounded-full ${dotColor} ${isRunning ? 'animate-pulse' : ''}`} />
        {/* Icon */}
        <span className="flex-shrink-0 text-sm">{getIcon(trace.tool)}</span>
        {/* Tool name */}
        <span className="flex-shrink-0 font-mono text-xs font-medium text-gray-200">{trace.tool}</span>
        {/* Description */}
        <span className="text-xs text-gray-500 truncate">{getDescription(trace.tool)}</span>
        {/* Duration / status — right side */}
        <span className="ml-auto flex-shrink-0 text-xs">
          {isDone && trace.elapsed_ms != null ? (
            <span className="text-green-400">{(trace.elapsed_ms / 1000).toFixed(1)}s</span>
          ) : isError ? (
            <span className="text-red-400">failed</span>
          ) : (
            <span className="text-amber-400">running</span>
          )}
        </span>
        {/* Expand chevron */}
        {trace.detail && (
          <span className={`flex-shrink-0 text-gray-600 transition-transform text-[10px] ${expanded ? 'rotate-90' : ''}`}>
            ▶
          </span>
        )}
      </button>
      {expanded && trace.detail && (
        <div className="px-3 pb-2.5 pt-0 text-xs text-gray-400 border-t border-white/[0.04] mt-0 pt-2 mx-3 mb-2">
          <pre className="whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed">{trace.detail}</pre>
        </div>
      )}
    </div>
  );
}

interface Props {
  traces: ToolTraceType[];
  isStreaming?: boolean;
}

export function ToolTrace({ traces, isStreaming }: Props) {
  const isRunning = isStreaming || traces.some(t => t.event === 'start' && !traces.find(d => d.tool === t.tool && d.event === 'done'));

  // Collapsed by default when done, expanded when running
  const [sectionCollapsed, setSectionCollapsed] = useState(false);

  // Auto-collapse when transitions from running to done
  useEffect(() => {
    if (!isRunning && traces.length > 0) {
      setSectionCollapsed(true);
    } else if (isRunning) {
      setSectionCollapsed(false);
    }
  }, [isRunning, traces.length]);

  if (traces.length === 0) return null;

  return (
    <div className="mb-2 text-xs">
      <button
        onClick={() => setSectionCollapsed(!sectionCollapsed)}
        className="w-full flex items-center gap-2 px-1 py-1.5 text-left text-gray-400 hover:text-gray-200 transition-colors"
      >
        <span className={`transition-transform text-[10px] ${sectionCollapsed ? '' : 'rotate-90'}`}>▶</span>
        <span className="font-medium">
          {isRunning ? 'Running tools...' : `${traces.length} tool action${traces.length !== 1 ? 's' : ''}`}
        </span>
        {isRunning && (
          <span className="ml-auto flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-amber-400 animate-bounce" style={{ animationDelay: '300ms' }} />
          </span>
        )}
      </button>
      {!sectionCollapsed && (
        <div className="space-y-1.5 mt-1">
          {traces.map((trace, i) => (
            <ToolCard
              key={`${trace.tool}-${trace.timestamp}-${i}`}
              trace={trace}
              defaultExpanded={trace.event === 'start'}
            />
          ))}
        </div>
      )}
    </div>
  );
}
