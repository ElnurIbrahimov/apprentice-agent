import { useState } from 'react';
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

interface Props {
  traces: ToolTraceType[];
  isStreaming?: boolean;
}

export function ToolTrace({ traces, isStreaming }: Props) {
  const [collapsed, setCollapsed] = useState(false);

  if (traces.length === 0) return null;

  // Auto-collapse when done streaming
  const isRunning = isStreaming || traces.some(t => t.event === 'start' && !traces.find(d => d.tool === t.tool && d.event === 'done'));

  return (
    <div className="mb-2 rounded-lg border border-gray-700/50 bg-gray-800/40 overflow-hidden text-xs">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-left text-gray-400 hover:text-gray-200 hover:bg-gray-700/30 transition-colors"
      >
        <span className={`transition-transform ${collapsed ? '' : 'rotate-90'}`}>▶</span>
        <span className="font-medium">
          {isRunning ? 'Running tools...' : `${traces.length} tool action${traces.length !== 1 ? 's' : ''}`}
        </span>
        {isRunning && (
          <span className="ml-auto flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '300ms' }} />
          </span>
        )}
      </button>
      {!collapsed && (
        <div className="px-3 pb-2 space-y-1">
          {traces.map((trace, i) => {
            const isDone = trace.event === 'done';
            const isError = trace.event === 'error';
            return (
              <div key={i} className="flex items-center gap-2 text-gray-400">
                <span className="flex-shrink-0">{getIcon(trace.tool)}</span>
                <span className="flex-shrink-0 font-mono text-gray-300">{trace.tool}</span>
                {trace.detail && <span className="text-gray-500 truncate">— {trace.detail}</span>}
                <span className="ml-auto flex-shrink-0">
                  {isDone ? (
                    <span className="text-green-400">✓ {trace.elapsed_ms ? `${(trace.elapsed_ms / 1000).toFixed(1)}s` : ''}</span>
                  ) : isError ? (
                    <span className="text-red-400">✗</span>
                  ) : (
                    <span className="text-purple-400 animate-pulse">●</span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
