import { useState } from 'react';

interface MemoryIndicatorProps {
  memories: string[];
  source?: string;
}

export function MemoryIndicator({ memories, source }: MemoryIndicatorProps) {
  const [expanded, setExpanded] = useState(false);

  if (!memories || memories.length === 0) return null;

  return (
    <div className="mt-2">
      {/* Collapsed pill */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-all duration-200 hover:scale-[1.02] active:scale-[0.98]"
        style={{
          background: expanded
            ? 'rgba(168, 85, 247, 0.15)'
            : 'rgba(168, 85, 247, 0.10)',
          border: '1px solid rgba(168, 85, 247, 0.25)',
          color: '#c4b5fd',
        }}
      >
        <span className="text-sm leading-none">&#129504;</span>
        <span className="font-medium">
          Remembered {memories.length} item{memories.length !== 1 ? 's' : ''}
        </span>
        <span
          className={`transition-transform duration-200 text-[10px] text-purple-400 ${expanded ? 'rotate-90' : ''}`}
        >
          &#9654;
        </span>
      </button>

      {/* Expanded glass card */}
      <div
        className="overflow-hidden transition-all duration-300 ease-in-out"
        style={{
          maxHeight: expanded ? `${memories.length * 40 + 24}px` : '0px',
          opacity: expanded ? 1 : 0,
        }}
      >
        <div
          className="mt-1.5 rounded-lg border border-white/[0.06] p-3"
          style={{ background: 'rgba(168, 85, 247, 0.06)' }}
        >
          {source && (
            <div className="text-[10px] uppercase tracking-wider text-purple-400/60 font-semibold mb-1.5">
              {source}
            </div>
          )}
          <ul className="space-y-1">
            {memories.map((snippet, idx) => (
              <li
                key={idx}
                className="flex items-start gap-2 text-xs text-gray-300/80"
              >
                <span className="text-purple-400/50 mt-0.5 flex-shrink-0">&#8226;</span>
                <span className="leading-relaxed break-words">
                  {snippet.length > 120 ? snippet.slice(0, 120) + '...' : snippet}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
