import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { ModelResult } from '../types';

interface ModelCompareProps {
  results: ModelResult[];
  query: string;
  onUseResponse?: (response: string) => void;
}

function modelDisplayName(modelId: string): string {
  // Strip `:cloud` suffix and clean up
  return modelId.replace(/:cloud$/, '').replace(/-preview$/, '');
}

function formatTime(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export function ModelCompare({ results, query, onUseResponse }: ModelCompareProps) {
  const [copied, setCopied] = useState<string | null>(null);

  const handleCopy = async (text: string, modelId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(modelId);
      setTimeout(() => setCopied(null), 2000);
    } catch { /* clipboard permission denied */ }
  };

  return (
    <div className="mt-3 space-y-2">
      <div className="text-xs text-chat-text-secondary mb-3 flex items-center gap-1.5">
        <span className="text-aura-purple font-medium">Comparing {results.length} models</span>
        <span className="text-chat-border">·</span>
        <span className="truncate max-w-[300px] opacity-70">"{query}"</span>
      </div>

      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(results.length, 3)}, minmax(0, 1fr))` }}>
        {results.map((result) => (
          <div
            key={result.model}
            className={`
              flex flex-col rounded-xl border bg-chat-assistant/60 overflow-hidden
              ${result.error ? 'border-red-500/30' : 'border-chat-border/50'}
            `}
          >
            {/* Header */}
            <div className="flex items-center justify-between px-3 py-2 border-b border-chat-border/30 bg-chat-bg/40">
              <span className="text-xs font-mono font-medium text-aura-purple truncate max-w-[70%]">
                {modelDisplayName(result.model)}
              </span>
              <span className={`text-xs tabular-nums ${result.error ? 'text-red-400' : 'text-chat-text-secondary'}`}>
                {formatTime(result.time_ms)}
              </span>
            </div>

            {/* Content */}
            <div className="flex-1 px-3 py-2.5 text-sm text-chat-text overflow-y-auto max-h-64">
              {result.error ? (
                <span className="text-red-400 text-xs">{result.error}</span>
              ) : (
                <ReactMarkdown
                  className="prose prose-invert prose-sm max-w-none"
                >
                  {result.response}
                </ReactMarkdown>
              )}
            </div>

            {/* Actions */}
            {!result.error && (
              <div className="flex items-center gap-2 px-3 py-2 border-t border-chat-border/20">
                <button
                  onClick={() => onUseResponse?.(result.response)}
                  className="text-xs px-2.5 py-1 rounded-md bg-aura-purple/20 text-aura-purple hover:bg-aura-purple/30 transition-colors"
                >
                  Use this
                </button>
                <button
                  onClick={() => handleCopy(result.response, result.model)}
                  className="text-xs px-2.5 py-1 rounded-md bg-chat-border/20 text-chat-text-secondary hover:text-chat-text hover:bg-chat-border/40 transition-colors"
                >
                  {copied === result.model ? 'Copied!' : 'Copy'}
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
