/**
 * TerminalOutput — displays terminal output from WebContainer operations
 * (npm install, dev server, build output, runtime errors).
 */

import { useEffect, useRef, useState } from 'react';
import { Copy, Check, Trash2, ChevronDown, ChevronUp } from 'lucide-react';

export interface TerminalLine {
  text: string;
  type: 'stdout' | 'stderr' | 'system';
  timestamp: number;
}

interface TerminalOutputProps {
  lines: TerminalLine[];
  maxLines?: number;
  autoScroll?: boolean;
  title?: string;
  collapsible?: boolean;
  defaultOpen?: boolean;
  onClear?: () => void;
}

const TYPE_COLORS: Record<string, string> = {
  stdout: '#e6edf3',
  stderr: '#f87171',
  system: '#7dd3fc',
};

export default function TerminalOutput({
  lines,
  maxLines = 500,
  autoScroll = true,
  title = 'Terminal',
  collapsible = true,
  defaultOpen = false,
  onClear,
}: TerminalOutputProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const visibleLines = lines.slice(-maxLines);

  useEffect(() => {
    if (autoScroll && open && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [visibleLines.length, autoScroll, open]);

  const copyAll = async () => {
    const text = visibleLines.map(l => l.text).join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard not available */ }
  };

  const errorCount = visibleLines.filter(l => l.type === 'stderr').length;

  if (collapsible && !open) {
    return (
      <button
        onClick={() => setOpen(true)}
        style={{
          display: 'flex', alignItems: 'center', gap: 6, width: '100%',
          padding: '6px 10px', background: '#0d1117', border: 'none',
          borderTop: '1px solid #30363d', color: '#8b949e',
          fontSize: '10.5px', cursor: 'pointer', fontFamily: 'inherit',
        }}
      >
        <ChevronUp size={12} />
        <span>{title}</span>
        {visibleLines.length > 0 && (
          <span style={{ color: '#7dd3fc' }}>({visibleLines.length} lines)</span>
        )}
        {errorCount > 0 && (
          <span style={{
            color: '#f87171', background: 'rgba(248,113,113,0.12)',
            padding: '1px 6px', borderRadius: 999, fontSize: '9.5px',
          }}>
            {errorCount} error{errorCount !== 1 ? 's' : ''}
          </span>
        )}
      </button>
    );
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', background: '#0d1117',
      borderTop: '1px solid #30363d', flexShrink: 0, maxHeight: 200,
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '4px 10px', borderBottom: '1px solid #21262d',
      }}>
        {collapsible && (
          <button
            onClick={() => setOpen(false)}
            style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', padding: 0, display: 'flex' }}
          >
            <ChevronDown size={12} />
          </button>
        )}
        <span style={{ fontSize: '10px', fontWeight: 600, color: '#8b949e', flex: 1 }}>{title}</span>
        <button
          onClick={copyAll}
          style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', padding: '2px', display: 'flex' }}
          title="Copy all"
        >
          {copied ? <Check size={11} style={{ color: '#3fb950' }} /> : <Copy size={11} />}
        </button>
        {onClear && (
          <button
            onClick={onClear}
            style={{ background: 'none', border: 'none', color: '#8b949e', cursor: 'pointer', padding: '2px', display: 'flex' }}
            title="Clear"
          >
            <Trash2 size={11} />
          </button>
        )}
      </div>
      <div
        ref={scrollRef}
        style={{
          flex: 1, overflow: 'auto', padding: '6px 10px',
          fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
          fontSize: '11px', lineHeight: 1.6, minHeight: 60,
        }}
      >
        {visibleLines.length === 0 ? (
          <div style={{ color: '#484f58', fontStyle: 'italic' }}>Waiting for output...</div>
        ) : (
          visibleLines.map((line, i) => (
            <div key={i} style={{ color: TYPE_COLORS[line.type] || '#e6edf3', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {line.text}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
