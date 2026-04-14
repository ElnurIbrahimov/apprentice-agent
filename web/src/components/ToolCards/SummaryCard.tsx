import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Lightweight summary card: title + prose + optional key points.

interface SummaryCardProps {
  result: Extract<ToolResult, { kind: 'summary' }>;
}

export function SummaryCard({ result }: SummaryCardProps) {
  const highlights = result.highlights || [];

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F4DD}'}
      title={result.title || 'Summary'}
      status={result.status}
    >
      {result.summary && (
        <p className="tool-card-summary-text">{result.summary}</p>
      )}
      {highlights.length > 0 && (
        <ul className="tool-card-highlights">
          {highlights.map((h, i) => (
            <li key={`${result.id}-hl-${i}`}>{h}</li>
          ))}
        </ul>
      )}
      {!result.summary && result.status === 'running' && (
        <div className="tool-card-placeholder">Summarizing\u2026</div>
      )}
    </ToolCardShell>
  );
}
