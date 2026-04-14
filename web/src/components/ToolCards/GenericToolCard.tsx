import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Fallback for any tool that doesn't have a dedicated card. Shows the tool
// name in the header and exposes args + result inside a <details> pane.

interface GenericToolCardProps {
  result: Extract<ToolResult, { kind: 'generic' }>;
}

function prettyArgs(args: Record<string, unknown> | undefined): string {
  if (!args) return '';
  try {
    return JSON.stringify(args, null, 2);
  } catch {
    return String(args);
  }
}

export function GenericToolCard({ result }: GenericToolCardProps) {
  const hasArgs = !!result.args && Object.keys(result.args).length > 0;
  const hasResult = typeof result.result === 'string' && result.result.length > 0;
  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F527}'}
      title={result.tool || 'Tool'}
      status={result.status}
      compact
    >
      {hasArgs && (
        <details className="tool-card-generic-section" open={false}>
          <summary>Arguments</summary>
          <pre>{prettyArgs(result.args)}</pre>
        </details>
      )}
      {hasResult && (
        <details className="tool-card-generic-section" open={result.status === 'done'}>
          <summary>Result</summary>
          <pre>{result.result}</pre>
        </details>
      )}
    </ToolCardShell>
  );
}
