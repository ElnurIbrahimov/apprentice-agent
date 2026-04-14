import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';
import { CodeBlock } from '../CodeBlock';

// Renders a code tool-call result inside the chat. Reuses the existing
// shiki-backed CodeBlock (syntax highlighting, copy, collapse, run-python)
// and shows captured stdout/stderr underneath when the tool returned output.

interface CodeCardProps {
  result: Extract<ToolResult, { kind: 'code' }>;
}

export function CodeCard({ result }: CodeCardProps) {
  const lang = result.language || 'text';
  const sourceText = (result.source || '').trim();
  const hasOutput = typeof result.output === 'string' && result.output.length > 0;

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F4BB}'}
      title="Code"
      subtitle={lang}
      status={result.status}
    >
      {sourceText && (
        <div className="tool-card-code">
          <CodeBlock language={lang}>{sourceText}</CodeBlock>
        </div>
      )}
      {hasOutput && (
        <details className="tool-card-output" open={result.status === 'done'}>
          <summary>Output</summary>
          <pre>{result.output}</pre>
        </details>
      )}
    </ToolCardShell>
  );
}
