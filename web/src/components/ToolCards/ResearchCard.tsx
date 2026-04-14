import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Deep-research tool card: renders the report as markdown and lists the
// sources in a collapsible section below.

interface ResearchCardProps {
  result: Extract<ToolResult, { kind: 'research' }>;
}

export function ResearchCard({ result }: ResearchCardProps) {
  const sources = result.sources || [];

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F9EA}'}
      title="Research"
      subtitle={result.query || undefined}
      status={result.status}
    >
      {result.report && (
        <div className="tool-card-report markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {result.report}
          </ReactMarkdown>
        </div>
      )}
      {!result.report && result.status === 'running' && (
        <div className="tool-card-placeholder">Researching\u2026</div>
      )}
      {sources.length > 0 && (
        <details className="tool-card-sources" open={false}>
          <summary>{sources.length} source{sources.length === 1 ? '' : 's'}</summary>
          <ol className="tool-card-source-list">
            {sources.map((s) => (
              <li key={s.id}>
                <a href={s.url} target="_blank" rel="noopener noreferrer">
                  {s.title || s.url}
                </a>
                {s.snippet && <p className="tool-card-source-snippet">{s.snippet}</p>}
              </li>
            ))}
          </ol>
        </details>
      )}
    </ToolCardShell>
  );
}
