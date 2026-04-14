import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Renders an uploaded document as a rich chat card: TL;DR summary,
// collapsible key facts, and tappable question chips that send the
// question into the chat WebSocket as a new user message.

interface DocumentCardProps {
  result: Extract<ToolResult, { kind: 'document' }>;
  onQuestionClick?: (question: string) => void;
}

export function DocumentCard({ result, onQuestionClick }: DocumentCardProps) {
  const facts = result.facts || [];
  const questions = result.questions || [];
  const hasSummary = !!result.summary && result.summary.trim().length > 0;
  const chunksLabel = result.chunks_count
    ? `${result.chunks_count} chunk${result.chunks_count === 1 ? '' : 's'} indexed`
    : 'Indexing\u2026';
  const sizeLabel = result.size_chars
    ? ` \u00B7 ${(result.size_chars / 1000).toFixed(1)}k chars`
    : '';

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F4C4}'}
      title={result.filename || 'Document'}
      subtitle={`${chunksLabel}${sizeLabel}`}
      status={result.status}
    >
      {hasSummary ? (
        <div className="tool-card-summary-text">{result.summary}</div>
      ) : (
        result.status !== 'error' && (
          <div className="tool-card-placeholder">Summarizing\u2026</div>
        )
      )}

      {facts.length > 0 && (
        <details className="tool-card-generic-section" open={false}>
          <summary>Key facts ({facts.length})</summary>
          <ul className="tool-card-highlights">
            {facts.map((f, i) => (
              <li key={`${result.id}-fact-${i}`}>{f}</li>
            ))}
          </ul>
        </details>
      )}

      {questions.length > 0 && (
        <div className="doc-card-questions">
          <div className="doc-card-questions-label">Ask this document:</div>
          <div className="doc-card-questions-chips">
            {questions.map((q, i) => (
              <button
                key={`${result.id}-q-${i}`}
                className="doc-card-question-chip"
                onClick={() => onQuestionClick?.(q)}
                type="button"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}
    </ToolCardShell>
  );
}
