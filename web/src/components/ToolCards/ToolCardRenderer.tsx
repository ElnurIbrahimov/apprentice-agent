import type { ToolResult } from '../../types';
import { CodeCard } from './CodeCard';
import { ImageCard } from './ImageCard';
import { ResearchCard } from './ResearchCard';
import { SearchCard } from './SearchCard';
import { SummaryCard } from './SummaryCard';
import { GenericToolCard } from './GenericToolCard';
import { DocumentCard } from './DocumentCard';

// Dispatch on ToolResult.kind and render the matching card. Used by
// MiniApp's ChatTab to render msg.toolResults above the markdown bubble.

interface ToolCardRendererProps {
  results: ToolResult[];
  onQuestionClick?: (question: string) => void;
}

export function ToolCardRenderer({ results, onQuestionClick }: ToolCardRendererProps) {
  if (!results || results.length === 0) return null;
  return (
    <div className="tool-card-stack">
      {results.map((r) => {
        switch (r.kind) {
          case 'code': return <CodeCard key={r.id} result={r} />;
          case 'image': return <ImageCard key={r.id} result={r} />;
          case 'research': return <ResearchCard key={r.id} result={r} />;
          case 'search': return <SearchCard key={r.id} result={r} />;
          case 'summary': return <SummaryCard key={r.id} result={r} />;
          case 'document': return <DocumentCard key={r.id} result={r} onQuestionClick={onQuestionClick} />;
          case 'generic': return <GenericToolCard key={r.id} result={r} />;
          default: return null;
        }
      })}
    </div>
  );
}

export * from './ToolCardShell';
export { CodeCard } from './CodeCard';
export { ImageCard } from './ImageCard';
export { ResearchCard } from './ResearchCard';
export { SearchCard } from './SearchCard';
export { SummaryCard } from './SummaryCard';
export { DocumentCard } from './DocumentCard';
export { GenericToolCard } from './GenericToolCard';
