import type { ToolResult } from '../../types';
import { ToolCardShell } from './ToolCardShell';

// Compact list of web search results: favicon + title + snippet, linkified.

interface SearchCardProps {
  result: Extract<ToolResult, { kind: 'search' }>;
}

function favicon(url: string): string {
  try {
    const host = new URL(url).hostname;
    return `https://www.google.com/s2/favicons?sz=32&domain=${host}`;
  } catch {
    return '';
  }
}

export function SearchCard({ result }: SearchCardProps) {
  const results = result.results || [];
  const count = results.length;

  return (
    <ToolCardShell
      icon={result.iconOverride || '\u{1F50D}'}
      title="Web search"
      subtitle={result.query || undefined}
      status={result.status}
    >
      {count > 0 ? (
        <ul className="tool-card-list">
          {results.slice(0, 8).map((r, i) => (
            <li key={`${result.id}-${i}`} className="tool-card-search-item">
              <a href={r.url} target="_blank" rel="noopener noreferrer">
                {favicon(r.url) && <img src={favicon(r.url)} alt="" className="tool-card-favicon" loading="lazy" />}
                <div className="tool-card-search-text">
                  <span className="tool-card-search-title">{r.title || r.url}</span>
                  {r.snippet && <span className="tool-card-search-snippet">{r.snippet}</span>}
                </div>
              </a>
            </li>
          ))}
          {count > 8 && (
            <li className="tool-card-more">\u2026 +{count - 8} more</li>
          )}
        </ul>
      ) : (
        result.status === 'running' && <div className="tool-card-placeholder">Searching\u2026</div>
      )}
    </ToolCardShell>
  );
}
