import type { ReactNode } from 'react';
import type { ToolStatus } from '../../types';

// Shared shell for every tool card. Keeps the header, status pill, and
// border treatment consistent across CodeCard, ImageCard, ResearchCard,
// SearchCard, SummaryCard, and GenericToolCard.

interface ToolCardShellProps {
  icon: string;
  title: string;
  subtitle?: string;
  status: ToolStatus;
  children?: ReactNode;
  compact?: boolean;
}

const STATUS_LABEL: Record<ToolStatus, string> = {
  running: 'Running\u2026',
  done: 'Done',
  error: 'Error',
};

export function ToolCardShell({ icon, title, subtitle, status, children, compact }: ToolCardShellProps) {
  return (
    <div className={`tool-card tool-card--${status}${compact ? ' tool-card--compact' : ''}`}>
      <div className="tool-card-header">
        <span className="tool-card-icon" aria-hidden>{icon}</span>
        <div className="tool-card-titles">
          <span className="tool-card-title">{title}</span>
          {subtitle && <span className="tool-card-subtitle">{subtitle}</span>}
        </div>
        <span className={`tool-card-status tool-card-status--${status}`}>
          {STATUS_LABEL[status]}
        </span>
      </div>
      {children != null && <div className="tool-card-body">{children}</div>}
    </div>
  );
}
