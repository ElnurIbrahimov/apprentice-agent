import React, { useState, useCallback } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  MessageSquare, Search, Languages, PenLine, CheckSquare,
  Database, Zap, FileText, Wrench, File, Mic,
  Camera, Youtube, FlaskConical, Calculator, Code2,
  Image, BarChart2, Bot, Cpu, ChevronDown,
} from 'lucide-react';
import { useStore } from '../store';
import type { PanelId } from '../types';

interface RailItem {
  id: PanelId;
  label: string;
  Icon: LucideIcon;
  group?: 'chat' | 'tools' | 'ai';
}

const PRIMARY_ITEMS: RailItem[] = [
  { id: 'chat', label: 'Chat', Icon: MessageSquare, group: 'chat' },
  { id: 'search', label: 'Search', Icon: Search, group: 'chat' },
  { id: 'translate', label: 'Translate', Icon: Languages, group: 'chat' },
  { id: 'write', label: 'Write', Icon: PenLine, group: 'tools' },
  { id: 'grammar', label: 'Grammar', Icon: CheckSquare, group: 'tools' },
  { id: 'wisebase', label: 'Wisebase', Icon: Database, group: 'tools' },
  { id: 'ask', label: 'Ask', Icon: Zap, group: 'ai' },
  { id: 'summary', label: 'Summary', Icon: FileText, group: 'ai' },
  { id: 'tools', label: 'Tools', Icon: Wrench, group: 'ai' },
];

const EXTRA_ITEMS: RailItem[] = [
  { id: 'pdf', label: 'PDF', Icon: File },
  { id: 'voice', label: 'Voice', Icon: Mic },
  { id: 'ocr', label: 'OCR', Icon: Camera },
  { id: 'youtube', label: 'YouTube', Icon: Youtube },
  { id: 'research', label: 'Research', Icon: FlaskConical },
  { id: 'math', label: 'Math', Icon: Calculator },
  { id: 'artifacts', label: 'Artifacts', Icon: Code2 },
  { id: 'image', label: 'Image', Icon: Image },
  { id: 'compare', label: 'Compare', Icon: BarChart2 },
  { id: 'agent', label: 'Agent', Icon: Bot },
  { id: 'models', label: 'Models', Icon: Cpu },
];

// Group separator between different groups
function GroupSeparator() {
  return (
    <div
      style={{
        height: 1,
        margin: '4px 10px',
        background: 'linear-gradient(90deg, transparent, var(--b1), transparent)',
      }}
    />
  );
}

export default function Rail() {
  const { activePanel, setPanel, moreOpen, setMoreOpen } = useStore();
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ label: string; top: number } | null>(null);

  const showTooltip = useCallback((label: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({ label, top: rect.top + rect.height / 2 });
  }, []);

  const hideTooltip = useCallback(() => setTooltip(null), []);

  const btn = (item: RailItem) => {
    const isActive = activePanel === item.id;
    const isHovered = hoveredId === item.id;

    return (
      <button
        key={item.id}
        onClick={() => setPanel(item.id)}
        onMouseEnter={(e) => { setHoveredId(item.id); showTooltip(item.label, e); }}
        onMouseLeave={() => { setHoveredId(null); hideTooltip(); }}
        className="rail-btn flex flex-col items-center justify-center w-full py-2 gap-0.5 relative"
        style={{
          background: isActive
            ? 'var(--pg)'
            : isHovered
              ? 'rgba(124, 58, 237, 0.06)'
              : 'transparent',
          borderLeft: isActive
            ? '2px solid var(--p)'
            : '2px solid transparent',
          color: isActive ? 'var(--pl)' : isHovered ? 'var(--pl2)' : 'var(--mu)',
          cursor: 'pointer',
          fontFamily: 'inherit',
          minHeight: 48,
          border: 'none',
          borderRight: 'none',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          transform: isHovered && !isActive ? 'scale(1.04)' : 'scale(1)',
        }}
      >
        {/* Active indicator bar */}
        <span
          className="absolute left-0 top-1/2"
          style={{
            width: 2,
            height: isActive ? '60%' : '0%',
            background: 'var(--p)',
            borderRadius: '0 2px 2px 0',
            transform: 'translateY(-50%)',
            transition: 'height 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
            boxShadow: isActive ? '0 0 8px rgba(124, 58, 237, 0.5)' : 'none',
          }}
        />
        {/* Hover glow */}
        {isHovered && !isActive && (
          <span
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(circle at center, rgba(124,58,237,0.08) 0%, transparent 70%)',
            }}
          />
        )}
        <item.Icon
          size={16}
          strokeWidth={isActive ? 2 : 1.5}
          style={{ transition: 'transform 0.2s', transform: isHovered ? 'scale(1.1)' : 'scale(1)' }}
        />
        <span style={{ fontSize: '9px', letterSpacing: '0.02em', marginTop: 1, transition: 'color 0.2s' }}>
          {item.label}
        </span>
      </button>
    );
  };

  // Build primary items with group separators
  const primaryWithSeparators: React.ReactNode[] = [];
  let lastGroup: string | undefined;
  for (const item of PRIMARY_ITEMS) {
    if (lastGroup && item.group !== lastGroup) {
      primaryWithSeparators.push(<GroupSeparator key={`sep-${lastGroup}-${item.group}`} />);
    }
    primaryWithSeparators.push(btn(item));
    lastGroup = item.group;
  }

  return (
    <nav
      className="flex flex-col flex-shrink-0 overflow-y-auto overflow-x-hidden rail-nav"
      style={{
        width: 54,
        background: 'var(--s1)',
        borderLeft: '1px solid var(--b1)',
        scrollbarWidth: 'thin',
      }}
    >
      {/* Primary items with group separators */}
      {primaryWithSeparators}

      {/* Separator before More */}
      <GroupSeparator />

      {/* More toggle */}
      <button
        onClick={() => setMoreOpen(!moreOpen)}
        onMouseEnter={() => setHoveredId('__more')}
        onMouseLeave={() => setHoveredId(null)}
        className="rail-btn flex flex-col items-center justify-center w-full py-2 gap-0.5 relative"
        style={{
          background: hoveredId === '__more' ? 'rgba(124, 58, 237, 0.06)' : 'transparent',
          borderLeft: '2px solid transparent',
          color: moreOpen ? 'var(--pl)' : 'var(--mu)',
          cursor: 'pointer',
          fontFamily: 'inherit',
          minHeight: 48,
          border: 'none',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        title="More"
      >
        <ChevronDown
          size={16}
          strokeWidth={1.5}
          style={{
            transform: moreOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />
        <span style={{ fontSize: '9px', letterSpacing: '0.02em', marginTop: 1 }}>
          More
        </span>
      </button>

      {/* Extra items with slide-in */}
      {moreOpen && (
        <div className="rail-extras">
          {EXTRA_ITEMS.map(btn)}
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="rail-tooltip"
          style={{
            position: 'fixed',
            left: 58,
            top: tooltip.top,
            transform: 'translateY(-50%)',
            zIndex: 9999,
          }}
        >
          {tooltip.label}
        </div>
      )}
    </nav>
  );
}
