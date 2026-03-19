import React, { useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import type { LucideIcon } from 'lucide-react';
import {
  MessageSquare, Search, Languages, PenLine, CheckSquare,
  Database, Zap, FileText, Wrench, File, Mic, Radio,
  Camera, Youtube, FlaskConical, Calculator, Terminal, Code2,
  Image, BarChart2, Bot, Cpu, ChevronDown, Settings, Crosshair,
  Presentation, Globe,
} from 'lucide-react';
import { useStore } from '../store';
import type { PanelId } from '../types';

interface RailItem {
  id: PanelId;
  label: string;
  Icon: LucideIcon;
  group?: 'chat' | 'tools' | 'ai';
}

const GROUP_LABELS: Record<string, string> = {
  chat: 'CHAT',
  tools: 'TOOLS',
  ai: 'AI',
};

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
  { id: 'record', label: 'REC Note', Icon: Radio },
  { id: 'ocr', label: 'OCR', Icon: Camera },
  { id: 'youtube', label: 'YouTube', Icon: Youtube },
  { id: 'research', label: 'Research', Icon: FlaskConical },
  { id: 'math', label: 'Math', Icon: Calculator },
  { id: 'code', label: 'Code', Icon: Terminal },
  { id: 'artifacts', label: 'Artifacts', Icon: Code2 },
  { id: 'image', label: 'Image', Icon: Image },
  { id: 'compare', label: 'Compare', Icon: BarChart2 },
  { id: 'capture', label: 'Capture', Icon: Crosshair },
  { id: 'slides', label: 'Slides', Icon: Presentation },
  { id: 'webcreator', label: 'Web Creator', Icon: Globe },
  { id: 'agent', label: 'Agent', Icon: Bot },
  { id: 'models', label: 'Models', Icon: Cpu },
  { id: 'settings', label: 'Settings', Icon: Settings },
];

// Group label rendered between groups
function GroupLabel({ text }: { text: string }) {
  return (
    <div className="rail-group-label">
      {text}
    </div>
  );
}

export default function Rail() {
  const { activePanel, setPanel, moreOpen, setMoreOpen } = useStore();
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{ label: string; top: number; right: number } | null>(null);

  const showTooltip = useCallback((label: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({ label, top: rect.top + rect.height / 2, right: window.innerWidth - rect.left + 8 });
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
        title={item.label}
        className={`rail-btn rail-icon-btn ${isActive ? 'rail-active' : ''}`}
      >
        <item.Icon
          size={18}
          strokeWidth={isActive ? 2.2 : 1.5}
          style={{
            transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s',
            transform: isHovered && !isActive ? 'scale(1.1)' : 'scale(1)',
          }}
        />
      </button>
    );
  };

  // Build primary items with group labels
  const primaryWithLabels: React.ReactNode[] = [];
  let lastGroup: string | undefined;
  for (const item of PRIMARY_ITEMS) {
    if (item.group && item.group !== lastGroup) {
      primaryWithLabels.push(
        <GroupLabel key={`lbl-${item.group}`} text={GROUP_LABELS[item.group] || item.group} />
      );
    }
    primaryWithLabels.push(btn(item));
    lastGroup = item.group;
  }

  return (
    <>
    <nav className="rail-nav">
      {/* Primary items with group labels */}
      <div className="rail-section">
        {primaryWithLabels}
      </div>

      {/* More toggle */}
      <div className="rail-section">
        <GroupLabel text="MORE" />
        <button
          onClick={() => setMoreOpen(!moreOpen)}
          onMouseEnter={() => setHoveredId('__more')}
          onMouseLeave={() => setHoveredId(null)}
          className={`rail-btn rail-icon-btn ${moreOpen ? 'rail-active' : ''}`}
          title="More"
        >
          <ChevronDown
            size={18}
            strokeWidth={1.5}
            style={{
              transform: moreOpen ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
            }}
          />
        </button>
      </div>

      {/* Extra items with slide-in */}
      {moreOpen && (
        <div className="rail-extras">
          {EXTRA_ITEMS.map(btn)}
        </div>
      )}

      {/* Spacer */}
      <div style={{ flex: 1 }} />
    </nav>

    {/* Tooltip — portaled to body to escape rail overflow clipping */}
    {tooltip && createPortal(
      <div
        className="rail-tooltip"
        style={{
          position: 'fixed',
          right: tooltip.right,
          top: tooltip.top,
          transform: 'translateY(-50%)',
          zIndex: 9999,
        }}
      >
        {tooltip.label}
      </div>,
      document.body
    )}
    </>
  );
}
