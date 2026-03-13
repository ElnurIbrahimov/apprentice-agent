import React from 'react';
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
  extra?: boolean;
}

const PRIMARY_ITEMS: RailItem[] = [
  { id: 'chat', label: 'Chat', Icon: MessageSquare },
  { id: 'search', label: 'Search', Icon: Search },
  { id: 'translate', label: 'Translate', Icon: Languages },
  { id: 'write', label: 'Write', Icon: PenLine },
  { id: 'grammar', label: 'Grammar', Icon: CheckSquare },
  { id: 'wisebase', label: 'Wisebase', Icon: Database },
  { id: 'ask', label: 'Ask', Icon: Zap },
  { id: 'summary', label: 'Summary', Icon: FileText },
  { id: 'tools', label: 'Tools', Icon: Wrench },
];

const EXTRA_ITEMS: RailItem[] = [
  { id: 'pdf', label: 'PDF', Icon: File, extra: true },
  { id: 'voice', label: 'Voice', Icon: Mic, extra: true },
  { id: 'ocr', label: 'OCR', Icon: Camera, extra: true },
  { id: 'youtube', label: 'YouTube', Icon: Youtube, extra: true },
  { id: 'research', label: 'Research', Icon: FlaskConical, extra: true },
  { id: 'math', label: 'Math', Icon: Calculator, extra: true },
  { id: 'artifacts', label: 'Artifacts', Icon: Code2, extra: true },
  { id: 'image', label: 'Image', Icon: Image, extra: true },
  { id: 'compare', label: 'Compare', Icon: BarChart2, extra: true },
  { id: 'agent', label: 'Agent', Icon: Bot, extra: true },
  { id: 'models', label: 'Models', Icon: Cpu, extra: true },
];

export default function Rail() {
  const { activePanel, setPanel, moreOpen, setMoreOpen } = useStore();

  const btn = (item: RailItem) => (
    <button
      key={item.id}
      onClick={() => setPanel(item.id)}
      className="flex flex-col items-center justify-center w-full py-2 gap-0.5 transition-all duration-150 relative"
      style={{
        background: activePanel === item.id ? 'var(--pg)' : 'transparent',
        borderLeft: activePanel === item.id ? '2px solid var(--p)' : '2px solid transparent',
        color: activePanel === item.id ? 'var(--pl)' : 'var(--mu)',
        cursor: 'pointer',
        fontFamily: 'inherit',
        minHeight: 48,
      }}
      title={item.label}
    >
      <item.Icon size={16} strokeWidth={activePanel === item.id ? 2 : 1.5} />
      <span style={{ fontSize: '9px', letterSpacing: '0.02em', marginTop: 1 }}>
        {item.label}
      </span>
    </button>
  );

  return (
    <nav
      className="flex flex-col flex-shrink-0 overflow-y-auto overflow-x-hidden"
      style={{
        width: 54,
        background: 'var(--s1)',
        borderLeft: '1px solid var(--b1)',
      }}
    >
      {/* Primary items */}
      {PRIMARY_ITEMS.map(btn)}

      {/* More toggle */}
      <button
        onClick={() => setMoreOpen(!moreOpen)}
        className="flex flex-col items-center justify-center w-full py-2 gap-0.5 transition-all duration-150"
        style={{
          background: 'transparent',
          borderLeft: '2px solid transparent',
          color: 'var(--mu)',
          cursor: 'pointer',
          fontFamily: 'inherit',
          minHeight: 48,
        }}
        title="More"
      >
        <ChevronDown
          size={16}
          strokeWidth={1.5}
          style={{
            transform: moreOpen ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s ease',
          }}
        />
        <span style={{ fontSize: '9px', letterSpacing: '0.02em', marginTop: 1 }}>
          More
        </span>
      </button>

      {/* Extra items */}
      {moreOpen && EXTRA_ITEMS.map(btn)}

      {/* Spacer */}
      <div className="flex-1" />
    </nav>
  );
}
