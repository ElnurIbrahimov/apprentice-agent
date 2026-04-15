import React, { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { LucideIcon } from 'lucide-react';
import {
  MessageSquare, Search, Languages, CheckSquare,
  Database, Zap, FileText, Wrench, File, Mic, Radio,
  Camera, Youtube, FlaskConical, Calculator, Code2,
  Image, BarChart2, Bot, Cpu, Settings, Crosshair,
  Brain, MousePointerClick, Headphones, Settings2, Hand, Plug,
  // SOTA upgrade icons
  GitBranch, Users, Activity, Sparkles, Flame, Dna,
  HardDrive, CalendarDays, Layers, Mail, Rss, Share2,
} from 'lucide-react';
import { useStore } from '../store';
import type { PanelId } from '../types';

/* ------------------------------------------------------------------ */
/*  Category tab definitions                                          */
/* ------------------------------------------------------------------ */

interface SubPanel {
  id: PanelId;
  label: string;
  Icon: LucideIcon;
}

interface CategoryTab {
  key: string;
  label: string;
  Icon: LucideIcon;
  panels: SubPanel[];
}

const CATEGORIES: CategoryTab[] = [
  {
    key: 'chat',
    label: 'Chat',
    Icon: MessageSquare,
    panels: [
      { id: 'chat', label: 'Chat', Icon: MessageSquare },
      { id: 'ask', label: 'Ask', Icon: Zap },
      { id: 'search', label: 'Search', Icon: Search },
      { id: 'translate', label: 'Translate', Icon: Languages },
    ],
  },
  {
    key: 'research',
    label: 'Research',
    Icon: FlaskConical,
    panels: [
      { id: 'research', label: 'Research', Icon: FlaskConical },
      { id: 'summary', label: 'Summary', Icon: FileText },
      { id: 'youtube', label: 'YouTube', Icon: Youtube },
      { id: 'pdf', label: 'PDF', Icon: File },
      { id: 'wisebase', label: 'Wisebase', Icon: Database },
      { id: 'math', label: 'Math', Icon: Calculator },
    ],
  },
  {
    key: 'page',
    label: 'Page',
    Icon: MousePointerClick,
    panels: [
      { id: 'capture', label: 'Capture', Icon: Crosshair },
      { id: 'ocr', label: 'OCR', Icon: Camera },
      { id: 'grammar', label: 'Grammar', Icon: CheckSquare },
      { id: 'image', label: 'Image', Icon: Image },
    ],
  },
  {
    key: 'media',
    label: 'Media',
    Icon: Headphones,
    panels: [
      { id: 'voice', label: 'Voice', Icon: Mic },
      { id: 'record', label: 'Record', Icon: Radio },
    ],
  },
  {
    key: 'tools',
    label: 'Tools',
    Icon: Wrench,
    panels: [
      { id: 'tools', label: 'Tools', Icon: Wrench },
      { id: 'compare', label: 'Compare', Icon: BarChart2 },
      { id: 'artifacts', label: 'Artifacts', Icon: Code2 },
      { id: 'calendar', label: 'Calendar', Icon: CalendarDays },
      { id: 'flashcards', label: 'Flashcards', Icon: Layers },
      { id: 'email', label: 'Email', Icon: Mail },
      { id: 'feed', label: 'Feed', Icon: Rss },
      { id: 'share', label: 'Share', Icon: Share2 },
    ],
  },
  {
    key: 'aura',
    label: 'Aura',
    Icon: Brain,
    panels: [
      { id: 'aura-status', label: 'Mind', Icon: Brain },
      { id: 'memory-browser', label: 'Memory', Icon: HardDrive },
      { id: 'activity', label: 'Activity', Icon: Activity },
      { id: 'hands', label: 'Hands', Icon: Hand },
      { id: 'agent', label: 'Agent', Icon: Bot },
    ],
  },
  {
    key: 'intelligence',
    label: 'Intelligence',
    Icon: Sparkles,
    panels: [
      { id: 'reasoning-tree', label: 'Reasoning', Icon: GitBranch },
      { id: 'multi-agent', label: 'Council', Icon: Users },
      { id: 'context-heatmap', label: 'Context', Icon: Flame },
      { id: 'bandit', label: 'Bandit', Icon: BarChart2 },
      { id: 'evolution', label: 'Evolution', Icon: Dna },
    ],
  },
  {
    key: 'settings',
    label: 'Settings',
    Icon: Settings2,
    panels: [
      { id: 'models', label: 'Models', Icon: Cpu },
      { id: 'mcp', label: 'Connections', Icon: Plug },
      { id: 'settings', label: 'Settings', Icon: Settings },
    ],
  },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

/** Return the category key that owns a given panel id */
function categoryForPanel(panelId: PanelId): string {
  for (const cat of CATEGORIES) {
    if (cat.panels.some((p) => p.id === panelId)) return cat.key;
  }
  return CATEGORIES[0].key;
}

/* ------------------------------------------------------------------ */
/*  Component                                                         */
/* ------------------------------------------------------------------ */

export default function Rail() {
  const { activePanel, setPanel } = useStore();

  // Which tab is currently expanded — auto-follows activePanel's parent
  const [expandedTab, setExpandedTab] = useState<string>(categoryForPanel(activePanel));

  // Sync expanded tab when activePanel changes externally
  useEffect(() => {
    const ownerKey = categoryForPanel(activePanel);
    if (ownerKey !== expandedTab) setExpandedTab(ownerKey);
  }, [activePanel]);

  // Tooltip state (portal-based)
  const [tooltip, setTooltip] = useState<{ label: string; top: number; right: number } | null>(null);

  const showTooltip = useCallback((label: string, e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    setTooltip({ label, top: rect.top + rect.height / 2, right: window.innerWidth - rect.left + 8 });
  }, []);

  const hideTooltip = useCallback(() => setTooltip(null), []);

  const handleTabClick = useCallback(
    (key: string) => {
      if (expandedTab === key) {
        // Already expanded — navigate to the first panel in this category
        const cat = CATEGORIES.find(c => c.key === key);
        if (cat?.panels[0]) setPanel(cat.panels[0].id);
        return;
      }
      setExpandedTab(key);
    },
    [expandedTab, setPanel],
  );

  return (
    <>
      <nav className="rail-nav" aria-label="Panel navigation">
        <div className="rail-section">
          {CATEGORIES.map((cat) => {
            const isExpanded = expandedTab === cat.key;
            const catOwnsActive = categoryForPanel(activePanel) === cat.key;

            return (
              <React.Fragment key={cat.key}>
                {/* ---- Category tab button ---- */}
                <button
                  onClick={() => handleTabClick(cat.key)}
                  onMouseEnter={(e) => showTooltip(cat.label, e)}
                  onMouseLeave={hideTooltip}
                  title={cat.label}
                  aria-label={cat.label}
                  aria-expanded={isExpanded}
                  className={`rail-btn rail-icon-btn ${catOwnsActive ? 'rail-active' : ''}`}
                  style={{ width: 36, height: 36 }}
                >
                  <cat.Icon
                    size={18}
                    strokeWidth={catOwnsActive ? 2.2 : 1.5}
                    style={{
                      transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s',
                    }}
                  />
                </button>

                {/* ---- Category label ---- */}
                {isExpanded && (
                  <div className="rail-group-label">{cat.label.toUpperCase()}</div>
                )}

                {/* ---- Expanded sub-panels ---- */}
                {isExpanded && (
                  <div className="rail-extras">
                    {cat.panels.map((panel) => {
                      const isActive = activePanel === panel.id;
                      return (
                        <button
                          key={panel.id}
                          onClick={() => setPanel(panel.id)}
                          onMouseEnter={(e) => showTooltip(panel.label, e)}
                          onMouseLeave={hideTooltip}
                          title={panel.label}
                          aria-label={panel.label}
                          aria-current={isActive ? 'page' : undefined}
                          className={`rail-btn rail-icon-btn ${isActive ? 'rail-active' : ''}`}
                          style={{ width: 32, height: 32, marginLeft: 4 }}
                        >
                          <panel.Icon
                            size={16}
                            strokeWidth={isActive ? 2.2 : 1.5}
                            style={{
                              transition: 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1), color 0.2s',
                            }}
                          />
                        </button>
                      );
                    })}
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />
      </nav>

      {/* Tooltip — portaled to body to escape rail overflow clipping */}
      {tooltip &&
        createPortal(
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
          document.body,
        )}
    </>
  );
}
